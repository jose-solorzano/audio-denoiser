import math
from typing import List
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from audio_denoiser.modules.AudioNoiseModel import AudioNoiseModel
from audio_denoiser.modules.SpectrogramScaler import SpectrogramScaler
from data.AudioFileDataset import AudioFileDataset
from training.SchedulingOptimizer import SchedulingOptimizer


class AudioDenoiserTrainer:
    def __init__(self, rnd: np.random.RandomState, train_audio_paths: List[str], valid_audio_paths: List[str],
                 norm_ambient_noise: List[torch.Tensor],
                 device: torch.device, batch_size: int,
                 learning_rate: float, in_channels: int = 257, num_frames: int = 32):
        self.norm_ambient_noise = norm_ambient_noise
        self.in_channels = in_channels
        self.valid_audio_paths = valid_audio_paths
        self.train_audio_paths = train_audio_paths
        self.device = device
        self.rnd = rnd
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_frames = num_frames
        # Loss function and optimizer
        self.mseLoss = nn.MSELoss()

    def train(self, num_epochs: int,
              n_files_for_scaler: int = 100, n_files_per_ds: int = 2000) -> AudioNoiseModel:
        audio_paths_copy = list(self.train_audio_paths)
        self.rnd.shuffle(audio_paths_copy)
        scaler_paths = audio_paths_copy[:n_files_for_scaler]
        print(f'Training scaler with {len(scaler_paths)} audio files...', flush=True)
        scaler = SpectrogramScaler.train_scaler(scaler_paths, self.norm_ambient_noise)
        n_fft = (self.in_channels - 1) * 2
        num_frames = 32
        config = dict(
            scaler=scaler.to_dict(),
            in_channels=self.in_channels,
            n_fft=n_fft,
            num_frames=num_frames,
        )
        model = AudioNoiseModel(config)
        model = model.to(self.device)
        model.train()  # Set the model to training mode
        expected_ds_size = min(len(self.train_audio_paths), n_files_per_ds)
        total_steps = math.ceil(expected_ds_size / self.batch_size) * num_epochs
        optimizer = SchedulingOptimizer(model.parameters(), lr=self.learning_rate, total_steps=total_steps)
        train_loader = None
        for epoch in range(num_epochs):
            if train_loader is None or epoch % 10 == 0:
                print('Building dataset...')
                file_paths = list(self.train_audio_paths)
                self.rnd.shuffle(file_paths)
                train_dataset = AudioFileDataset(self.rnd, file_paths[:n_files_per_ds], self.norm_ambient_noise,
                                                 scaler=scaler,
                                                 n_fft=n_fft, num_frames=num_frames)
                ds_size = len(train_dataset)
                print(f'Training dataset size: {ds_size}')
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            running_loss = 0.0
            count = 0
            for i, (source_spect, target_spect) in enumerate(train_loader):
                source_spect = source_spect.to(self.device)
                target_spect = target_spect.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(source_spect)

                # Compute the loss
                loss = self.mseLoss(outputs, target_spect)

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                count += 1

            # Print the average loss for this epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / count:.4g}')
        return model

    def evaluate(self, model: AudioNoiseModel, num_epochs: int, n_files_per_ds: int = 2000):
        model.eval()
        for e in range(num_epochs):
            print(f'Validation epoch {e+1}...')
            total_mse = 0.0
            total_var = 0.0
            count = 0
            audio_paths_copy = list(self.valid_audio_paths)
            self.rnd.shuffle(audio_paths_copy)
            ds_audio_paths = audio_paths_copy[:n_files_per_ds]
            valid_dataset = AudioFileDataset(self.rnd, ds_audio_paths, self.norm_ambient_noise,
                                             scaler=model.scaler,
                                             n_fft=(self.in_channels - 1) * 2,
                                             num_frames=self.num_frames)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
            with torch.no_grad():
                for source_spect, target_spect in valid_loader:
                    source_spect = source_spect.to(self.device)
                    target_spect = target_spect.to(self.device)
                    # Forward pass
                    pred_spect = model(source_spect)
                    # Compute the loss
                    mse = self.mseLoss(pred_spect, target_spect)
                    variance = torch.var(target_spect).item()
                    total_mse += mse.item()
                    total_var += variance
                    count += 1
            # Calculate the average loss on the validation set
            avg_mse = total_mse / count
            rel_var = total_mse / total_var
            print(f'MSE: {avg_mse:.4g} - Relative variance: {rel_var:.4g}')
