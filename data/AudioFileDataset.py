import logging
import math
from typing import List

import numpy as np
import torchaudio
import torch
from torch import nn
from torch.utils.data import Dataset

from audio_denoiser.modules.SpectrogramScaler import SpectrogramScaler
from audio_denoiser.helpers.audio_helper import create_spectrogram


class AudioFileDataset(Dataset):
    def __init__(self, rnd: np.random.RandomState, file_paths: List[str],
                 ambient_noise: List[torch.Tensor],
                 scaler: SpectrogramScaler,
                 num_frames: int = 32, sample_rate=16000, n_fft=512, max_noise_level=1.5):
        self.ambient_noise = ambient_noise
        self.max_noise_level = max_noise_level
        self.num_frames = num_frames
        self.rnd = rnd
        self.scaler = scaler
        self.file_paths = file_paths
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.audio_data = []
        self.hop_len = n_fft // 2
        self.expected_waveform_len = self.num_frames * (self.hop_len - 1) + n_fft
        for file_path in self.file_paths:
            try:
                waveform, wf_sample_rate = torchaudio.load(file_path)
            except RuntimeError:
                logging.exception(f'Failed to load audio file {file_path}')
                continue
            if wf_sample_rate != sample_rate:
                transform = torchaudio.transforms.Resample(orig_freq=wf_sample_rate, new_freq=sample_rate)
                waveform = transform(waveform)
            waveform = torch.sum(waveform, dim=0, keepdim=True)
            self.audio_data.append(waveform)

    def add_noise(self, waveform: torch.Tensor, p_gaussian=0.2):
        sd = torch.std(waveform).item()
        batch_size, _ = waveform.shape
        noise_level = torch.rand((batch_size, 1)) * self.max_noise_level
        noise_sd = noise_level * sd
        if self.rnd.uniform() < p_gaussian:
            noise = torch.randn_like(waveform) * noise_sd
        else:
            idx = self.rnd.randint(0, len(self.ambient_noise))
            an_tensor = self.ambient_noise[idx]
            _, waveform_ns = waveform.shape
            _, an_ns = an_tensor.shape
            assert an_ns >= waveform_ns, f'an_ns={an_ns}, waveform_ns={waveform_ns}'
            start = self.rnd.randint(0, an_ns - waveform_ns + 1)
            noise = an_tensor[:, start:start + waveform_ns] * noise_sd
        denominator = torch.sqrt(sd ** 2 + noise_sd ** 2) + 1e-10
        return (waveform + noise) * sd / denominator

    @staticmethod
    def sp_log(spectrogram: torch.Tensor, eps=0.01):
        return torch.log(spectrogram + eps)

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        waveform = self.audio_data[idx]
        # waveform: (1, num_samples,)
        waveform_len = waveform.size(1)
        expected_waveform_len = self.expected_waveform_len
        start = self.rnd.randint(0, 1 + max(0, waveform_len - expected_waveform_len))
        end = start + expected_waveform_len
        waveform = waveform[:, start:end]
        if waveform.size(1) < expected_waveform_len:
            waveform = nn.functional.pad(waveform, (0, expected_waveform_len - waveform.size(1)))
        spectrogram = create_spectrogram(waveform, n_fft=self.n_fft, hop_length=self.hop_len,
                                         n_frames=self.num_frames)
        num_sp_frames = spectrogram.size(2)
        assert num_sp_frames == self.num_frames, f'num_sp_frames={num_sp_frames}'
        noisy_waveform = self.add_noise(waveform)
        noisy_spectrogram = create_spectrogram(noisy_waveform, n_fft=self.n_fft, hop_length=self.hop_len,
                                               n_frames=self.num_frames)
        log_noisy_sp = self.sp_log(noisy_spectrogram)
        scaled_log_noisy_sp = self.scaler(log_noisy_sp)
        log_sp = self.sp_log(spectrogram)
        target_noise = log_noisy_sp - log_sp
        return scaled_log_noisy_sp.detach().squeeze(0), target_noise.detach().squeeze(0),
