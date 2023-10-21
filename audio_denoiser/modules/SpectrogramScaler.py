import os
from typing import List

import numpy as np
import torch
from torch import nn


class SpectrogramScaler(nn.Module):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.std = std
        self.mean = mean

    def to_dict(self):
        return dict(mean=self.mean, std=self.std)

    def __str__(self):
        return f'{self.__class__.__name__} {self.to_dict()}'

    @classmethod
    def from_dict(cls, d: dict):
        mean = d['mean']
        std = d['std']
        return SpectrogramScaler(mean, std)

    @staticmethod
    def train_scaler(audio_files: List[str], ambient_noise: List[torch.Tensor],
                     sample_rate=16000, n_fft=512) -> 'SpectrogramScaler':
        from data.AudioFileDataset import AudioFileDataset

        base_scaler = SpectrogramScaler(0, 1.0)
        rnd = np.random.RandomState(1)
        ds = AudioFileDataset(rnd, audio_files, ambient_noise, base_scaler, sample_rate=sample_rate, n_fft=n_fft)
        spectrogram_list = []
        for x, _ in ds:
            spectrogram_list.append(x)
        spectrogram_all = torch.cat(spectrogram_list)
        mean = torch.mean(spectrogram_all).item()
        std = torch.std(spectrogram_all).item()
        return SpectrogramScaler(mean, std)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return (spectrogram - self.mean) / self.std
