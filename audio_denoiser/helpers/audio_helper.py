from typing import Optional, Union

import torch
import torchaudio.transforms as T


def create_spectrogram(waveform: torch.Tensor, n_fft: int, hop_length: int = None, power=2,
                       n_frames: Optional[int] = None, device: Union[torch.device, str] = None):
    if hop_length is None:
        hop_length = n_fft // 2
    spectrogram_transform = T.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True,
                                          power=power)
    if device is not None:
        spectrogram_transform = spectrogram_transform.to(device)
    spectrogram = spectrogram_transform(waveform)
    if n_frames is not None:
        spectrogram = spectrogram[:, :, :n_frames]
    return spectrogram


def reconstruct_from_spectrogram(spectrogram: torch.Tensor, num_iterations=100,
                                 power=2, device: Union[torch.device, str] = None):
    _, half_fft, _ = spectrogram.shape
    n_fft = (half_fft - 1) * 2
    transform = T.GriffinLim(n_fft=n_fft, n_iter=num_iterations, rand_init=False, power=power)
    if device is not None:
        transform = transform.to(device)
    return transform(spectrogram)
