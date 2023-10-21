import math

import torch
import torchaudio
from torch import nn

from audio_denoiser.helpers.audio_helper import create_spectrogram, reconstruct_from_spectrogram


if __name__ == '__main__':
    waveform, sample_rate = torchaudio.load('/data/audio-example-1.wav')
    print('Waveform: ', waveform.shape)
    print('Waveform std: ', torch.std(waveform).item())
    print('Sample rate: ', sample_rate)
    noise = torch.randn_like(waveform) * torch.std(waveform).item()
    noisy_waveform = waveform + noise
    torchaudio.save('/data/audio-example-1-add-noise.wav', noisy_waveform, sample_rate=sample_rate)
    n_fft = 512
    waveform_spectrogram = create_spectrogram(waveform, n_fft=n_fft)
    noisy_spectrogram = create_spectrogram(noisy_waveform, n_fft=n_fft)
    noise_spectrogram = create_spectrogram(noise, n_fft=n_fft)
    expected_noisy_spectrogram = noise_spectrogram + waveform_spectrogram
    loss_fn = nn.MSELoss()
    ns_loss = loss_fn(noisy_spectrogram, expected_noisy_spectrogram).item()
    print(f'NS loss: {ns_loss:.4g}')
    ns_var = torch.var(expected_noisy_spectrogram).item()
    rel_var = ns_loss / ns_var
    print(f'Relative loss: {rel_var:.4g}')
    reduced_spectrogram = torch.clamp(noisy_spectrogram - noise_spectrogram, min=0)
    denoised_waveform = reconstruct_from_spectrogram(reduced_spectrogram)
    torchaudio.save('/data/audio-example-1-remove-noise.wav', denoised_waveform, sample_rate=sample_rate)
    print('Saved altered waveform.')
