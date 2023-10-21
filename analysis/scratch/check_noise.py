import math

import torch
import torchaudio


def add_noise(waveform: torch.Tensor, noise_level=0.5):
    sd = torch.std(waveform).item()
    noise_sd = noise_level * sd
    noise = torch.randn_like(waveform) * noise_sd
    return (waveform + noise) * sd / math.sqrt(sd ** 2 + noise_sd ** 2)


if __name__ == '__main__':
    waveform, sample_rate = torchaudio.load('/data/audio-example-1.wav')
    print('Waveform: ', waveform.shape)
    print('Waveform std: ', torch.std(waveform).item())
    print('Sample rate: ', sample_rate)
    noisy_waveform = add_noise(waveform)
    print('Noisy waveform std: ', torch.std(noisy_waveform).item())
    torchaudio.save('/data/audio-example-1-out.wav', noisy_waveform, sample_rate=sample_rate)
    print('Saved altered waveform.')
