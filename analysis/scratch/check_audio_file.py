import math

import torch
import torchaudio

from audio_denoiser.helpers.audio_helper import create_spectrogram, reconstruct_from_spectrogram

if __name__ == '__main__':
    waveform, sample_rate = torchaudio.load('/data/audio-example-1.wav')
    print('Waveform: ', waveform.shape)
    print('Sample rate: ', sample_rate)
    # waveform shape: (num_channels, samples)
    nn_fft = 512
    hop_len = 256
    spectrogram = create_spectrogram(waveform, n_fft=nn_fft, hop_length=hop_len)
    expected_sp_len = math.ceil(waveform.size(1) / hop_len)
    print('Expected num frames: ', expected_sp_len)
    print('Spectrogram: ', spectrogram.shape)
    print('Spectrogram std: ', torch.std(spectrogram).item())
    # spectrogram: (num_channels, n_fft / 2, samples / hop_length)
    recovered_waveform = reconstruct_from_spectrogram(spectrogram, num_iterations=100)
    print('Recovered waveform: ', recovered_waveform.shape)
    torchaudio.save('/data/audio-example-1-reconstructed.wav', recovered_waveform, sample_rate=sample_rate)
    print('Saved recovered waveform.')

