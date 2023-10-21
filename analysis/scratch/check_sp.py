import math

import torch
import torchaudio

from audio_denoiser.helpers.audio_helper import create_spectrogram, reconstruct_from_spectrogram

if __name__ == '__main__':
    ws = (1, 512 * 3)
    waveform = torch.randn(ws)
    print('Waveform: ', waveform.shape)
    nn_fft = 512
    hop_len = 256
    spectrogram = create_spectrogram(waveform, n_fft=nn_fft, hop_length=hop_len)
    expected_sp_len = math.ceil((waveform.size(1) + 1) / hop_len)
    print('Expected num frames: ', expected_sp_len)
    print('Spectrogram: ', spectrogram.shape)

