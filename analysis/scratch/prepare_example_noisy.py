import math
import os

import torch
import torchaudio


def add_noise(waveform: torch.Tensor, noise_level=0.5):
    sd = torch.std(waveform).item()
    noise_sd = noise_level * sd
    noise = torch.randn_like(waveform) * noise_sd
    return (waveform + noise) * sd / math.sqrt(sd ** 2 + noise_sd ** 2)


if __name__ == '__main__':
    dir_name = '/data/audio'
    files = ["Phrase_de_Neil_Armstrong.mp3", "Nixon_resignation_audio_with_buzz_removed.mp3"]
    scales = [1.0, 0.03]
    noise_levels = [0.3, 1.2]
    for i in range(len(files)):
        file = files[i]
        file_path = os.path.join(dir_name, file)
        waveform, sample_rate = torchaudio.load(file_path)
        print('std: ', torch.std(waveform).item())
        waveform = waveform * scales[i]
        waveform = add_noise(waveform, noise_level=noise_levels[i])
        out_file = os.path.join(dir_name, f'noisy-example-{i+1}.wav')
        torchaudio.save(out_file, waveform, sample_rate=sample_rate)
        print(f'Saved {out_file}')


