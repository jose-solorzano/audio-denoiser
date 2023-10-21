import numpy as np
import torch
import torchaudio

from audio_denoiser.AudioDenoiser import AudioDenoiser
from helpers.data_helper import get_file_paths


if __name__ == '__main__':
    rnd = np.random.RandomState(1)
    dataset_path_1 = '/kaggle/input/120h-spanish-speech'
    dataset_path_2 = '/kaggle/input/arabic-natural-audio-dataset'
    dataset_path_3 = '/kaggle/input/speaker-recognition-audio-dataset'
    audio_paths_1 = get_file_paths(dataset_path_1, '*.wav')
    audio_paths_2 = get_file_paths(dataset_path_2, '*.wav')
    audio_paths_3 = get_file_paths(dataset_path_3, '*.wav')
    audio_paths = audio_paths_1 + audio_paths_2 + audio_paths_3
    print(f'Found {len(audio_paths)} audio files.')
    rnd.shuffle(audio_paths)
    audio_paths = audio_paths[:100]
    waveforms = []
    for audio_path in audio_paths:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torch.sum(waveform, dim=0, keepdim=True)
        waveforms.append(waveform.flatten())
    waveform_t = torch.cat(waveforms)
    td = AudioDenoiser._trimmed_dev(waveform_t)
    print(f'Trimmed dev: {td}')

