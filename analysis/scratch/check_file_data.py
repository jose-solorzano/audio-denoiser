import torch
import torchaudio

if __name__ == '__main__':
    in_audio_file = '/data/audio-example-1-out.wav'
    waveform, sample_rate = torchaudio.load(in_audio_file)
    print('Waveform: ', waveform.shape)
    print('Sample rate: ', sample_rate)
    print('Waveform mean: ', torch.mean(waveform).item())
    print('Waveform std: ', torch.std(waveform).item())
    print("Data type:", waveform.dtype)

    available_backends = torchaudio.list_audio_backends()
    print("Available audio backends:", available_backends)
