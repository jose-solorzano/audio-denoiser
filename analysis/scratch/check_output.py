import torch

from audio_denoiser.AudioDenoiser import AudioDenoiser

if __name__ == '__main__':
    denoiser = AudioDenoiser()
    torch.manual_seed(1)
    model = denoiser.model
    x = (torch.randn((5, 257, 3)) + 3.0)
    y = model(x)
    print(y[0, :, 1])
    