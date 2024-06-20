import math
from dataclasses import dataclass

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn


@dataclass
class LightweightDenoiserConfig:
    frame_size: int = 512
    tag: str = None


class LightweightDenoiser(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: LightweightDenoiserConfig):
        super().__init__()
        if isinstance(config, dict):
            config = LightweightDenoiserConfig(**config)
        self.config = config
        self.frame_size = config.frame_size
        self.abs_fft_mean = math.sqrt(self.frame_size) * 0.886
        self.abs_fft_scale = self.abs_fft_mean / 2
        self.model = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=1, padding=0),
            nn.ELU(),
            nn.Conv1d(8, 1, kernel_size=1, padding=0, bias=False),
        )
        bias_param = torch.zeros((1, self.frame_size))
        self.bias = nn.Parameter(bias_param)

    def forward(self, frame: torch.Tensor):
        # frame: (batch_size, frame_size)
        frame_fft = torch.fft.fft(frame, dim=1)
        frame_fft_abs = torch.abs(frame_fft)
        frame_fft_abs = (frame_fft_abs - self.abs_fft_mean) / self.abs_fft_scale
        frame_fft_abs = frame_fft_abs[:, None, :]
        preserve_f = (self.model(frame_fft_abs) + self.bias) / 2 + 0.5
        preserve_f = torch.clamp(preserve_f, min=0, max=1)
        frame_fft = frame_fft * preserve_f
        out_frame = torch.fft.ifft(frame_fft).real
        return out_frame


if __name__ == '__main__':
    _config = LightweightDenoiserConfig()
    x = torch.randn((500, _config.frame_size))
    _m = LightweightDenoiser(_config)
    _y = _m(x)
    print(_y.shape)
    print(torch.std(_y))



