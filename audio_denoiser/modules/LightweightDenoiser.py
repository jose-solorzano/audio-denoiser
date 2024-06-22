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

    def forward(self, frame_fft: torch.Tensor):
        # frame_fft: (batch_size, 2, frame_size)
        # First channel is real, second is complex
        frame_fft_abs = torch.sqrt(torch.sum(frame_fft.pow(2), dim=1, keepdim=True))
        frame_fft_abs = (frame_fft_abs - self.abs_fft_mean) / self.abs_fft_scale
        preserve_f = (self.model(frame_fft_abs) + self.bias) / 2 + 0.5
        preserve_f = torch.clamp(preserve_f, min=0, max=1)
        return preserve_f.squeeze(1)


if __name__ == '__main__':
    _config = LightweightDenoiserConfig()
    x = torch.randn((500, 2, _config.frame_size))
    _m = LightweightDenoiser(_config)
    _y = _m(x)
    print(_y.shape)
    print(torch.std(_y))



