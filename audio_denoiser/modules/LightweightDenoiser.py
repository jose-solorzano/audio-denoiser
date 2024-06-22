import math
from dataclasses import dataclass

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn


@dataclass
class LightweightDenoiserConfig:
    frame_size: int = 512
    avg_pool_ks: int = 32
    tag: str = None


class LightweightDenoiser(nn.Module, PyTorchModelHubMixin):
    frame_pos: torch.Tensor

    def __init__(self, config: LightweightDenoiserConfig):
        super().__init__()
        if isinstance(config, dict):
            config = LightweightDenoiserConfig(**config)
        self.config = config
        self.frame_size = config.frame_size
        self.avg_pool_ks = config.avg_pool_ks
        self.abs_fft_mean = math.sqrt(self.frame_size) * 0.886
        self.abs_fft_scale = self.abs_fft_mean / 2
        self.pool = nn.AvgPool1d(kernel_size=self.avg_pool_ks)
        self.frame_pool_size = self.frame_size // self.avg_pool_ks
        self.pool_model = nn.Sequential(
            nn.Linear(self.avg_pool_ks, self.avg_pool_ks * 5),
            nn.ELU(),
            nn.Linear(self.avg_pool_ks * 5, 1),
        )
        hidden = (self.frame_pool_size + 1) * 5
        self.head1 = nn.Sequential(
            nn.Linear(self.frame_pool_size, hidden, bias=False),
            nn.ELU(),
        )
        self.head2 = nn.Sequential(
            nn.Linear(1, hidden, bias=True),
            nn.ELU(),
        )
        self.tail = nn.Linear(hidden, 1)
        frame_range = torch.arange(0, self.frame_size)
        frame_pos = (frame_range / self.frame_size) * 2 - 1.0
        frame_pos = frame_pos.unsqueeze(1)
        self.register_buffer("frame_pos", frame_pos)

    def forward(self, frame_fft: torch.Tensor):
        # frame_fft: (batch_size, 2, frame_size)
        # First channel is real, second is complex
        frame_fft_abs = torch.sqrt(torch.sum(frame_fft.pow(2), dim=1))
        frame_fft_abs = (frame_fft_abs - self.abs_fft_mean) / self.abs_fft_scale
        # frame_fft_abs: (batch_size, frame_size)
        batch_size = frame_fft_abs.size(0)
        frame_fft_abs = frame_fft_abs.view(batch_size * self.frame_pool_size, self.avg_pool_ks)
        frame_pool = self.pool_model(frame_fft_abs)
        # frame_pool: (batch_size * frame_pool_size, 1)
        frame_pool = frame_pool.view(batch_size, self.frame_pool_size)
        head_out_1 = self.head1(frame_pool)
        # head_out_1: (batch_size, 24)
        head_out_2 = self.head2(self.frame_pos)
        # head_out_2: (frame_size, 24)
        head_out = head_out_1[:, None, :] + head_out_2[None, :, :]
        # head_out: (batch_size, frame_size, 24)
        preserve_f = self.tail(head_out)
        # preserve_f: (batch_size * frame_size, 1)
        preserve_f = preserve_f.view((batch_size, self.frame_size))
        preserve_f = torch.clamp(preserve_f, min=0, max=1)
        return preserve_f


if __name__ == '__main__':
    _config = LightweightDenoiserConfig()
    x = torch.randn((500, 2, _config.frame_size))
    _m = LightweightDenoiser(_config)
    _y = _m(x)
    print(_y.shape)
    print(torch.std(_y))



