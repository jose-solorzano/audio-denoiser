from typing import List, Any

import torch
from torch import nn


class View(nn.Module):
    def __init__(self, *shape: Any, contiguous: bool = False):
        super().__init__()
        self.contiguous = contiguous
        self.params = shape

    def forward(self, x: torch.Tensor):
        if self.contiguous:
            return x.contiguous().view(*self.params)
        else:
            return x.view(*self.params)
