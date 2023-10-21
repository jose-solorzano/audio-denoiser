from typing import List

import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, *dims: int):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
