import unittest
import torch
import torch.nn.functional as F
from audio_denoiser.helpers.torch_helper import unfold_2d, fold_2d


class TestTorchHelper(unittest.TestCase):
    def test_reverse_unfold_2d(self):
        height = 5
        width = 4
        x = torch.randn(3, height, width)
        x_unfold = unfold_2d(x, patch_size=2, step_size=2)
        x_refold = fold_2d(x_unfold, width, height)
        self.assertTrue(torch.all(torch.isclose(x, x_refold)))

    def test_padded_unfold_step_1(self):
        height = 4
        width = 6
        patch_size = 5
        channels = 3
        pad_size = patch_size // 2
        x = torch.randn(channels, height, width)
        x = F.pad(x, (pad_size, pad_size, pad_size, pad_size))
        x_unfold = unfold_2d(x, patch_size=patch_size, step_size=1)
        self.assertEqual((width * height, channels, patch_size, patch_size), x_unfold.shape)
