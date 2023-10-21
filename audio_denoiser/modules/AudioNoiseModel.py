import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin

from audio_denoiser.modules.Permute import Permute
from audio_denoiser.modules.SimpleRoberta import SimpleRoberta
from audio_denoiser.modules.SpectrogramScaler import SpectrogramScaler


class AudioNoiseModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dict):
        super(AudioNoiseModel, self).__init__()

        # Encoder layers
        self.config = config
        scaler_dict = config['scaler']
        self.scaler = SpectrogramScaler.from_dict(scaler_dict)
        self.in_channels = config.get('in_channels', 257)
        self.roberta_hidden_size = config.get('roberta_hidden_size', 768)
        self.model1 = nn.Sequential(
            nn.Conv1d(self.in_channels, 1024, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(1024, 1024, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(1024, self.in_channels, kernel_size=1),
        )
        self.model2 = nn.Sequential(
            Permute(0, 2, 1),
            nn.Linear(self.in_channels, self.roberta_hidden_size),
            SimpleRoberta(num_hidden_layers=5, hidden_size=self.roberta_hidden_size),
            nn.Linear(self.roberta_hidden_size, self.in_channels),
            Permute(0, 2, 1),
        )

    @property
    def sample_rate(self) -> int:
        return self.config.get('sample_rate', 16000)

    @property
    def n_fft(self) -> int:
        return self.config.get('n_fft', 512)

    @property
    def num_frames(self) -> int:
        return self.config.get('num_frames', 32)

    def forward(self, x, use_scaler: bool = False, out_scale: float = 1.0):
        if use_scaler:
            x = self.scaler(x)
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = x1 + x2
        return x * out_scale
