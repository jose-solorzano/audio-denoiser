import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel


class SimpleRoberta(nn.Module):
    def __init__(self, num_hidden_layers: int, hidden_size: int,
                 num_attention_heads: int = 8,
                 hidden_act: str = 'gelu'):
        super().__init__()
        config = RobertaConfig()
        config.vocab_size = 1
        config.pad_token_id = 0
        config.hidden_size = hidden_size
        config.num_hidden_layers = num_hidden_layers
        config.num_attention_heads = num_attention_heads
        config.hidden_act = hidden_act
        self.model = RobertaModel(config)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        rm_output = self.model(inputs_embeds=inputs_embeds)
        return rm_output.last_hidden_state
