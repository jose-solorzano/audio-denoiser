import math
from typing import Iterable
import torch
from torch import nn
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_from_list(param_list: Iterable[nn.Parameter]):
    return sum(p.numel() for p in param_list)


def sin_pos_embeddings(seq_len: int, emb_dim: int) -> torch.Tensor:
    factors = torch.arange(1, emb_dim + 1)
    seq_range = torch.arange(0, seq_len)
    angles = math.pi * 2 * seq_range / seq_len
    return torch.sin(angles[:, None] * factors[None, :])


def concatenate_tensors(list_of_tensors):
    if isinstance(list_of_tensors[0], tuple):
        concatenated_tuples = []
        for i in range(len(list_of_tensors[0])):
            concatenated_tensors = torch.cat([t[i] for t in list_of_tensors], dim=0)
            concatenated_tuples.append(concatenated_tensors)
        return tuple(concatenated_tuples)
    else:
        return torch.cat(list_of_tensors, dim=0)


def batched_apply(model: nn.Module, x: torch.Tensor, batch_size: int = 64, detached: bool = False):
    output_list = []
    num_items = x.size(0)
    for b0 in range(0, num_items, batch_size):
        input_batch = x[b0:b0 + batch_size]
        output_batch = model(input_batch)
        if detached:
            if isinstance(output_batch, tuple):
                output_batch = tuple([x.detach() for x in output_batch])
            else:
                output_batch = output_batch.detach()
        output_list.append(output_batch)
    output = concatenate_tensors(output_list)
    return output


def sample_tensor(tensor: torch.Tensor, count: int):
    perm = torch.randperm(tensor.size(0), device=tensor.device)
    return tensor[perm[:count]]


def sample_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, count: int):
    if tensor2.size(0) != tensor1.size(0):
        raise ValueError('Different batch sizes!')
    perm = torch.randperm(tensor1.size(0), device=tensor1.device)
    selection = perm[:count]
    return tensor1[selection], tensor2[selection],


def fixed_hash(text: str):
    h = 0
    for ch in text:
        h = (h * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
    return h


def tensor_hash(t: torch.Tensor):
    return fixed_hash(str(t.cpu().tolist()))


def unfold_2d(tensor: torch.Tensor, patch_size: int, step_size: int):
    # tensor shape: (channels, height, width)
    # result shape: (num_patches, channels, patch_size, patch_size)
    _, orig_height, orig_width = tensor.shape
    if orig_width < patch_size or orig_height < patch_size:
        raise ValueError(f'Tensor too small for patch_size={patch_size}')
    num_rows = math.ceil((orig_height - patch_size) / step_size) + 1
    num_cols = math.ceil((orig_width - patch_size) / step_size) + 1
    num_patches = num_rows * num_cols
    new_height = (num_rows - 1) * step_size + patch_size
    new_width = (num_cols - 1) * step_size + patch_size
    if new_height != orig_height or new_width != orig_width:
        padding_height = new_height - orig_height
        padding_width = new_width - orig_width
        tensor = F.pad(tensor, pad=(0, padding_width, 0, padding_height))
    # tensor shape: (channels, height, width)
    patches = tensor.unfold(1, patch_size, step_size)
    patches = patches.unfold(2, patch_size, step_size)
    patches = patches.contiguous().view((3, num_patches, patch_size, patch_size))
    patches_input = torch.permute(patches, (1, 0, 2, 3))
    return patches_input


def fold_2d(tensor: torch.Tensor, width: int, height: int):
    # tensor shape: (num_patches, channels, patch_size, patch_size)
    # result shape: (channels, height, width)
    num_patches, channels, patch_size, _ = tensor.shape
    step_size = patch_size
    num_rows = math.ceil(height / step_size)
    num_cols = math.ceil(width / step_size)
    if num_patches != num_rows * num_cols:
        raise ValueError(f'Expected num_rows={num_rows} * num_cols={num_cols} to be {num_patches}')
    new_height = num_rows * patch_size
    new_width = num_cols * patch_size
    tensor = torch.permute(tensor, (1, 0, 2, 3))
    # tensor: (channels, num_patches, patch_size, patch_size)
    tensor = tensor.view(channels, num_rows, num_cols, patch_size, patch_size)
    tensor = torch.permute(tensor, (0, 1, 3, 2, 4))
    # tensor: (channels, num_rows, patch_size, num_cols, patch_size)
    tensor = tensor.contiguous().view(channels, new_height, new_width)
    return torch.clone(tensor[:, :height, :width])
