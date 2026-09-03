"""Small raw-process-group helpers for Ulysses-style sequence parallelism."""

from __future__ import annotations

import torch
import torch.distributed as dist


def all_gather_sequence(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Gather equally-sized sequence chunks along dimension 1."""
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor
    tensor = tensor.contiguous()
    chunks = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(chunks, tensor, group=group)
    return torch.cat(chunks, dim=1).contiguous()


def all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int, group=None) -> torch.Tensor:
    """Scatter one dimension and gather another, matching Wan2.2's Ulysses layout."""
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor

    world_size = dist.get_world_size(group)
    if tensor.shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"Dimension {scatter_dim} ({tensor.shape[scatter_dim]}) must be divisible by "
            f"SP size {world_size}"
        )
    inputs = [part.contiguous() for part in tensor.chunk(world_size, dim=scatter_dim)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(world_size)]
    dist.all_to_all(outputs, inputs, group=group)
    return torch.cat(outputs, dim=gather_dim).contiguous()


def ulysses_attention(q, k, v, attention_fn, *, k_lens=None, window_size=(-1, -1), group=None):
    """Run attention on local sequence chunks with Ulysses head/sequence exchange."""
    q = all_to_all(q, scatter_dim=2, gather_dim=1, group=group)
    k = all_to_all(k, scatter_dim=2, gather_dim=1, group=group)
    v = all_to_all(v, scatter_dim=2, gather_dim=1, group=group)
    output = attention_fn(q, k, v, k_lens=k_lens, window_size=window_size)
    return all_to_all(output, scatter_dim=1, gather_dim=2, group=group)
