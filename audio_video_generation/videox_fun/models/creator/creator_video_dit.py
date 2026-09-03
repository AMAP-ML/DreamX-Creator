"""Shared DiT blocks used by the Creator audio transformer."""

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import RMSNorm

from ..attention_utils import attention


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift).to(shift.dtype)


def rope_apply_head_dim(x, freqs, head_dim):
    x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:3], -1, 2))
    return torch.view_as_real(x_complex * freqs).flatten(2).to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, freqs, seq_lens=None):
        q = rope_apply_head_dim(self.norm_q(self.q(x)), freqs, self.head_dim)
        k = rope_apply_head_dim(self.norm_k(self.k(x)), freqs, self.head_dim)
        v = self.v(x)
        b, s = q.shape[:2]
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_heads, self.head_dim)
        v = v.view(b, s, self.num_heads, self.head_dim)
        return self.o(attention(q, k, v, k_lens=seq_lens).flatten(2))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        b, s = q.shape[:2]
        n = self.num_heads
        d = q.shape[-1] // n
        output = attention(
            q.view(b, s, n, d),
            k.view(b, k.shape[1], n, d),
            v.view(b, v.shape[1], n, d),
        )
        return self.o(output.flatten(2))


class GateModule(nn.Module):
    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, seq_lens=None):
        chunk_dim = 2 if t_mod.ndim == 4 else 1
        modulation = (self.modulation.to(t_mod) + t_mod).chunk(6, dim=chunk_dim)
        if chunk_dim == 2:
            modulation = tuple(part.squeeze(2) for part in modulation)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        x = self.gate(x, gate_msa, self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs, seq_lens))
        x = x + self.cross_attn(self.norm3(x), context)
        return self.gate(x, gate_mlp, self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp)))
