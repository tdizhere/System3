from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _sdpa_supports_gqa() -> bool:
    """F.scaled_dot_product_attention is a C builtin, so inspect.signature() raises
    ValueError; probe it with a tiny call rather than sniffing the torch version."""
    try:
        _one = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(torch.zeros(1, 2, 1, 4), _one, _one, enable_gqa=True)
        return True
    except TypeError:
        return False


_SDPA_SUPPORTS_GQA = _sdpa_supports_gqa()


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _unit_rms(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Projects vectors onto the unit-RMS sphere (dimension- and scale-invariant)."""
    return x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return _unit_rms(x, self.eps).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(2)
        cos = self.cos_cached[:, :, :T, :].to(x.dtype)
        sin = self.sin_cached[:, :, :T, :].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def parallel_associative_scan(gates: Tensor, impulses: Tensor) -> Tensor:
    """Exact O(log2 T) Kogge-Stone associative prefix scan for s_t = a_t * s_{t-1} + b_t."""
    B, T, D = gates.shape
    a, b = gates.clone(), impulses.clone()
    for d in range(int(math.ceil(math.log2(max(T, 2))))):
        offset = 1 << d
        if offset >= T:
            break
        a_prev, b_prev = a[:, :-offset].clone(), b[:, :-offset].clone()
        a_curr, b_curr = a[:, offset:].clone(), b[:, offset:].clone()
        a[:, offset:] = a_curr * a_prev
        b[:, offset:] = a_curr * b_prev + b_curr
    return b


@torch.no_grad()
def log_sinkhorn_knopp(logits: Tensor, eps: float = 0.05, n_iters: int = 5) -> Tensor:
    N, M = logits.shape
    if N == 0:
        return torch.zeros_like(logits)
    log_Q = logits.float() / eps
    log_Q = log_Q - torch.logsumexp(log_Q, dim=-1, keepdim=True)
    log_row, log_col = -math.log(max(N, 1)), -math.log(max(M, 1))
    for _ in range(n_iters):
        log_Q = log_Q - (torch.logsumexp(log_Q, dim=0, keepdim=True) - log_col)
        log_Q = log_Q - (torch.logsumexp(log_Q, dim=1, keepdim=True) - log_row)
    return (log_Q + math.log(max(N, 1))).exp().to(logits.dtype)
