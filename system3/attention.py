from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import System3Config
from .utils import RotaryEmbedding, _SDPA_SUPPORTS_GQA


class StateSteeredGQA(nn.Module):
    """Uses native SDPA enable_gqa=True when available to avoid materializing expanded KV tensors."""
    def __init__(self, config: System3Config) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.d_model // self.n_heads
        self.dropout = config.dropout

        self.wq = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

        if config.use_head_temp_steering:
            self.state_temp_proj = nn.Linear(config.d_state, self.n_heads)
            self.state_v_gate = nn.Linear(config.d_state, config.d_model)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta)

    def forward(self, x: Tensor, state: Tensor, valid_mask: Tensor, causal_fastpath: bool) -> Tensor:
        B, T, D = x.shape
        q = self.rope(self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2))
        k = self.rope(self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2))
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.config.use_head_temp_steering:
            head_scale = F.softplus(self.state_temp_proj(state)).transpose(1, 2).unsqueeze(-1)
            q = q * head_scale

        drop_p = self.dropout if self.training else 0.0

        # Memory-efficient native GQA fastpath (no KV tensor expansion in VRAM)
        if _SDPA_SUPPORTS_GQA and causal_fastpath:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=True, enable_gqa=(self.n_rep > 1))
        else:
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            if causal_fastpath:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=True)
            else:
                causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
                allowed = causal.view(1, 1, T, T) & valid_mask[:, None, None, :]
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=drop_p)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        if self.config.use_head_temp_steering:
            out = out * torch.sigmoid(self.state_v_gate(state))
        return self.wo(out)
