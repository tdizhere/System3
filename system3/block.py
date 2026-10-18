from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import StateSteeredGQA
from .config import System3Config
from .mlp import StateRoutedSwiGLU
from .utils import RMSNorm, parallel_associative_scan


class CoupledSystem3Block(nn.Module):
    def __init__(self, config: System3Config) -> None:
        super().__init__()
        self.config = config
        D, S = config.d_model, config.d_state
        self.norm_attn = RMSNorm(D)
        self.norm_mlp = RMSNorm(D)
        self.attn = StateSteeredGQA(config)
        self.mlp = StateRoutedSwiGLU(config)
        self.ada_mod = nn.Linear(S, 4 * D)

        if config.use_state_scan:
            self.state_norm = RMSNorm(S)
            self.h_to_state_impulse = nn.Linear(D, S, bias=False)
            self.state_forget_gate = nn.Linear(D + S, S)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, h: Tensor, s: Tensor, switch_trace: Tensor, valid_mask: Tensor, causal_fastpath: bool
    ) -> Tuple[Tensor, Tensor]:
        valid_3d = valid_mask.unsqueeze(-1)
        valid_f = valid_3d.to(h.dtype)

        if self.config.use_state_scan:
            retention = torch.sigmoid(self.state_forget_gate(torch.cat([h, s], dim=-1)))
            effective_gate = retention * (1.0 - switch_trace.unsqueeze(-1)) * valid_f
            # FIXED: Explicitly mask impulse with valid_f so padded tokens NEVER inject state into the scan
            impulse = (1.0 - effective_gate) * F.silu(self.h_to_state_impulse(h)) * valid_f
            scan_out = parallel_associative_scan(effective_gate, impulse)
            s_refined = self.state_norm(s + 0.2 * scan_out).masked_fill(~valid_3d, 0.0)
        else:
            s_refined = s.masked_fill(~valid_3d, 0.0)

        g1, b1, g2, b2 = self.ada_mod(s_refined).chunk(4, dim=-1)
        h = h + self.dropout(self.attn((1.0 + g1) * self.norm_attn(h) + b1, s_refined, valid_mask, causal_fastpath))
        h = h + self.dropout(self.mlp((1.0 + g2) * self.norm_mlp(h) + b2, s_refined))
        return h.masked_fill(~valid_3d, 0.0), s_refined
