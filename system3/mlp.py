from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import System3Config


class StateRoutedSwiGLU(nn.Module):
    def __init__(self, config: System3Config) -> None:
        super().__init__()
        self.use_lora = config.use_lora_swiglu
        D, R = config.d_model, config.lora_rank
        hidden = ((int(8 * D / 3) + 63) // 64) * 64
        self.w_gate = nn.Linear(D, hidden, bias=False)
        self.w_up = nn.Linear(D, hidden, bias=False)
        self.w_down = nn.Linear(hidden, D, bias=False)

        if self.use_lora:
            self.lora_down = nn.Linear(D, R, bias=False)
            self.state_to_lora_gate = nn.Linear(config.d_state, R, bias=False)
            self.lora_up = nn.Linear(R, D, bias=False)

    def forward(self, x: Tensor, state: Tensor) -> Tensor:
        base = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        if not self.use_lora:
            return base
        delta = self.lora_up(self.lora_down(x) * F.silu(self.state_to_lora_gate(state)))
        return base + delta
