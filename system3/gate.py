from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import System3Config


class CalibratedSwitchGate(nn.Module):
    """
    Supports both simple 2-way Gumbel-Softmax routing ('softmax') and stretched
    Hard-Concrete gates ('hard_concrete'), with calibrated unit-RMS surprise normalization.
    """
    def __init__(self, config: System3Config, gamma: float = -0.1, zeta: float = 1.1) -> None:
        super().__init__()
        self.config = config
        self.gamma = gamma
        self.zeta = zeta
        self.register_buffer("dwell_freqs", torch.exp(torch.linspace(0, -math.log(100.0), 8)), persistent=False)

        # Running EMA buffers to keep surprise zero-centered and unit-variance regardless of training stage
        self.register_buffer("surprise_ema_mean", torch.tensor(1.0), persistent=True)
        self.register_buffer("surprise_ema_var", torch.tensor(0.5), persistent=True)

        extra_dim = (1 + 16) if config.use_surprise_gate else 0
        out_dim = 1 if config.gate_type == "hard_concrete" else 2
        self.network = nn.Sequential(
            nn.Linear(config.d_model + config.d_state + extra_dim, config.d_state),
            nn.SiLU(),
            nn.Linear(config.d_state, out_dim),
        )
        if config.gate_type == "hard_concrete":
            # Was -1.8, which put every position at sigmoid(-1.8) ~ 0.14 open probability
            # but a *closed* mode (logits < 0 everywhere), so the deterministic eval gate
            # never opened and the router never received gradient. Start it undecided so the
            # rate target and the surprise signal can move it either way.
            nn.init.constant_(self.network[-1].bias, 0.0)
        else:
            nn.init.constant_(self.network[-1].bias[0], 0.5)
            nn.init.constant_(self.network[-1].bias[1], -0.5)

    def normalize_surprise(self, raw_surprise: Tensor, valid_step: Tensor) -> Tensor:
        """Standardizes bounded [0, 4] unit-RMS residual error using running EMA statistics."""
        if self.training and valid_step.any():
            valid_vals = raw_surprise.detach().float()[valid_step]
            batch_mean = valid_vals.mean()
            batch_var = valid_vals.var(unbiased=False).clamp_min(1e-4)
            self.surprise_ema_mean.lerp_(batch_mean, 0.05)
            self.surprise_ema_var.lerp_(batch_var, 0.05)
        std = (self.surprise_ema_var + 1e-5).sqrt()
        return ((raw_surprise.float() - self.surprise_ema_mean) / std).clamp(-3.0, 3.0).to(raw_surprise.dtype)

    def forward(
        self,
        h_t: Tensor,
        s_prev: Tensor,
        surprise: Tensor,
        tau_prev: Tensor,
        *,
        temperature: float,
        deterministic: bool,
    ) -> Tuple[Tensor, Tensor]:
        if self.config.use_surprise_gate:
            angles = tau_prev.float() * self.dwell_freqs.unsqueeze(0)
            dwell_emb = torch.cat([angles.sin(), angles.cos()], dim=-1).to(h_t.dtype)
            gate_in = torch.cat([h_t, s_prev, surprise, dwell_emb], dim=-1)
        else:
            gate_in = torch.cat([h_t, s_prev], dim=-1)

        logits = self.network(gate_in).float()

        if self.config.gate_type == "hard_concrete":
            # P(gate is open) = P(s_bar > 0.5). Training draws s = sigmoid((noise + logits)/temperature)
            # and s is monotone in `noise`, so P(s > 0.5) = P(noise > -logits) = sigmoid(logits) -- the
            # temperature cancels out. The previous expression added a fixed offset, so the rate target
            # was reported as met while the gate actually stayed shut ~97% of the time.
            open_prob = torch.sigmoid(logits)
            if not deterministic:
                u = torch.empty_like(logits).uniform_(1e-6, 1.0 - 1e-6)
                s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + logits) / temperature)
                s_bar = (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)
                g_hard = (s_bar > 0.5).float()
                g_t = g_hard - s_bar.detach() + s_bar
            else:
                # Evaluate the gate's mode: open whenever P(open) = sigmoid(logits) exceeds 0.5,
                # i.e. the same quantity the router reports as open_prob.
                g_t = (torch.sigmoid(logits) > 0.5).float()
            return g_t.to(h_t.dtype), open_prob.to(h_t.dtype)
        else:
            open_prob = (logits / max(temperature, 1e-6)).softmax(dim=-1)[..., 1:2]
            if not deterministic:
                weights = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
            else:
                idx = logits.argmax(dim=-1, keepdim=True)
                weights = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
            return weights[..., 1:2].to(h_t.dtype), open_prob.to(h_t.dtype)
