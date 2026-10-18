from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import StateSteeredGQA
from .block import CoupledSystem3Block
from .config import System3Config
from .mlp import StateRoutedSwiGLU
from .outputs import System3Output
from .router import ModularSystem12Router
from .utils import RMSNorm, _masked_mean


class System3Model(nn.Module):
    def __init__(self, config: Optional[System3Config] = None) -> None:
        super().__init__()
        self.config = config or System3Config()
        cfg = self.config
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.stem_norm1 = RMSNorm(cfg.d_model)
        self.stem_attn = StateSteeredGQA(cfg)
        self.stem_norm2 = RMSNorm(cfg.d_model)
        self.stem_mlp = StateRoutedSwiGLU(cfg)
        self.neutral_state = nn.Parameter(torch.zeros(1, 1, cfg.d_state))

        self.router = ModularSystem12Router(cfg)
        self.layers = nn.ModuleList([CoupledSystem3Block(cfg) for _ in range(cfg.n_layers - 1)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        for layer in self.layers:
            nn.init.zeros_(layer.ada_mod.weight)
            nn.init.zeros_(layer.ada_mod.bias)
            if cfg.use_lora_swiglu:
                nn.init.zeros_(layer.mlp.lora_up.weight)
        if cfg.use_lora_swiglu:
            nn.init.zeros_(self.stem_mlp.lora_up.weight)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        *,
        temperature: float = 0.7,
    ) -> System3Output:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        B, T = input_ids.shape
        if T == 0 or T > self.config.max_seq_len:
            raise ValueError(f"sequence length must be in [1, {self.config.max_seq_len}]")

        if attention_mask is None:
            valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
            causal_fastpath = True
        else:
            valid_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            causal_fastpath = False
        if not valid_mask[:, 0].all():
            raise ValueError("Each sequence must start with a valid (non-padding) token")

        valid_3d = valid_mask.unsqueeze(-1)
        x = self.token_embedding(input_ids).masked_fill(~valid_3d, 0.0)
        s0 = self.neutral_state.expand(B, T, -1)
        context = x + self.stem_attn(self.stem_norm1(x), s0, valid_mask, causal_fastpath)
        context = (context + self.stem_mlp(self.stem_norm2(context), s0)).masked_fill(~valid_3d, 0.0)

        routed = self.router(
            context,
            valid_mask,
            embed_seq=x,
            temperature=temperature,
            deterministic=not self.training,
        )

        hidden, state = context, routed.states
        for layer in self.layers:
            hidden, state = layer(hidden, state, routed.switch_trace, valid_mask, causal_fastpath)

        logits = self.lm_head(self.final_norm(hidden))

        components: Dict[str, Tensor] = {}
        loss: Optional[Tensor] = None
        if labels is not None:
            token_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="none"
            ).view(B, T)
            next_token_loss = _masked_mean(token_loss, valid_mask & labels.ne(-100))
            loss = (
                next_token_loss
                + self.config.balance_weight * routed.balance_loss
                + self.config.diversity_weight * routed.diversity_loss
                + self.config.jepa_weight * routed.jepa_loss
                + self.config.dwell_weight * routed.dwell_loss
            )
            components = {
                "total": loss,
                "next_token": next_token_loss,
                "balance": routed.balance_loss,
                "diversity": routed.diversity_loss,
                "jepa": routed.jepa_loss,
                "dwell": routed.dwell_loss,
            }

        return System3Output(logits=logits, loss=loss, loss_components=components, router=routed)
