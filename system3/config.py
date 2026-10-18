from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


StateMode = Literal["discrete", "continuous", "hybrid"]
GateType = Literal["softmax", "hard_concrete"]
BalanceMode = Literal["none", "switch", "sinkhorn"]
DiversityMode = Literal["none", "orthogonality", "dpp"]
DwellMode = Literal["none", "rate_target", "l0_penalty"]
SurpriseTarget = Literal["hidden", "embedding"]


@dataclass(frozen=True)
class System3Config:
    vocab_size: int = 32_000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    d_state: int = 128
    num_macro_states: int = 6
    num_depth_modes: int = 4             # Set to 1 for flat (non-compositional) base states
    num_proposals: int = 4
    lora_rank: int = 16
    max_seq_len: int = 2_048
    rope_theta: float = 10_000.0
    dropout: float = 0.0

    # --- Modular Ablation Switches (Toggle features one by one) ---
    state_mode: StateMode = "hybrid"
    closed_loop_proposals: bool = True   # False = Open-loop V1; True = conditioned on s_{t-1}
    use_surprise_gate: bool = True       # False = Gate only sees (h_t, s_{t-1}); True = + surprise & dwell
    gate_type: GateType = "hard_concrete"
    balance_mode: BalanceMode = "sinkhorn"
    diversity_mode: DiversityMode = "dpp"
    dwell_mode: DwellMode = "rate_target"
    use_state_scan: bool = True          # Inter-layer parallel associative state refinement
    use_lora_swiglu: bool = True         # Dynamic rank-r state-routed FFN perturbation
    use_head_temp_steering: bool = True  # State-modulated per-head attention temperature
    surprise_target: SurpriseTarget = "embedding"  # Predictive-coding target for the switch gate

    # --- Loss Hyperparameters ---
    target_switch_rate: float = 0.08
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 5
    balance_weight: float = 0.02
    diversity_weight: float = 0.01
    dwell_weight: float = 1.0   # was 0.10: at 0.10 the rate target was ~0.04% of the LM loss
                                # and the gate simply stayed shut (measured 3% vs 8% target)
    jepa_weight: float = 0.05
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "num_macro_states": self.num_macro_states,
            "num_depth_modes": self.num_depth_modes,
            "num_proposals": self.num_proposals,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "sinkhorn_eps": self.sinkhorn_eps,
            "sinkhorn_iters": self.sinkhorn_iters,
            "initializer_range": self.initializer_range,
        }
        bad = [k for k, v in positive.items() if v <= 0]
        if bad:
            raise ValueError(f"Parameters must be strictly positive: {', '.join(bad)}")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for Rotary Position Embeddings (RoPE)")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for Grouped-Query Attention")
        if self.n_layers < 2:
            raise ValueError("n_layers must be at least 2 (1 context stem + >=1 steered block)")

        total_discrete = self.num_macro_states * self.num_depth_modes
        if self.state_mode in ("discrete", "hybrid") and self.num_proposals > total_discrete:
            raise ValueError(
                f"num_proposals ({self.num_proposals}) cannot exceed total discrete states "
                f"num_macro_states * num_depth_modes ({total_discrete}) required by top-k routing"
            )
        if self.diversity_mode in ("orthogonality", "dpp") and self.num_proposals > self.d_state:
            raise ValueError(f"num_proposals ({self.num_proposals}) cannot exceed d_state ({self.d_state})")
        if self.use_lora_swiglu and self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive when use_lora_swiglu=True")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.target_switch_rate <= 1.0:
            raise ValueError("target_switch_rate must be in [0, 1]")
        for w_name in ("balance_weight", "diversity_weight", "dwell_weight", "jepa_weight"):
            if getattr(self, w_name) < 0.0:
                raise ValueError(f"{w_name} cannot be negative")

    @classmethod
    def minimal_baseline(cls, **overrides) -> System3Config:
        """
        Rung 0 Baseline: Simple open-loop routing, flat base states, standard softmax switch gate,
        Switch-MoE balance loss, pairwise orthogonality, and FiLM-only steering.
        """
        defaults = dict(
            num_depth_modes=1,
            closed_loop_proposals=False,
            use_surprise_gate=False,
            gate_type="softmax",
            balance_mode="switch",
            diversity_mode="orthogonality",
            dwell_mode="rate_target",
            use_state_scan=False,
            use_lora_swiglu=False,
            use_head_temp_steering=False,
            jepa_weight=0.0,
        )
        defaults.update(overrides)
        return cls(**defaults)

    def with_ablation(self, **changes) -> System3Config:
        """Helper to toggle individual mechanisms on top of a baseline config."""
        return replace(self, **changes)
