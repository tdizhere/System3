from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from torch import Tensor


@dataclass
class RouterOutput:
    states: Tensor
    active_macro_ids: Tensor
    active_depth_ids: Tensor
    active_proposal_ids: Tensor
    switch_trace: Tensor
    open_probabilities: Tensor
    surprise_trace: Tensor
    dwell_trace: Tensor
    balance_loss: Tensor
    diversity_loss: Tensor
    jepa_loss: Tensor
    dwell_loss: Tensor
    actual_switch_rate: Tensor


@dataclass
class System3Output:
    logits: Tensor
    loss: Optional[Tensor]
    loss_components: Dict[str, Tensor]
    router: RouterOutput
