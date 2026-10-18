"""System3: a neural architecture coupling a JEPA-style predictive-coding
System 1 state router with an autoregressive System 2 transformer."""

from .attention import StateSteeredGQA
from .block import CoupledSystem3Block
from .config import (
    BalanceMode,
    DiversityMode,
    DwellMode,
    GateType,
    StateMode,
    SurpriseTarget,
    System3Config,
)
from .gate import CalibratedSwitchGate
from .mlp import StateRoutedSwiGLU
from .model import System3Model
from .outputs import RouterOutput, System3Output
from .router import ModularSystem12Router
from .utils import RMSNorm, RotaryEmbedding, log_sinkhorn_knopp, parallel_associative_scan

__all__ = [
    "System3Config",
    "System3Model",
    "System3Output",
    "RouterOutput",
    "ModularSystem12Router",
    "CalibratedSwitchGate",
    "StateSteeredGQA",
    "StateRoutedSwiGLU",
    "CoupledSystem3Block",
    "RMSNorm",
    "RotaryEmbedding",
    "parallel_associative_scan",
    "log_sinkhorn_knopp",
    "StateMode",
    "GateType",
    "BalanceMode",
    "DiversityMode",
    "DwellMode",
    "SurpriseTarget",
]
