"""Run the System3 diagnostic audit against a saved checkpoint.

    python audit.py --data tokens.uint16
    python audit.py --ckpt checkpoints/best.pt --data tokens.uint16 --batches 20

Prints the driver-vs-passenger counterfactual table and shows the 3-panel figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from system3 import System3Config, System3Model
from system3.audit import run_system3_audit
from system3.data import build_dataloaders
from system3.training import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit a trained System3 checkpoint.")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (defaults to best.pt then latest.pt).")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--data", type=str, default=None, help="Override the token shard path.")
    p.add_argument("--batches", type=int, default=15)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--save", type=str, default=None, help="Save the figure to this path instead of showing it.")
    return p.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.ckpt:
        path = Path(args.ckpt)
    else:
        ckpt_dir = Path(args.ckpt_dir)
        path = ckpt_dir / "best.pt"
        if not path.exists():
            path = ckpt_dir / "latest.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint found at {path}")
    return path


def main() -> None:
    args = parse_args()
    ckpt_path = resolve_checkpoint(args)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_cfg = System3Config(**blob["model_config"])
    model = System3Model(model_cfg)
    model.load_state_dict(blob["model"])

    train_cfg = TrainConfig(**{k: v for k, v in blob.get("config", {}).items()
                               if k in TrainConfig.__dataclass_fields__ and k != "device"})
    if args.data is not None:
        train_cfg.data_path = args.data
    if args.device is not None:
        train_cfg.device = args.device
    if args.no_amp:
        train_cfg.use_amp = False
    if not train_cfg.data_path or not Path(train_cfg.data_path).exists():
        raise SystemExit("No token shard available; pass --data path/to/tokens.uint16 to build the validation loader.")

    _, val_loader = build_dataloaders(train_cfg)

    fig, _ = run_system3_audit(
        model, val_loader,
        device=train_cfg.device,
        use_amp=train_cfg.use_amp,
        num_batches=args.batches,
        ckpt_label=ckpt_path.name,
        step=blob.get("step"),
    )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved figure -> {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
