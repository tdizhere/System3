"""Train / continue a System3 language model on a uint16 token shard.

The shard is a flat array of token ids (dtype uint16) produced by your tokenizer
prep step. The tail `val_fraction` of the stream is held out for validation.

    python train.py --data tokens.uint16 --steps-per-run 200

Each invocation adds `--steps-per-run` optimizer steps to the existing run
(resuming from <ckpt_dir>/latest.pt when present), so it is always safe to re-run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from system3 import System3Config, System3Model
from system3.data import build_dataloaders
from system3.training import (
    CheckpointManager,
    TrainConfig,
    make_optimizer,
    make_scheduler,
    new_train_state,
    train_loop,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train System3.")
    p.add_argument("--data", type=str, default=None, help="Path to a uint16 token shard.")
    p.add_argument("--steps-per-run", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    # model geometry overrides
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--d-state", type=int, default=128)
    return p.parse_args()


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.data is not None:
        cfg.data_path = args.data
    if args.steps_per_run is not None:
        cfg.steps_per_run = args.steps_per_run
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.lr is not None:
        cfg.lr = args.lr
    if args.device is not None:
        cfg.device = args.device
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if args.no_amp:
        cfg.use_amp = False
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(TrainConfig(), args)

    if not cfg.data_path or not Path(cfg.data_path).exists():
        raise SystemExit(
            "No token shard found. Pass --data path/to/tokens.uint16 (a flat uint16 token array)."
        )

    # cuBLAS/cuDNN fastest paths: TF32 for the matmuls autocast leaves in fp32, plus
    # autotuning for the fixed (batch, seq) shapes used here.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(cfg.seed)

    model_cfg = System3Config(
        vocab_size=cfg.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_state=args.d_state,
        num_macro_states=6,
        num_depth_modes=4,
        num_proposals=4,
        lora_rank=16,
        max_seq_len=max(cfg.seq_len, 1024),
    )
    model = System3Model(model_cfg)

    # nn.Embedding starts at N(0, 1) and the lm_head is tied to it, so the initial
    # logits are ~O(10) and the starting loss is ~100 nats. Rescale the shared
    # embedding to the config's initializer_range so training starts at ~ln(vocab).
    torch.nn.init.normal_(model.token_embedding.weight, mean=0.0, std=model_cfg.initializer_range)
    model.to(cfg.device)

    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    train_loader, val_loader = build_dataloaders(cfg)
    ckpt = CheckpointManager(model, optimizer, scheduler, cfg)
    state = new_train_state()

    n_params = sum(p.numel() for p in model.parameters())
    resumed = cfg.auto_resume and ckpt.load(ckpt.latest_path, state)
    print(
        f"System3Model: {n_params / 1e6:.1f}M params on {cfg.device}"
        f" | {'resumed ' + str(ckpt.latest_path) if resumed else 'fresh init'}"
        f" | global step {state['step']}/{cfg.max_steps}"
    )
    print(f"batch {cfg.batch_size} x seq {cfg.seq_len} = {cfg.batch_size * cfg.seq_len:,} tokens/step")

    train_loop(model, optimizer, scheduler, train_loader, val_loader, ckpt, state, cfg)


if __name__ == "__main__":
    main()
