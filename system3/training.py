from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .model import System3Model


@dataclass
class TrainConfig:
    data_path: str = ""              # uint16 GPT-2 token shard written by the prep step
    vocab_size: int = 50257          # must match the tokenizer used for the shard
    seq_len: int = 256               # context length of each training example
    batch_size: int = 192            # ~60 GB peak; lower it if you OOM
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 200          # linear warmup, then cosine decay to zero
    max_steps: int = 4000            # shard / (batch*seq) is ~3.0k steps per epoch
    steps_per_run: int = 200         # optimizer steps executed per run
    grad_clip: float = 1.0
    log_every: int = 1               # print one line per optimizer step
    eval_every: int = 50
    eval_batches: int = 10           # validation batches per evaluation
    val_fraction: float = 0.01       # held-out tail of the token stream
    patience: int = 5                # stop after N evals with no val improvement (0 = off)
    restore_best: bool = True        # reload best-val weights when stopping
    save_every: int = 50             # checkpoint cadence (steps)
    keep_last: int = 2               # numbered snapshots to retain
    ckpt_dir: str = "checkpoints"
    auto_resume: bool = True         # resume from <ckpt_dir>/latest.pt when present
    seed: int = 0
    use_amp: bool = True             # bfloat16 autocast on CUDA
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def new_train_state() -> dict:
    """State carried across runs and across sessions via checkpoints."""
    return {
        "step": 0,
        "best_val": float("inf"),
        "evals_no_improve": 0,
        "running": False,
        "history": {
            "step": [],
            "train_ntp": [],
            "train_total": [],
            "lr": [],
            "val_step": [],
            "val_ntp": [],
            "switch_rate": [],
        },
    }


def make_optimizer(model: System3Model, cfg: TrainConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )


def _lr_scale_factory(cfg: TrainConfig):
    def _lr_scale(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return _lr_scale


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_scale_factory(cfg))


class CheckpointManager:
    """Writes resumable checkpoints (weights + optimizer + scheduler + step + curves)."""

    def __init__(self, model: System3Model, optimizer, scheduler, cfg: TrainConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.dir = Path(cfg.ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.dir / "latest.pt"
        self.best_path = self.dir / "best.pt"

    def save(self, path, state: dict, numbered: bool = False) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": state["step"],
                "best_val": state["best_val"],
                "evals_no_improve": state["evals_no_improve"],
                "history": state["history"],
                # plain dicts, not the dataclass, so the file loads in a later session
                "config": asdict(self.cfg),
                "model_config": asdict(self.model.config),
            },
            path,
        )
        if numbered:
            snap = self.dir / f"step_{state['step']:06d}.pt"
            torch.save(
                {"model": self.model.state_dict(), "step": state["step"], "config": asdict(self.cfg)},
                snap,
            )
            for old in sorted(self.dir.glob("step_*.pt"))[: -self.cfg.keep_last]:
                old.unlink()

    def load(self, path, state: dict) -> bool:
        """Restore weights, optimizer, scheduler, step and curves. Returns True on success."""
        if not Path(path).exists():
            return False
        blob = torch.load(path, map_location=self.cfg.device, weights_only=True)
        self.model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            self.optimizer.load_state_dict(blob["optimizer"])
        if "scheduler" in blob:
            self.scheduler.load_state_dict(blob["scheduler"])
        state["step"] = int(blob.get("step", 0))
        state["best_val"] = float(blob.get("best_val", float("inf")))
        state["evals_no_improve"] = int(blob.get("evals_no_improve", 0))
        if blob.get("history"):
            # merge, so keys added after this checkpoint was written keep their empty template
            for k, v in blob["history"].items():
                state["history"][k] = list(v)
        return True

    def restore_best(self) -> float | None:
        """Reload the best-validation weights without rewinding step/history."""
        if not self.best_path.exists():
            return None
        blob = torch.load(self.best_path, map_location=self.cfg.device, weights_only=True)
        self.model.load_state_dict(blob["model"])
        return float(blob.get("best_val", float("inf")))


@torch.no_grad()
def evaluate(model: System3Model, loader, cfg: TrainConfig, max_batches: int) -> float:
    """Mean next-token cross-entropy over a few validation batches."""
    model.eval()
    device_type = cfg.device.split(":")[0]
    amp = cfg.use_amp and device_type == "cuda"
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            out = model(x, labels=y)
        total += float(out.loss_components["next_token"])
        n += 1
    model.train()
    return total / max(n, 1)


def train_loop(model, optimizer, scheduler, train_loader, val_loader, ckpt, state, cfg, eval_fn=evaluate) -> None:
    """Guard against two runs mutating the same state at once."""
    if state.get("running"):
        print("a training run is already in progress; ignoring")
        return
    state["running"] = True
    try:
        train_steps(model, optimizer, scheduler, train_loader, val_loader, ckpt, state, cfg, eval_fn)
    finally:
        state["running"] = False


def train_steps(model, optimizer, scheduler, train_loader, val_loader, ckpt, state, cfg, eval_fn=evaluate) -> None:
    """Run up to `cfg.steps_per_run` steps, printing one line per step.

    Continues from `state`, which the caller seeds from the checkpoint on disk,
    so running this again extends the run instead of starting over. Stops early
    when validation has not improved for `cfg.patience` evaluations, then reloads
    the best-validation weights.
    """
    device_type = cfg.device.split(":")[0]
    amp = cfg.use_amp and device_type == "cuda"
    model.to(cfg.device).train()

    if state["step"] >= cfg.max_steps:
        print(f"already at cfg.max_steps ({cfg.max_steps}); raise it in TrainConfig to keep training")
        return

    hist = state["history"]
    keys = ("ntp", "total", "balance", "diversity", "jepa", "dwell", "switch")
    acc = dict.fromkeys(keys, 0.0)
    n = 0
    tokens_seen, t0 = 0, time.time()
    target = min(cfg.max_steps, state["step"] + cfg.steps_per_run)
    stop = False

    print(f"{'step':>6} | {'ntp':>8} {'total':>8} {'lr':>9} | {'bal':>6} {'div':>7} {'jepa':>7} {'dwell':>7} "
          f"| {'sw':>5} | {'tok/s':>7} | elapsed")
    print("-" * 108)

    while state["step"] < target and not stop:
        for x, y in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
                out = model(x, labels=y)

            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            state["step"] += 1
            step = state["step"]
            comp = out.loss_components
            for k_out, k_acc in (("next_token", "ntp"), ("total", "total"), ("balance", "balance"),
                                 ("diversity", "diversity"), ("jepa", "jepa"), ("dwell", "dwell")):
                acc[k_acc] += float(comp[k_out])
            acc["switch"] += float(out.router.actual_switch_rate)
            n += 1
            tokens_seen += x.numel()
            lr_now = scheduler.get_last_lr()[0]

            if step % cfg.log_every == 0 or step == target:
                hist["step"].append(step)
                hist["train_ntp"].append(acc["ntp"] / n)
                hist["train_total"].append(acc["total"] / n)
                hist["lr"].append(lr_now)
                hist["switch_rate"].append(acc["switch"] / n)
                elapsed = time.time() - t0
                print(f"{step:6d} | {acc['ntp'] / n:8.4f} {acc['total'] / n:8.4f} {lr_now:9.2e} "
                      f"| {acc['balance'] / n:6.3f} {acc['diversity'] / n:7.4f} {acc['jepa'] / n:7.4f} "
                      f"{acc['dwell'] / n:7.4f} | {acc['switch'] / n:5.3f} "
                      f"| {tokens_seen / elapsed:7,.0f} | {elapsed:5.0f}s")
                acc = dict.fromkeys(keys, 0.0)
                n = 0

            if step % cfg.eval_every == 0 or step == target:
                val = eval_fn(model, val_loader, cfg, cfg.eval_batches)
                hist["val_step"].append(step)
                hist["val_ntp"].append(val)
                if val < state["best_val"]:
                    state["best_val"] = val
                    state["evals_no_improve"] = 0
                    ckpt.save(ckpt.best_path, state)
                    note = "   *** new best ***"
                else:
                    state["evals_no_improve"] += 1
                    note = f"   (no improve {state['evals_no_improve']}/{cfg.patience})"
                print(f"{'':>6} |   VAL {val:.4f}   best {state['best_val']:.4f}{note}")
                if cfg.patience and state["evals_no_improve"] >= cfg.patience:
                    print(f"{'':>6} |   early stop: validation flat for {cfg.patience} evals")
                    stop = True

            if step % cfg.save_every == 0 or step == target:
                ckpt.save(ckpt.latest_path, state, numbered=True)
                print(f"{'':>6} |   checkpoint saved -> {ckpt.latest_path}")

            if step >= target or stop:
                break

    print(f"done: {tokens_seen:,} tokens in {time.time() - t0:.0f}s | global step {state['step']}/{cfg.max_steps}")
    if cfg.restore_best:
        best = ckpt.restore_best()
        if best is not None:
            print(f"restored best-validation weights (val {best:.4f})")
