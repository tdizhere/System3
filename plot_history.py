"""Plot the training curves from a saved System3 checkpoint.

    python plot_history.py --ckpt-dir checkpoints
    python plot_history.py --ckpt checkpoints/best.pt --save curves.png

Reads the finished run off disk, so results are viewable without re-running any
training.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot System3 training curves.")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (defaults to best.pt then latest.pt).")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--save", type=str, default=None, help="Save the figure to this path instead of showing it.")
    return p.parse_args()


def resolve_checkpoint(ckpt: str | None, ckpt_dir: str) -> Path:
    if ckpt:
        path = Path(ckpt)
    else:
        d = Path(ckpt_dir)
        path = d / "latest.pt"
        if not path.exists():
            path = d / "best.pt"
    if not path.exists():
        raise SystemExit(f"No checkpoint found at {path}")
    return path


def plot_history(blob: dict):
    import matplotlib.pyplot as plt

    saved = blob.get("config", {})
    hist = blob.get("history", {})
    xs, tn = hist.get("step", []), hist.get("train_ntp", [])
    vs, vn = hist.get("val_step", []), hist.get("val_ntp", [])
    sw = hist.get("switch_rate", [])
    best = float(blob.get("best_val", float("inf")))
    vocab_size = saved.get("vocab_size", 50257)
    guess = math.log(vocab_size)
    n_params = sum(v.numel() for v in blob.get("model", {}).values())

    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.32)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    # A linear axis crushes the converged 2.0-3.3 region into the bottom of the plot,
    # so use a log scale to give the converged regime the space it needs.
    ax.set_yscale("log")
    ax.plot(vs, vn, "o-", color="#c1121f", lw=2.6, ms=5, zorder=5,
            label="VALIDATION - held-out text, deterministic routing")
    ax.plot(xs, tn, lw=0.7, color="#1d3557", alpha=0.22,
            label="training - every step, stochastic routing")
    w = max(10, len(tn) // 40)
    if len(tn) > w:
        ax.plot(xs[w - 1:], np.convolve(tn, np.ones(w) / w, mode="valid"), lw=2.4, color="#1d3557",
                label=f"training - mean of {w} steps")
    ax.axhline(guess, ls="--", lw=1.2, color="0.6",
               label=f"random guessing = {guess:.2f} nats (perplexity {vocab_size:,})")

    if vn:
        i = int(np.argmin(vn))
        ax.plot([vs[i]], [vn[i]], marker="*", ms=22, color="#c1121f", zorder=6)
        ax.annotate(f"best validation {vn[i]:.3f} nats\n(perplexity {math.exp(vn[i]):.1f}) at step {vs[i]}",
                    (vs[i], vn[i]), textcoords="offset points", xytext=(16, 20), fontsize=9.5,
                    fontweight="bold", color="#8b0000",
                    arrowprops=dict(arrowstyle="-", color="#8b0000", lw=1.2))

    ax.set_ylabel("next-token cross-entropy\n(nats, log scale - lower is better)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.93)
    ax.grid(alpha=0.25, which="both")
    ax.text(0.013, 0.035, "Read the RED line, ignore the blue one:\n"
                          "validation is what says whether the model generalises.\n"
                          "It is still falling => under-trained, not broken.",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff7f7", ec="#c1121f", lw=1))

    sw_txt = f"{float(np.mean(sw[-20:])):.3f}" if sw else "n/a"
    fig.suptitle("Language-model loss on TinyStories + TinyGSM8k", fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             f"best validation {best:.3f} nats (perplexity {math.exp(best):.1f})   |   "
             f"step {int(blob.get('step', 0))}/{saved.get('max_steps', '?')}   |   "
             f"{saved.get('batch_size', '?')} x {saved.get('seq_len', '?')} tokens/step   |   "
             f"{n_params / 1e6:.1f}M params   |   router switch {sw_txt}",
             ha="center", fontsize=9.5, family="monospace")

    ax2.plot(xs, hist.get("lr", []), lw=2, color="#6a4c93")
    ax2.set_xlabel("training step", fontsize=10)
    ax2.set_ylabel("learning rate", fontsize=10)
    ax2.set_title("Learning-rate schedule: linear warmup, then cosine decay to zero", fontsize=9.5, loc="left")
    ax2.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main() -> None:
    args = parse_args()
    ckpt_path = resolve_checkpoint(args.ckpt, args.ckpt_dir)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    fig = plot_history(blob)
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved figure -> {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
