import numpy as np
import pytest
import torch

from system3 import System3Config, System3Model
from system3.data import TokenChunkDataset, build_dataloaders
from system3.training import (
    CheckpointManager,
    TrainConfig,
    evaluate,
    make_optimizer,
    make_scheduler,
    new_train_state,
    train_loop,
)


def write_shard(path, n_tokens=256, vocab=32, seed=0):
    rng = np.random.default_rng(seed)
    rng.integers(0, vocab, size=n_tokens, dtype=np.uint16).tofile(str(path))


def tiny_model(vocab=32):
    cfg = System3Config(
        vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_state=16,
        num_macro_states=3, num_depth_modes=2, num_proposals=2, lora_rank=4, max_seq_len=64,
    )
    return System3Model(cfg)


def make_cfg(tmp_path, **overrides):
    kwargs = dict(
        data_path=str(tmp_path / "tokens.uint16"),
        vocab_size=32, seq_len=8, batch_size=2, val_fraction=0.2,
        max_steps=3, steps_per_run=2, eval_every=2, save_every=2, log_every=1,
        eval_batches=1, ckpt_dir=str(tmp_path / "ckpt"), device="cpu", use_amp=False, seed=0,
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_token_chunk_dataset_shifts_labels(tmp_path):
    path = tmp_path / "tokens.uint16"
    write_shard(path)
    ds = TokenChunkDataset(str(path), seq_len=8)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (8,) and y.shape == (8,)
    x, y = ds[1]
    # y is x shifted by exactly one token
    assert torch.equal(y[:-1], x[1:])


def test_build_dataloaders(tmp_path):
    path = tmp_path / "tokens.uint16"
    write_shard(path, n_tokens=256)
    cfg = make_cfg(tmp_path)
    train_loader, val_loader = build_dataloaders(cfg)
    assert len(train_loader) > 0
    x, y = next(iter(train_loader))
    assert x.shape == (cfg.batch_size, cfg.seq_len)
    assert y.shape == (cfg.batch_size, cfg.seq_len)


def test_evaluate_returns_finite(tmp_path):
    path = tmp_path / "tokens.uint16"
    write_shard(path)
    cfg = make_cfg(tmp_path)
    _, val_loader = build_dataloaders(cfg)
    model = tiny_model()
    val = evaluate(model, val_loader, cfg, max_batches=1)
    assert np.isfinite(val)


def test_checkpoint_roundtrip(tmp_path):
    model = tiny_model()
    cfg = make_cfg(tmp_path)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    ckpt = CheckpointManager(model, optimizer, scheduler, cfg)
    state = new_train_state()
    state["step"] = 7
    state["best_val"] = 1.23

    assert ckpt.load(ckpt.latest_path, new_train_state()) is False  # nothing written yet
    ckpt.save(ckpt.latest_path, state, numbered=True)

    restored = new_train_state()
    assert ckpt.load(ckpt.latest_path, restored) is True
    assert restored["step"] == 7
    assert restored["best_val"] == pytest.approx(1.23)
    assert ckpt.restore_best() is None  # no best.pt yet


def test_train_loop_smoke(tmp_path):
    path = tmp_path / "tokens.uint16"
    write_shard(path, n_tokens=512)
    cfg = make_cfg(tmp_path)
    train_loader, val_loader = build_dataloaders(cfg)

    model = tiny_model()
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    ckpt = CheckpointManager(model, optimizer, scheduler, cfg)
    state = new_train_state()

    train_loop(model, optimizer, scheduler, train_loader, val_loader, ckpt, state, cfg)

    assert state["step"] == cfg.steps_per_run
    assert ckpt.latest_path.exists()
    assert len(state["history"]["train_ntp"]) == cfg.steps_per_run


def test_audit_runs(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    from system3.audit import run_system3_audit

    path = tmp_path / "tokens.uint16"
    write_shard(path)
    cfg = make_cfg(tmp_path)
    _, val_loader = build_dataloaders(cfg)

    model = tiny_model()
    fig, metrics = run_system3_audit(model, val_loader, device="cpu", use_amp=False, num_batches=1)
    assert "verdict" in metrics
    assert set(metrics["losses"]) == {"normal_eval", "calibrated_gate", "zero_state", "scrambled_state"}
    assert fig is not None
