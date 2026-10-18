import pytest
import torch

from system3 import System3Config, System3Model


def tiny_config(**overrides) -> System3Config:
    base = dict(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_state=16,
        num_macro_states=3,
        num_depth_modes=2,
        num_proposals=2,
        lora_rank=4,
        max_seq_len=32,
    )
    base.update(overrides)
    return System3Config(**base)


def make_batch(batch: int = 2, seq: int = 8, vocab: int = 32):
    input_ids = torch.randint(0, vocab, (batch, seq))
    labels = input_ids.clone()
    return input_ids, labels


def test_config_validation_rejects_bad_geometry():
    with pytest.raises(ValueError):
        System3Config(d_model=30, n_heads=4)  # d_model not divisible by n_heads
    with pytest.raises(ValueError):
        System3Config(n_heads=8, n_kv_heads=3)  # n_heads not divisible by n_kv_heads
    with pytest.raises(ValueError):
        System3Config(n_layers=1)  # need a stem + >=1 block


def test_forward_shapes_and_loss_components():
    model = System3Model(tiny_config())
    input_ids, labels = make_batch()
    out = model(input_ids, labels)

    assert out.logits.shape == (2, 8, 32)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert set(out.loss_components) == {"total", "next_token", "balance", "diversity", "jepa", "dwell"}
    assert out.router.states.shape == (2, 8, 16)
    assert out.router.switch_trace.shape == (2, 8)


def test_backward_populates_grads():
    model = System3Model(tiny_config())
    input_ids, labels = make_batch()
    out = model(input_ids, labels)
    out.loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert model.router.s_init.grad is not None
    assert torch.isfinite(model.router.s_init.grad).all()


def test_eval_is_deterministic():
    model = System3Model(tiny_config())
    model.eval()
    input_ids, _ = make_batch()
    with torch.no_grad():
        a = model(input_ids).logits
        b = model(input_ids).logits
    assert torch.allclose(a, b)


@pytest.mark.parametrize(
    "changes",
    [
        {"state_mode": "discrete"},
        {"state_mode": "continuous"},
        {"closed_loop_proposals": False},
        {"use_surprise_gate": False},
        {"gate_type": "softmax"},
        {"balance_mode": "switch"},
        {"balance_mode": "none"},
        {"diversity_mode": "orthogonality"},
        {"diversity_mode": "none"},
        {"dwell_mode": "l0_penalty"},
        {"dwell_mode": "none"},
        {"use_state_scan": False},
        {"use_lora_swiglu": False},
        {"use_head_temp_steering": False},
        {"surprise_target": "hidden"},
    ],
)
def test_ablation_switches_run(changes):
    model = System3Model(tiny_config(**changes))
    input_ids, labels = make_batch()
    out = model(input_ids, labels)
    assert torch.isfinite(out.loss)


def test_minimal_baseline_runs():
    model = System3Model(System3Config.minimal_baseline(**tiny_config_kwargs()))
    input_ids, labels = make_batch()
    out = model(input_ids, labels)
    assert torch.isfinite(out.loss)


def tiny_config_kwargs():
    return dict(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_state=16,
        num_macro_states=3,
        num_depth_modes=2,
        num_proposals=2,
        lora_rank=4,
        max_seq_len=32,
    )


def test_right_padded_attention_mask():
    model = System3Model(tiny_config())
    model.eval()
    input_ids = torch.randint(0, 32, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[1, 5:] = 0  # right padding: row 1 keeps its leading tokens valid
    with torch.no_grad():
        out = model(input_ids, attention_mask=mask)
    assert out.logits.shape == (2, 8, 32)
    assert torch.isfinite(out.logits).all()


def test_left_padding_without_leading_token_raises():
    model = System3Model(tiny_config())
    input_ids = torch.randint(0, 32, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[1, :3] = 0  # row 1 starts on a pad token -> not supported
    with pytest.raises(ValueError):
        model(input_ids, attention_mask=mask)


def test_with_ablation_helper():
    cfg = System3Config.minimal_baseline(**tiny_config_kwargs())
    steered = cfg.with_ablation(use_state_scan=True, use_lora_swiglu=True, jepa_weight=0.05)
    assert isinstance(steered, System3Config)
    assert steered.use_state_scan and steered.use_lora_swiglu
    assert cfg.use_state_scan is False  # original untouched
