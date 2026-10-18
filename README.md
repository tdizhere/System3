# System3

New paradigm of architecture of neural network that embeds a System 1 module (a
JEPA-style predictive-coding, state-routing router) inside a typical
autoregressive System 2 transformer.

The **router** maintains a latent macro-state `s_t` that switches on surprising
tokens (predictive-coding residual), scores discrete+continuous state proposals
with a System 1 value/bilinear metric, and steers every transformer layer through
AdaRMSNorm modulation, a state-routed LoRA-SwiGLU, and per-head attention
temperature. Everything is ablatable one switch at a time via `System3Config`.

## Layout

```
system3/
  config.py     System3Config + ablation switches, minimal_baseline()
  outputs.py    RouterOutput, System3Output dataclasses
  utils.py      RMSNorm, RoPE, associative scan, Sinkhorn, helpers
  gate.py       CalibratedSwitchGate (softmax / hard-concrete)
  router.py     ModularSystem12Router (System 1 state router)
  attention.py  StateSteeredGQA (state-steered grouped-query attention)
  mlp.py        StateRoutedSwiGLU (state-routed LoRA-SwiGLU)
  block.py      CoupledSystem3Block (used by the steered stack)
  model.py      System3Model (stem + router + steered stack + tied LM head)
  data.py       TokenChunkDataset + build_dataloaders (uint16 token shards)
  training.py   TrainConfig, optimizer/scheduler, checkpointing, train_loop
  audit.py      run_system3_audit (driver-vs-passenger diagnostic + figure)
train.py        Training / resume entrypoint
audit.py        Diagnostic audit entrypoint
plot_history.py Training-curve plotting entrypoint
tests/          Smoke tests (forward/backward, ablations, data, training, audit)
```

## Install

```
pip install -r requirements.txt          # torch, numpy, matplotlib
pip install -e ".[dev]"                   # + pytest, editable
```

## Use

```python
from system3 import System3Config, System3Model

model = System3Model(System3Config(d_model=512, n_layers=8))
out = model(input_ids, labels=labels)     # System3Output(logits, loss, loss_components, router)

# Rung-0 baseline, then toggle one mechanism at a time
cfg = System3Config.minimal_baseline()
cfg = cfg.with_ablation(use_state_scan=True, use_lora_swiglu=True)
```

## Train

Expects a flat `uint16` token shard (e.g. from your tokenizer prep step); the tail
`val_fraction` is held out for validation.

```
python train.py --data tokens.uint16 --steps-per-run 200
```

Re-running adds `steps_per_run` steps to the existing run (auto-resumes from
`checkpoints/latest.pt`). Inspect with:

```
python plot_history.py --ckpt-dir checkpoints
python audit.py --data tokens.uint16          # driver-vs-passenger diagnostic
```

## Test

```
python -m pytest -q
```
