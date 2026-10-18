from __future__ import annotations

import math
from typing import Optional

import torch
from torch.nn import functional as F

from .model import System3Model


def run_system3_audit(
    model: System3Model,
    val_loader,
    *,
    device: str = "cpu",
    use_amp: bool = False,
    num_batches: int = 15,
    ckpt_label: str = "",
    step: Optional[int] = None,
):
    """Diagnostic audit of a trained System3 model.

    Runs the driver-vs-passenger counterfactual ablation, a layer-by-layer weight
    and activation audit, and renders a 3-panel figure. Returns (figure, metrics).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cfg = model.config
    device_type = device.split(":")[0]
    amp = use_amp and device_type == "cuda"
    model.to(device).eval()

    print(f"=== System3 Diagnostic Audit ({ckpt_label or 'model'} @ step {step if step is not None else '?'}) ===\n")

    # -------------------------------------------------------------------------
    # 1. DRIVER VS. PASSENGER COUNTERFACTUAL ABLATION
    # -------------------------------------------------------------------------
    modes = ("normal_eval", "calibrated_gate", "zero_state", "scrambled_state")
    losses = {m: 0.0 for m in modes}
    switch_rates = {"normal_eval": 0.0, "calibrated_gate": 0.0}
    macro_counts = torch.zeros(cfg.num_macro_states, dtype=torch.long)

    # Accumulators for live activation steering ratios per layer
    n_steered = len(model.layers)
    lora_to_base_ratio = torch.zeros(n_steered)
    ada_scale_mag = torch.zeros(n_steered)
    n_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= num_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            B, T = x.shape
            valid_mask = torch.ones_like(x, dtype=torch.bool)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
                # Shared perceptual stem (Layer 0)
                emb = model.token_embedding(x)
                s0 = model.neutral_state.expand(B, T, -1)
                context = emb + model.stem_attn(model.stem_norm1(emb), s0, valid_mask, True)
                context = context + model.stem_mlp(model.stem_norm2(context), s0)

                # A. Standard deterministic router (.eval() threshold = 0.5)
                routed_eval = model.router(
                    context, valid_mask, embed_seq=emb, temperature=0.7, deterministic=True
                )
                switch_rates["normal_eval"] += float(routed_eval.actual_switch_rate)

                # B. Calibrated router (stochastic Hard-Concrete matching training distribution)
                routed_cal = model.router(
                    context, valid_mask, embed_seq=emb, temperature=0.5, deterministic=False
                )
                switch_rates["calibrated_gate"] += float(routed_cal.actual_switch_rate)
                macro_counts += torch.bincount(
                    routed_cal.active_macro_ids.cpu().flatten(), minlength=cfg.num_macro_states
                )

                # Helper to run the steered transformer stack under different state interventions
                def run_stack(state_seq, sw_trace, record_activations=False):
                    h, s = context, state_seq
                    for l_idx, layer in enumerate(model.layers):
                        if record_activations:
                            # Measure AdaRMSNorm modulation magnitude & LoRA-to-Base FFN ratio
                            g1, b1, g2, b2 = layer.ada_mod(s).chunk(4, dim=-1)
                            ada_scale_mag[l_idx] += float(
                                (g1.float().norm(dim=-1).mean() + g2.float().norm(dim=-1).mean()) * 0.5
                            )
                            h_mlp_in = (1.0 + g2) * layer.norm_mlp(h) + b2
                            base_out = layer.mlp.w_down(
                                F.silu(layer.mlp.w_gate(h_mlp_in)) * layer.mlp.w_up(h_mlp_in)
                            )
                            if layer.mlp.use_lora:
                                delta_out = layer.mlp.lora_up(
                                    layer.mlp.lora_down(h_mlp_in)
                                    * F.silu(layer.mlp.state_to_lora_gate(s))
                                )
                                ratio = delta_out.float().norm(dim=-1) / (
                                    base_out.float().norm(dim=-1) + 1e-6
                                )
                                lora_to_base_ratio[l_idx] += float(ratio.mean())
                        h, s = layer(h, s, sw_trace, valid_mask, True)

                    logits = model.lm_head(model.final_norm(h))
                    return float(
                        F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
                    )

                # 1. Normal eval
                losses["normal_eval"] += run_stack(
                    routed_eval.states, routed_eval.switch_trace, record_activations=True
                )
                # 2. Calibrated gate
                losses["calibrated_gate"] += run_stack(routed_cal.states, routed_cal.switch_trace)
                # 3. Zeroed states (ablate all steering back to unsteered backbone)
                losses["zero_state"] += run_stack(
                    torch.zeros_like(routed_eval.states), torch.zeros_like(routed_eval.switch_trace)
                )
                # 4. Scrambled states (permute across batch & time to test state specificity)
                perm_b = torch.randperm(B, device=device)
                perm_t = torch.randperm(T, device=device)
                scrambled_s = routed_cal.states[perm_b][:, perm_t]
                losses["scrambled_state"] += run_stack(scrambled_s, routed_cal.switch_trace)

            n_batches += 1

    for m in modes:
        losses[m] /= max(n_batches, 1)
    switch_rates["normal_eval"] /= max(n_batches, 1)
    switch_rates["calibrated_gate"] /= max(n_batches, 1)
    lora_to_base_ratio /= max(n_batches, 1)
    ada_scale_mag /= max(n_batches, 1)

    # Compute macro-state perplexity (effective number of states used out of N)
    macro_probs = macro_counts.float() / macro_counts.sum().clamp_min(1)
    macro_entropy = -(macro_probs * (macro_probs + 1e-12).log()).sum().item()
    eff_states = math.exp(macro_entropy)

    # Print table
    delta_zero = losses["zero_state"] - losses["normal_eval"]
    delta_scram = losses["scrambled_state"] - losses["normal_eval"]
    print(f"  Normal Eval Loss (deterministic >0.5): {losses['normal_eval']:.4f} nats | Switch rate: {switch_rates['normal_eval']*100:5.2f}%")
    print(f"  Calibrated Gate Loss (stochastic):     {losses['calibrated_gate']:.4f} nats | Switch rate: {switch_rates['calibrated_gate']*100:5.2f}%")
    print(f"  Zeroed State Loss (s_t = 0):           {losses['zero_state']:.4f} nats | d = {delta_zero:+.4f} nats")
    print(f"  Scrambled State Loss (shuffled s_t):   {losses['scrambled_state']:.4f} nats | d = {delta_scram:+.4f} nats")
    print(f"  Active Macro-State Usage:              {eff_states:.2f} / {cfg.num_macro_states} effective states")
    print(f"  Macro Distribution:                    {[round(p.item()*100, 1) for p in macro_probs]}%\n")

    # Automated verdict
    if delta_zero > 0.03 or delta_scram > 0.05:
        verdict = "WIN (ACTIVE DRIVER): Ablating or scrambling s_t degrades validation loss - the backbone actively relies on the router!"
    elif delta_zero > 0.005 or delta_scram > 0.01:
        verdict = "EARLY AWAKENING: The backbone is starting to use s_t, but steering pathways are still small."
    else:
        verdict = "PASSENGER: Zeroing/scrambling s_t barely changes loss (<0.005 nats) - the backbone is ignoring the router so far."
    print(f"  >>> VERDICT: {verdict}\n")

    # -------------------------------------------------------------------------
    # 2. LAYER-BY-LAYER WEIGHT & ACTIVATION AUDIT
    # -------------------------------------------------------------------------
    print(f"  {'Layer':>5} | {'ada_mod W RMS':>13} | {'Ada Gate ||g||':>14} | {'lora_up W RMS':>13} | {'LoRA/Base FFN %':>15}")
    print("  " + "-" * 72)
    for l_idx, layer in enumerate(model.layers):
        ada_rms = layer.ada_mod.weight.float().pow(2).mean().sqrt().item()
        lora_rms = layer.mlp.lora_up.weight.float().pow(2).mean().sqrt().item() if layer.mlp.use_lora else 0.0
        print(
            f"  {l_idx+1:5d} | {ada_rms:13.5f} | {ada_scale_mag[l_idx].item():14.4f} | "
            f"{lora_rms:13.5f} | {lora_to_base_ratio[l_idx].item()*100:14.2f}%"
        )

    # -------------------------------------------------------------------------
    # 3. VISUALIZATION
    # -------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle(f"System 3 Diagnostic Audit (Step {step if step is not None else '?'})", fontsize=12, fontweight="bold")

    # Panel 1: Counterfactual Loss Delta
    labels = ["Calibrated\nGate", "Zeroed\nState (s=0)", "Scrambled\nState"]
    deltas = [losses["calibrated_gate"] - losses["normal_eval"], delta_zero, delta_scram]
    colors = ["#2a9d8f", "#e76f51", "#c1121f"]
    ax1.bar(labels, deltas, color=colors, alpha=0.88, width=0.55)
    ax1.axhline(0.0, color="black", lw=1, ls="--")
    ax1.set_ylabel("d Validation Loss vs Normal (nats)\n(Higher = Worse when ablated)")
    ax1.set_title("1. Driver vs. Passenger Test", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2: Layer-by-Layer Steering Strength
    layers_x = np.arange(1, n_steered + 1)
    ax2.plot(layers_x, ada_scale_mag.numpy(), "o-", color="#1d3557", lw=2, label="AdaRMSNorm ||g||")
    ax2.plot(layers_x, lora_to_base_ratio.numpy() * 100, "s-", color="#e63946", lw=2, label="LoRA / Base FFN (%)")
    ax2.set_xlabel("Steered Transformer Layer")
    ax2.set_ylabel("Activation Magnitude / Ratio")
    ax2.set_title("2. Steering Circuit Wake-Up", fontsize=10, fontweight="bold")
    ax2.set_xticks(layers_x)
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.25)

    # Panel 3: Macro Codebook Utilization
    base_labels = ["Plan", "Think", "Decomp", "Exec", "Verify", "Concl"]
    state_labels = [base_labels[i] if i < len(base_labels) else f"S{i}" for i in range(cfg.num_macro_states)]
    ax3.bar(state_labels, (macro_probs * 100).numpy(), color="#457b9d", alpha=0.88)
    ax3.axhline(100.0 / cfg.num_macro_states, color="#c1121f", ls="--", lw=1.2,
                label=f"Ideal Uniform ({100.0 / cfg.num_macro_states:.1f}%)")
    ax3.set_ylabel("Token Assignment Share (%)")
    ax3.set_title(f"3. Macro State Balance ({eff_states:.1f}/{cfg.num_macro_states} active)", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8.5)
    ax3.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    metrics = {
        "losses": losses,
        "switch_rates": switch_rates,
        "delta_zero": delta_zero,
        "delta_scram": delta_scram,
        "eff_states": eff_states,
        "macro_probs": macro_probs,
        "verdict": verdict,
    }
    return fig, metrics
