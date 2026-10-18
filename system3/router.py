from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import System3Config
from .gate import CalibratedSwitchGate
from .outputs import RouterOutput
from .utils import RMSNorm, _masked_mean, _unit_rms, log_sinkhorn_knopp


class ModularSystem12Router(nn.Module):
    def __init__(self, config: System3Config) -> None:
        super().__init__()
        self.config = config
        D, S = config.d_model, config.d_state
        M1, M2, K = config.num_macro_states, config.num_depth_modes, config.num_proposals
        self.num_compositional = M1 * M2

        self.macro_codebook = nn.Parameter(torch.randn(M1, S) * config.initializer_range)
        self.depth_codebook = nn.Parameter(torch.randn(M2, S) * config.initializer_range)
        self.codebook_fuse = nn.Linear(S, S, bias=False)
        self.s_init = nn.Parameter(torch.randn(1, S) * config.initializer_range)

        self.forward_predictor = nn.Sequential(nn.Linear(D + S, D), nn.SiLU(), nn.Linear(D, D))

        self.joint_norm = RMSNorm(S)
        self.ctx_proj = nn.Linear(D, S, bias=False)
        self.state_feedback_proj = nn.Linear(S, S, bias=False)

        self.continuous_nuance_gen = nn.Sequential(nn.Linear(S, K * S), nn.SiLU(), nn.Linear(K * S, K * S))
        self.nuance_scale = nn.Parameter(torch.tensor(0.25))

        self.sys1_metric = nn.Linear(S, S, bias=False)
        self.sys1_value_head = nn.Sequential(nn.Linear(S, S // 2), nn.SiLU(), nn.Linear(S // 2, 1))
        self.transition_smoothness = nn.Parameter(torch.tensor(0.1))

        self.switch_gate = CalibratedSwitchGate(config)

    def build_compositional_manifold(self) -> Tensor:
        if self.config.num_depth_modes == 1:
            return F.normalize(self.macro_codebook.float(), p=2, dim=-1).to(self.macro_codebook.dtype)
        m1 = self.macro_codebook.unsqueeze(1)
        m2 = self.depth_codebook.unsqueeze(0)
        comp = m1 + m2 + self.codebook_fuse(m1 * m2)
        return F.normalize(comp.view(-1, self.config.d_state).float(), p=2, dim=-1).to(self.macro_codebook.dtype)

    def forward(
        self,
        h_seq: Tensor,
        valid_mask: Tensor,
        *,
        embed_seq: Optional[Tensor] = None,
        temperature: float = 0.7,
        deterministic: bool = False,
    ) -> RouterOutput:
        B, T, D = h_seq.shape
        S, K = self.config.d_state, self.config.num_proposals
        device = h_seq.device

        manifold = self.build_compositional_manifold()
        # Predictive-coding target. The next token embedding is learnable from the
        # previous position; the post-attention context is not (consecutive positions
        # correlate at cos ~0.33 and the predictor plateaued at ~0.48), which left
        # `surprise` a near-constant and the switch gate with no usable input.
        target_seq = embed_seq if (self.config.surprise_target == "embedding" and embed_seq is not None) else h_seq
        h_proj_all = self.ctx_proj(h_seq)

        s_prev = self.s_init.expand(B, -1)
        tau_prev = torch.zeros(B, 1, device=device, dtype=h_seq.dtype)
        h_prev = h_seq[:, 0, :]

        active_macro = torch.zeros(B, device=device, dtype=torch.long)
        active_depth = torch.zeros(B, device=device, dtype=torch.long)
        active_prop = torch.zeros(B, device=device, dtype=torch.long)

        states_seq, gates_seq, open_prob_seq, surprise_seq, dwell_seq = [], [], [], [], []
        macro_seq, depth_seq, prop_seq = [], [], []
        comp_logits_seq, comp_weights_seq, proposals_seq, pred_h_seq = [], [], [], []

        for t in range(T):
            h_t = h_seq[:, t, :]
            h_p = h_proj_all[:, t, :]
            valid_step = valid_mask[:, t : t + 1]
            is_valid = valid_step.to(h_seq.dtype)

            # 1. Dimension-Invariant Unit-RMS Predictive Residual (Bounded in [0, 4])
            pred_h_t = self.forward_predictor(torch.cat([h_prev, s_prev], dim=-1))
            pred_h_seq.append(pred_h_t)
            if t == 0:
                raw_surprise = torch.ones(B, 1, device=device, dtype=h_seq.dtype)
                norm_surprise = torch.zeros(B, 1, device=device, dtype=h_seq.dtype)
            else:
                # Unit-RMS difference has mean squared error in [0, 4] invariant to D!
                u_true = _unit_rms(target_seq[:, t, :].detach())
                u_pred = _unit_rms(pred_h_t)
                raw_surprise = (u_true - u_pred).pow(2).mean(dim=-1, keepdim=True).to(h_seq.dtype)
                norm_surprise = self.switch_gate.normalize_surprise(raw_surprise, valid_step)

            # 2. Switch Gate Decision
            g_t, open_prob = self.switch_gate(
                h_t, s_prev, norm_surprise, tau_prev, temperature=temperature, deterministic=deterministic
            )
            if t == 0:
                g_t = torch.ones_like(g_t)
                open_prob = torch.zeros_like(open_prob)
            g_t = g_t * is_valid

            # 3. Proposal Generation (Closed-Loop vs Open-Loop toggleable for ablation)
            if self.config.closed_loop_proposals:
                u_t = self.joint_norm(F.silu(h_p + self.state_feedback_proj(s_prev)))
            else:
                u_t = self.joint_norm(F.silu(h_p))

            comp_logits = torch.matmul(u_t, manifold.t()) / math.sqrt(S)
            comp_logits_seq.append(comp_logits)

            top_vals, top_idx = torch.topk(comp_logits, k=K, dim=-1)
            discrete_anchors = manifold[top_idx]

            raw_nuance = self.continuous_nuance_gen(u_t).view(B, K, S)
            norm_nuance = F.normalize(raw_nuance.float(), p=2, dim=-1, eps=1e-6).to(h_seq.dtype)
            proposals_seq.append(norm_nuance)

            if self.config.state_mode == "discrete":
                candidates = discrete_anchors
            elif self.config.state_mode == "continuous":
                candidates = norm_nuance
            else:
                candidates = discrete_anchors + self.nuance_scale * norm_nuance

            # 4. System 1 Candidate Scoring
            q_sys1 = self.sys1_metric(u_t).unsqueeze(1)
            bilinear_score = (q_sys1 * candidates).sum(dim=-1) / math.sqrt(S)
            value_score = self.sys1_value_head(candidates).squeeze(-1)
            jump_cost = (candidates - s_prev.unsqueeze(1)).pow(2).mean(dim=-1)
            sys1_logits = bilinear_score + value_score + top_vals - self.transition_smoothness * jump_cost

            if not deterministic:
                sys1_weights = F.gumbel_softmax(sys1_logits.float(), tau=temperature, hard=True, dim=-1).to(h_seq.dtype)
            else:
                idx = sys1_logits.argmax(dim=-1, keepdim=True)
                sys1_weights = torch.zeros_like(sys1_logits).scatter_(-1, idx, 1.0)

            s_candidate = torch.einsum("bk,bks->bs", sys1_weights, candidates)

            # 5. State, Dwell, and Active ID Update
            s_t = g_t * s_candidate + (1.0 - g_t) * s_prev
            tau_t = (1.0 - g_t) * (tau_prev + 1.0) * is_valid

            switched_bool = g_t.squeeze(-1) > 0.5
            win_slot = sys1_weights.detach().argmax(dim=-1)
            win_comp_id = torch.gather(top_idx, dim=-1, index=win_slot.unsqueeze(-1)).squeeze(-1)
            new_macro = win_comp_id // self.config.num_depth_modes
            new_depth = win_comp_id % self.config.num_depth_modes

            active_macro = torch.where(switched_bool, new_macro, active_macro)
            active_depth = torch.where(switched_bool, new_depth, active_depth)
            active_prop = torch.where(switched_bool, win_slot, active_prop)

            # Record full-manifold one-hot dispatch for Switch-MoE balance mode
            comp_one_hot = F.one_hot(win_comp_id, self.num_compositional).to(h_seq.dtype)
            comp_weights_seq.append(comp_one_hot)

            states_seq.append(s_t)
            gates_seq.append(g_t.squeeze(-1))
            open_prob_seq.append(open_prob.squeeze(-1) * is_valid.squeeze(-1))
            surprise_seq.append(raw_surprise.squeeze(-1))
            dwell_seq.append(tau_t.squeeze(-1))
            macro_seq.append(active_macro.masked_fill(~valid_mask[:, t], -1))
            depth_seq.append(active_depth.masked_fill(~valid_mask[:, t], -1))
            prop_seq.append(active_prop.masked_fill(~valid_mask[:, t], -1))

            s_prev = s_t
            tau_prev = tau_t
            h_prev = torch.where(valid_step, h_t, h_prev)

        states = torch.stack(states_seq, dim=1)
        switch_trace = torch.stack(gates_seq, dim=1)
        open_probs = torch.stack(open_prob_seq, dim=1)
        all_comp_logits = torch.stack(comp_logits_seq, dim=1)
        all_comp_weights = torch.stack(comp_weights_seq, dim=1)
        all_proposals = torch.stack(proposals_seq, dim=1)
        pred_h = torch.stack(pred_h_seq, dim=1)

        # --- Modular Auxiliary Losses ---
        zero = h_seq.new_zeros((), dtype=torch.float32)

        # 1. Balance Loss ("none" | "switch" | "sinkhorn") over switched/active positions
        switch_mask = valid_mask & (switch_trace > 0.5)
        if self.config.state_mode in ("discrete", "hybrid") and switch_mask.any():
            if self.config.balance_mode == "sinkhorn":
                valid_logits = all_comp_logits[switch_mask].float()
                q_ot = log_sinkhorn_knopp(
                    valid_logits.detach(), eps=self.config.sinkhorn_eps, n_iters=self.config.sinkhorn_iters
                )
                balance_loss = -(q_ot * F.log_softmax(valid_logits, dim=-1)).sum(dim=-1).mean()
            elif self.config.balance_mode == "switch":
                probs = all_comp_logits.float().softmax(dim=-1)
                mask_3d = switch_mask.unsqueeze(-1).float()
                denom = mask_3d.sum().clamp_min(1.0)
                frac = (all_comp_weights.float() * mask_3d).sum(dim=(0, 1)) / denom
                mean_p = (probs * mask_3d).sum(dim=(0, 1)) / denom
                balance_loss = self.num_compositional * torch.sum(frac * mean_p)
            else:
                balance_loss = zero
        else:
            balance_loss = zero

        # 2. Proposal Diversity Loss ("none" | "orthogonality" | "dpp")
        if self.config.state_mode in ("continuous", "hybrid") and K > 1:
            gram = torch.matmul(all_proposals.float(), all_proposals.float().transpose(-1, -2))
            if self.config.diversity_mode == "dpp":
                eye = torch.eye(K, device=device, dtype=torch.float32).view(1, 1, K, K) * 1e-3
                _, logabsdet = torch.linalg.slogdet(gram + eye)
                diversity_loss = _masked_mean(-logabsdet, valid_mask)
            elif self.config.diversity_mode == "orthogonality":
                off_diag = ~torch.eye(K, device=device, dtype=torch.bool)
                pair_err = gram[..., off_diag].square().mean(dim=-1)
                diversity_loss = _masked_mean(pair_err, valid_mask)
            else:
                diversity_loss = zero
        else:
            diversity_loss = zero

        # 3. Predictive Coding JEPA & Dwell/Sparsity Loss
        transitions_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        if T > 1 and transitions_valid.any():
            if self.config.jepa_weight > 0.0:
                # Unit-RMS MSE / 2 == 1 - CosineSimilarity (strictly consistent with gate surprise!)
                u_next = _unit_rms(target_seq[:, 1:].detach())
                u_hat = _unit_rms(pred_h[:, 1:])
                jepa_err = 0.5 * (u_next - u_hat).pow(2).mean(dim=-1)
                jepa_loss = _masked_mean(jepa_err, transitions_valid)
            else:
                jepa_loss = zero

            expected_open_rate = _masked_mean(open_probs[:, 1:].float(), transitions_valid)
            if self.config.dwell_mode == "rate_target":
                dwell_loss = (expected_open_rate - self.config.target_switch_rate).square()
            elif self.config.dwell_mode == "l0_penalty":
                dwell_loss = expected_open_rate  # True L0 sparsity penalty (Louizos et al.)
            else:
                dwell_loss = zero

            actual_rate = _masked_mean(switch_trace[:, 1:].float(), transitions_valid)
        else:
            jepa_loss, dwell_loss, actual_rate = zero, zero, zero

        return RouterOutput(
            states=states,
            active_macro_ids=torch.stack(macro_seq, dim=1),
            active_depth_ids=torch.stack(depth_seq, dim=1),
            active_proposal_ids=torch.stack(prop_seq, dim=1),
            switch_trace=switch_trace,
            open_probabilities=open_probs,
            surprise_trace=torch.stack(surprise_seq, dim=1),
            dwell_trace=torch.stack(dwell_seq, dim=1),
            balance_loss=balance_loss,
            diversity_loss=diversity_loss,
            jepa_loss=jepa_loss,
            dwell_loss=dwell_loss,
            actual_switch_rate=actual_rate,
        )
