"""Config-A agent: flow-based intent model + deterministic MLP action decoder.

Architecture overview:
    ┌────────────────────────────┐
    │   Shared Context Encoder   │  obs (B,To,obs_dim) → obs_emb (B,To,emb_dim)
    └──────────┬─────────────────┘
               │  shared conditioning
    ┌──────────▼─────────────┐      ┌──────────────────────────────┐
    │  Flow Intent Model     │      │  MLP Action Decoder          │
    │  FlowMap(net,          │      │  (obs_emb, intent)           │
    │    Ta=1, D=intent_dim) │      │  → action (B, horizon, D)    │
    │  noise → intent (7D)   │      └──────────────────────────────┘
    └────────────────────────┘              ↑ intent conditioning ↗

Training (two independent gradient steps per iteration):
    Step 1 — flow intent loss:
        flow_loss(intent_gt.unsqueeze(1), obs) → ∂/∂(encoder + intent_flow_map)

    Step 2 — action decoder MSE:
        MSE(action_decoder(obs_emb.detach(), intent_gt), act_gt) → ∂/∂(action_decoder)

    The encoder is updated only via the flow intent loss.  The action decoder
    conditions on obs_emb with stopped gradient so it cannot destabilise the
    encoder's intent-tuned representation.

Inference (Config A):
    1. obs → encoder → obs_emb           # (B, To, emb_dim)
    2. noise → intent ODE (obs_emb)      # (B, 1, intent_dim) → squeeze → (B, intent_dim)
    3. (obs_emb, intent) → action        # (B, horizon, act_dim)

    Multiple intent samples can be drawn at inference (num_intent_samples > 1);
    currently the first sample is used.  Beam search / best-of-k is a future
    extension.

Config comparison:
    Config B (baseline):  FlowMap (action) + optional IntentPredictor (MLP)
    Config A (this file): FlowMap (intent) + MLPActionDecoder (deterministic action)

Both variants use the same context encoder and Hydra task config.  Switch via:
    network.arch_variant: "flow_intent"   # Config A
    network.arch_variant: "flow_action"   # Config B  (default, existing behaviour)
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy

import loguru
import torch
import torch.nn as nn

from mip.config import Config
from mip.encoders import IdentityEncoder
from mip.flow_map import FlowMap
from mip.intent_encoder import IntentEncoder
from mip.interpolant import Interpolant
from mip.losses import get_loss_fn
from mip.mlp_action_decoder import MLPActionDecoder
from mip.network_utils import get_encoder, get_network
from mip.torch_utils import at_least_ndim, report_parameters


class FlowIntentAgent:
    """Config-A: flow intent model + MLP action decoder.

    Public attributes (needed by train_robomimic.py):
        intent_optimizer  — AdamW over encoder + intent_flow_map
        action_optimizer  — AdamW over action_decoder only
        encoder           — shared context encoder (train copy)
        encoder_ema       — EMA copy of encoder (used at eval)

    The agent does NOT expose a single .optimizer, unlike TrainingAgent.
    train_robomimic.py detects arch_variant == "flow_intent" and uses
    agent.intent_optimizer for the main LR scheduler.
    """

    def __init__(self, config: Config):
        self.config = config
        device = config.optimization.device

        # ── Shared context encoder ────────────────────────────────────────
        # Identical to Config B: maps obs (B, To, obs_dim) → (B, To, emb_dim).
        # IMPORTANT: obs_dim here is base_obs_dim (NOT augmented with intent),
        # because Config A never appends intent to the raw observation.
        self.encoder = get_encoder(config.network, config.task).to(device)
        self.encoder_ema = deepcopy(self.encoder).requires_grad_(False)
        report_parameters(self.encoder, model_name="Shared Encoder [Config A]")

        # ── Optional intent sequence encoder (encoded_mean variant) ──────
        # When intent_type == "encoded_mean", the dataset provides (N, 7) per-step
        # sequences instead of a pre-averaged 7D vector.  A small per-step MLP
        # encodes each step → (N, intent_emb_dim), then mean-pool → (intent_emb_dim).
        # The flow model then operates in intent_emb_dim space, not raw 7D space.
        #
        # For all other intent_type values ("mean", "final", "sequence"), this is None
        # and the flow model operates directly in the raw intent_dim (e.g. 7D) space.
        self._intent_type = getattr(config.task, "intent_type", "mean")
        self.intent_seq_encoder: IntentEncoder | None = None
        if self._intent_type == "encoded_mean":
            _raw_intent_dim = config.task.intent_dim   # 7 (pos3+quat4)
            _intent_emb_dim = getattr(config.task, "intent_emb_dim", 64)
            self.intent_seq_encoder = IntentEncoder(
                raw_intent_dim=_raw_intent_dim,
                intent_emb_dim=_intent_emb_dim,
            ).to(device)
            self.intent_seq_encoder_ema = deepcopy(self.intent_seq_encoder).requires_grad_(False)
            effective_intent_dim = _intent_emb_dim
            report_parameters(self.intent_seq_encoder, model_name="Intent Seq Encoder [Config A, encoded_mean]")
        else:
            self.intent_seq_encoder_ema = None
            effective_intent_dim = config.task.intent_dim  # e.g. 7

        # ── Flow intent model ─────────────────────────────────────────────
        # Reuses the standard FlowMap + network stack, but in intent space:
        #   Ta       = 1                   (intent is a single vector, not a trajectory)
        #   act_dim  = effective_intent_dim (7 for raw; intent_emb_dim for encoded_mean)
        #   obs_dim  = emb_dim             (encoder output, same as Config B)
        #   To       = obs_steps           (unchanged)
        #
        # A temporary TaskConfig copy carries the modified dims so that
        # get_network() builds the right MLP without touching the real config.
        intent_task_cfg = deepcopy(config.task)
        intent_task_cfg.horizon = 1                        # Ta = 1 in intent space
        intent_task_cfg.act_dim = effective_intent_dim     # flow model output dim
        intent_net = get_network(config.network, intent_task_cfg)

        self.intent_flow_map = FlowMap(intent_net).to(device)
        self.intent_flow_map_ema = deepcopy(self.intent_flow_map).requires_grad_(False)
        report_parameters(self.intent_flow_map, model_name="Flow Intent Model [Config A]")

        # ── Action model: MLP decoder OR ChiUNet flow map ────────────────
        # Gated by arch_variant:
        #   "flow_intent"        → deterministic MLPActionDecoder (original Config A)
        #   "flow_intent_chiunet"→ ChiUNet FlowMap conditioned on obs_emb ⊕ intent
        self._arch_variant = config.network.arch_variant
        self._use_chiunet_action = self._arch_variant == "flow_intent_chiunet"

        if self._use_chiunet_action:
            # ── ChiUNet action flow map ──────────────────────────────────
            # obs_dim for ChiUNet = emb_dim + intent_dim so that obs_emb and
            # intent are concatenated along the feature axis before ChiUNet's
            # global_cond_encoder linearises them.
            from mip.networks.chiunet import ChiUNet

            _action_obs_dim = config.network.emb_dim + effective_intent_dim
            action_net = ChiUNet(
                act_dim=config.task.act_dim,
                Ta=config.task.horizon,
                obs_dim=_action_obs_dim,
                To=config.task.obs_steps,
                model_dim=getattr(config.network, "model_dim", 128),
                emb_dim=config.network.emb_dim,
                kernel_size=getattr(config.network, "kernel_size", 5),
                cond_predict_scale=getattr(config.network, "cond_predict_scale", True),
                obs_as_global_cond=True,
                dim_mult=getattr(config.network, "dim_mult", None) or [1, 2, 2],
                timestep_emb_type=getattr(config.network, "timestep_emb_type", "positional"),
            )
            self.action_flow_map = FlowMap(action_net).to(device)
            self.action_flow_map_ema = deepcopy(self.action_flow_map).requires_grad_(False)
            report_parameters(self.action_flow_map, model_name="ChiUNet Action FlowMap [Config A-Chi]")
            # Identity encoder: passes pre-computed obs_emb⊕intent straight through
            # to flow_loss without any additional transformation.
            self._identity_encoder = IdentityEncoder().to(device)
            self._action_loss_fn = get_loss_fn("flow")

            self.action_decoder = None
            self.action_decoder_ema = None
            _action_optimizer_params = list(self.action_flow_map.parameters())
        else:
            # ── MLP action decoder (original) ────────────────────────────
            # Deterministic: (obs_emb, intent) → action sequence.
            # intent_dim here is effective_intent_dim (7 or intent_emb_dim).
            # hidden_dim mirrors emb_dim so capacity scales with the encoder.
            self.action_decoder = MLPActionDecoder(
                obs_steps=config.task.obs_steps,
                emb_dim=config.network.emb_dim,
                intent_dim=effective_intent_dim,
                horizon=config.task.horizon,
                act_dim=config.task.act_dim,
                hidden_dim=config.network.emb_dim,
                n_layers=config.network.num_layers,
                dropout=config.network.dropout,
            ).to(device)
            self.action_decoder_ema = deepcopy(self.action_decoder).requires_grad_(False)
            report_parameters(self.action_decoder, model_name="MLP Action Decoder [Config A]")

            self.action_flow_map = None
            self.action_flow_map_ema = None
            _action_optimizer_params = list(self.action_decoder.parameters())

        # ── Optimizers ────────────────────────────────────────────────────
        # intent_optimizer: encoder + intent_flow_map + (optional) intent_seq_encoder (step 1)
        # action_optimizer: action_decoder or action_flow_map                          (step 2)
        intent_params = (
            list(self.encoder.parameters())
            + list(self.intent_flow_map.parameters())
            + (list(self.intent_seq_encoder.parameters()) if self.intent_seq_encoder is not None else [])
        )
        self.intent_optimizer = torch.optim.AdamW(
            intent_params,
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )
        self.action_optimizer = torch.optim.AdamW(
            _action_optimizer_params,
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )

        # ── Loss / interpolant for flow intent model ──────────────────────
        # Always "flow" regardless of config.optimization.loss_type so that
        # the intent ODE at inference matches the training objective exactly.
        self.interpolant = Interpolant(config.optimization.interp_type)
        self._intent_loss_fn = get_loss_fn("flow")

        loguru.logger.info(
            "[FlowIntentAgent] Config-A agent ready. "
            f"arch_variant={self._arch_variant}, "
            f"intent_dim={config.task.intent_dim}, "
            f"emb_dim={config.network.emb_dim}, "
            f"horizon={config.task.horizon}, "
            f"act_dim={config.task.act_dim}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        act: torch.Tensor,         # (B, horizon, act_dim)  — GT action
        obs: torch.Tensor,         # (B, obs_steps, base_obs_dim) — NO intent appended
        delta_t: torch.Tensor,     # (B,)
        intent_gt: torch.Tensor,   # (B, intent_dim) — GT mean future eef
    ) -> dict:
        """One training step for both models.

        Step 1 — Flow intent loss:
            Treats intent as "the action to be generated" in a 1-step space.
            Uses the standard flow_loss which calls encoder(obs) internally,
            so obs must be the raw base observation (no intent appended).

        Step 2 — Action decoder MSE:
            Uses GT intent (independent mode) to train the decoder.
            obs_emb is computed with torch.no_grad() so encoder gradients
            do not flow through the action decoder loss.

        Args:
            act       : (B, horizon, act_dim)
            obs       : (B, obs_steps, base_obs_dim) — base obs, no intent
            delta_t   : (B,)
            intent_gt : (B, intent_dim)

        Returns:
            dict with keys: "loss", "intent_loss", "action_loss"
        """
        cfg = self.config.optimization

        # ── Step 1: flow intent loss ──────────────────────────────────────
        # Compute the intent target vector from the GT batch data.
        #
        # "encoded_mean": intent_gt is (B, N, 7) — apply intent_seq_encoder per step
        #    then mean-pool → (B, intent_emb_dim).  Gradients flow through the encoder.
        # All other types: intent_gt is already the pre-computed vector (B, intent_dim).
        #
        # Reshape to (B, 1, D) to match the (B, Ta, act_dim) convention.
        if self.intent_seq_encoder is not None:
            # intent_gt: (B, N, raw_intent_dim) → (B, N, intent_emb_dim) → (B, intent_emb_dim)
            assert intent_gt.dim() == 3, (
                f"encoded_mean: expected intent_gt (B, N, raw_dim), got {intent_gt.shape}"
            )
            intent_vec = self.intent_seq_encoder(intent_gt).mean(dim=1)  # (B, intent_emb_dim)
        else:
            assert intent_gt.dim() == 2, (
                f"mean/final/sequence: expected intent_gt (B, intent_dim), got {intent_gt.shape}"
            )
            intent_vec = intent_gt  # (B, intent_dim)
        intent_target = intent_vec.unsqueeze(1)  # (B, 1, D) ←→ act in flow_loss

        intent_loss, _ = self._intent_loss_fn(
            cfg,
            self.intent_flow_map,
            self.encoder,
            self.interpolant,
            intent_target,   # "act" in intent space
            obs,             # raw obs — encoder is called inside flow_loss
            delta_t,
        )

        self.intent_optimizer.zero_grad()
        intent_loss.backward()
        if cfg.grad_clip_norm:
            intent_params = (
                list(self.encoder.parameters())
                + list(self.intent_flow_map.parameters())
            )
            nn.utils.clip_grad_norm_(intent_params, cfg.grad_clip_norm)
        self.intent_optimizer.step()

        # ── Step 2: action model (MLP decoder MSE *or* ChiUNet flow loss) ─
        # obs_emb is always computed with no_grad so the encoder is not pulled
        # by the action loss — gradients flow to the encoder only via Step 1.
        # intent_vec is detached for the same reason.
        with torch.no_grad():
            obs_emb = self.encoder(obs, None)  # (B, obs_steps, emb_dim)

        if self._use_chiunet_action:
            # Build condition: obs_emb ⊕ intent → (B, To, emb_dim + intent_dim)
            # intent_vec is (B, intent_dim); expand to each obs step.
            intent_expanded = intent_vec.detach().unsqueeze(1).expand(
                -1, self.config.task.obs_steps, -1
            )  # (B, To, intent_dim)
            action_condition = torch.cat(
                [obs_emb.detach(), intent_expanded], dim=-1
            )  # (B, To, emb_dim + intent_dim)

            # flow_loss calls encoder(obs, None) internally; we pass IdentityEncoder
            # so action_condition is used unchanged as the conditioning signal.
            action_loss, _ = self._action_loss_fn(
                cfg,
                self.action_flow_map,
                self._identity_encoder,
                self.interpolant,
                act,               # (B, horizon, act_dim)
                action_condition,  # (B, To, emb_dim + intent_dim)
                delta_t,
            )
            _action_params = list(self.action_flow_map.parameters())
        else:
            # Original MLP decoder path — deterministic MSE
            action_pred = self.action_decoder(obs_emb, intent_vec.detach())  # (B, horizon, act_dim)
            assert action_pred.shape == act.shape, (
                f"action shape mismatch: pred {action_pred.shape} vs GT {act.shape}"
            )
            action_loss = nn.functional.mse_loss(action_pred, act)
            _action_params = list(self.action_decoder.parameters())

        self.action_optimizer.zero_grad()
        action_loss.backward()
        if cfg.grad_clip_norm:
            nn.utils.clip_grad_norm_(_action_params, cfg.grad_clip_norm)
        self.action_optimizer.step()

        # ── EMA update ────────────────────────────────────────────────────
        if cfg.ema_rate < 1:
            self._ema_update()

        return {
            "loss": (intent_loss + action_loss).item(),
            "intent_loss": intent_loss.item(),
            "action_loss": action_loss.item(),
        }

    def _ema_update(self):
        rate = self.config.optimization.ema_rate
        pairs = [
            (self.encoder, self.encoder_ema),
            (self.intent_flow_map, self.intent_flow_map_ema),
        ]
        if self._use_chiunet_action:
            pairs.append((self.action_flow_map, self.action_flow_map_ema))
        else:
            pairs.append((self.action_decoder, self.action_decoder_ema))
        if self.intent_seq_encoder is not None:
            pairs.append((self.intent_seq_encoder, self.intent_seq_encoder_ema))
        with torch.no_grad():
            for model, ema in pairs:
                for p, p_ema in zip(model.parameters(), ema.parameters()):
                    p_ema.data.mul_(rate).add_(p.data, alpha=1.0 - rate)

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    def sample(
        self,
        obs: torch.Tensor,           # (B, obs_steps, base_obs_dim) — NO intent
        use_ema: bool = True,
        num_steps: int = -1,
        num_intent_samples: int = 1,  # K: draw K intent candidates, use first
        return_intent: bool = False,  # if True, return (action, intent_vec)
    ) -> torch.Tensor:
        """Sample an action sequence given the current observation.

        Pipeline:
            1. obs → encoder → obs_emb       (B, obs_steps, emb_dim)
            2. noise → intent ODE             (B, 1, intent_dim) → (B, intent_dim)
            3. (obs_emb, intent) → action     (B, horizon, act_dim)

        Args:
            obs                : (B, obs_steps, base_obs_dim) — normalized base obs
            use_ema            : use EMA model copies (recommended at eval)
            num_steps          : ODE integration steps; -1 uses config default
            num_intent_samples : number of intent candidates (currently first is used)

        Returns:
            action : (B, horizon, act_dim)
        """
        encoder = self.encoder_ema if use_ema else self.encoder
        intent_flow = self.intent_flow_map_ema if use_ema else self.intent_flow_map
        # action_dec only used in MLP path; ChiUNet path resolves its own model inline
        action_dec = (None if self._use_chiunet_action
                      else (self.action_decoder_ema if use_ema else self.action_decoder))
        # intent_seq_encoder is not used at inference (we sample from the flow model)
        # so it is irrelevant here regardless of intent_type.

        # Build a local config copy with possibly overridden num_steps
        cfg = deepcopy(self.config.optimization)
        if num_steps >= 1:
            cfg.num_steps = int(num_steps)

        # obs can be a tensor (state) or dict/TensorDict (image); extract B and device accordingly
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            _first = next(iter(obs.values()))
            B = _first.shape[0]
            device = _first.device
        else:
            B = obs.shape[0]
            device = obs.device

        with torch.no_grad():
            # 1. Encode obs → context embedding
            obs_emb = encoder(obs, None)  # (B, obs_steps, emb_dim) or (B, emb_dim) for image
            if obs_emb.dim() == 2:
                obs_emb = obs_emb.unsqueeze(1)  # (B, emb_dim) → (B, 1, emb_dim)

            # 2. Sample intent via ODE in intent space (Ta=1, D=effective_intent_dim)
            #    For encoded_mean, D=intent_emb_dim; for raw types, D=intent_dim.
            #    We bypass the standard ode_sampler to avoid a redundant encoder call.
            _eff_dim = (
                getattr(self.config.task, "intent_emb_dim", self.config.task.intent_dim)
                if self._intent_type == "encoded_mean"
                else self.config.task.intent_dim
            )
            intent_noise = torch.randn(B, 1, _eff_dim, device=device)  # (B, 1, D)
            intent_sampled = self._run_intent_ode(
                cfg, intent_flow, obs_emb, intent_noise
            )  # (B, 1, intent_dim)
            intent_vec = intent_sampled.squeeze(1)  # (B, intent_dim)

            # 3. Decode (obs_emb, intent) → action
            if self._use_chiunet_action:
                # Build condition: obs_emb ⊕ intent → (B, To, emb_dim + intent_dim)
                intent_expanded = intent_vec.unsqueeze(1).expand(
                    -1, self.config.task.obs_steps, -1
                )
                action_condition = torch.cat([obs_emb, intent_expanded], dim=-1)
                action_flow = self.action_flow_map_ema if use_ema else self.action_flow_map
                action_noise = torch.randn(
                    B, self.config.task.horizon, self.config.task.act_dim, device=device
                )
                action = self._run_action_ode(cfg, action_flow, action_condition, action_noise)
            else:
                #    Shape: obs_emb (B, obs_steps, emb_dim), intent (B, intent_dim)
                action = action_dec(obs_emb, intent_vec)  # (B, horizon, act_dim)

        if return_intent:
            return action, intent_vec
        return action

    def sample_intent(self, obs: torch.Tensor, use_ema: bool = True, num_steps: int = -1) -> torch.Tensor:
        """Sample an intent vector from the intent ODE given obs. Returns (B, intent_dim)."""
        encoder = self.encoder_ema if use_ema else self.encoder
        intent_flow = self.intent_flow_map_ema if use_ema else self.intent_flow_map
        cfg = deepcopy(self.config.optimization)
        if num_steps >= 1:
            cfg.num_steps = int(num_steps)
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            _first = next(iter(obs.values()))
            B, device = _first.shape[0], _first.device
        else:
            B, device = obs.shape[0], obs.device
        _eff_dim = (
            getattr(self.config.task, "intent_emb_dim", self.config.task.intent_dim)
            if self._intent_type == "encoded_mean"
            else self.config.task.intent_dim
        )
        with torch.no_grad():
            obs_emb = encoder(obs, None)
            if obs_emb.dim() == 2:
                obs_emb = obs_emb.unsqueeze(1)
            intent_noise = torch.randn(B, 1, _eff_dim, device=device)
            intent_sampled = self._run_intent_ode(cfg, intent_flow, obs_emb, intent_noise)
        return intent_sampled.squeeze(1)  # (B, intent_dim)

    def sample_given_intent(
        self,
        obs: torch.Tensor,
        intent_vec: torch.Tensor,  # (B, intent_dim)
        use_ema: bool = True,
        num_steps: int = -1,
    ) -> torch.Tensor:
        """Decode action given a pre-sampled intent vector. Returns (B, horizon, act_dim)."""
        encoder = self.encoder_ema if use_ema else self.encoder
        action_dec = (None if self._use_chiunet_action
                      else (self.action_decoder_ema if use_ema else self.action_decoder))
        cfg = deepcopy(self.config.optimization)
        if num_steps >= 1:
            cfg.num_steps = int(num_steps)
        with torch.no_grad():
            obs_emb = encoder(obs, None)
            if obs_emb.dim() == 2:
                obs_emb = obs_emb.unsqueeze(1)
            if self._use_chiunet_action:
                intent_expanded = intent_vec.unsqueeze(1).expand(-1, self.config.task.obs_steps, -1)
                action_condition = torch.cat([obs_emb, intent_expanded], dim=-1)
                action_flow = self.action_flow_map_ema if use_ema else self.action_flow_map
                action_noise = torch.randn(
                    intent_vec.shape[0], self.config.task.horizon, self.config.task.act_dim,
                    device=intent_vec.device,
                )
                action = self._run_action_ode(cfg, action_flow, action_condition, action_noise)
            else:
                action = action_dec(obs_emb, intent_vec)
        return action

    def _run_intent_ode(
        self,
        cfg,
        intent_flow_map: FlowMap,
        obs_emb: torch.Tensor,   # (B, obs_steps, emb_dim) — pre-computed label
        intent_0: torch.Tensor,  # (B, 1, intent_dim) — initial noise
    ) -> torch.Tensor:
        """Euler ODE integration for the intent flow model.

        Replicates the logic of ode_sampler but uses a pre-computed obs_emb
        as the label, avoiding a second encoder forward pass.

        FlowMap.get_velocity(s, xs, label) internally calls:
            net(xs, s, s, label)
        where xs (B, 1, intent_dim) and label (B, obs_steps, emb_dim).

        Returns:
            intent : (B, 1, intent_dim) — sampled intent
        """
        num_steps = cfg.num_steps
        t_schedule = np.linspace(0, 1, num_steps + 1)
        act_s = intent_0  # (B, 1, intent_dim)
        bs = act_s.shape[0]

        for i in range(num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((bs,), s_val, device=act_s.device)
            t = torch.full((bs,), t_val, device=act_s.device)
            # get_velocity(t, xs, label): label = obs_emb (B, obs_steps, emb_dim)
            b_s = intent_flow_map.get_velocity(s, act_s, obs_emb)
            s_expanded = at_least_ndim(s, act_s.dim())
            t_expanded = at_least_ndim(t, act_s.dim())
            act_s = act_s + b_s * (t_expanded - s_expanded)

        return act_s  # (B, 1, intent_dim)

    def _run_action_ode(
        self,
        cfg,
        action_flow_map: FlowMap,
        action_condition: torch.Tensor,  # (B, To, emb_dim + intent_dim) — pre-computed
        action_0: torch.Tensor,          # (B, horizon, act_dim) — initial noise
    ) -> torch.Tensor:
        """Euler ODE integration for the ChiUNet action flow model.

        action_condition is passed directly as the label to get_velocity —
        no additional encoding step since it was already built from obs_emb⊕intent.

        Returns:
            action : (B, horizon, act_dim)
        """
        num_steps = cfg.num_steps
        t_schedule = np.linspace(0, 1, num_steps + 1)
        act_s = action_0
        bs = act_s.shape[0]

        for i in range(num_steps):
            s_val = t_schedule[i]
            t_val = t_schedule[i + 1]
            s = torch.full((bs,), s_val, device=act_s.device)
            t = torch.full((bs,), t_val, device=act_s.device)
            b_s = action_flow_map.get_velocity(s, act_s, action_condition)
            s_expanded = at_least_ndim(s, act_s.dim())
            t_expanded = at_least_ndim(t, act_s.dim())
            act_s = act_s + b_s * (t_expanded - s_expanded)

        return act_s  # (B, horizon, act_dim)

    # ──────────────────────────────────────────────────────────────────────
    # Mode switching
    # ──────────────────────────────────────────────────────────────────────

    def eval(self):
        """Switch all sub-models to eval mode."""
        self.encoder.eval()
        self.encoder_ema.eval()
        self.intent_flow_map.eval()
        self.intent_flow_map_ema.eval()
        if self._use_chiunet_action:
            self.action_flow_map.eval()
            self.action_flow_map_ema.eval()
        else:
            self.action_decoder.eval()
            self.action_decoder_ema.eval()
        if self.intent_seq_encoder is not None:
            self.intent_seq_encoder.eval()
            self.intent_seq_encoder_ema.eval()

    def train(self):
        """Switch all sub-models to train mode."""
        self.encoder.train()
        self.encoder_ema.train()
        self.intent_flow_map.train()
        self.intent_flow_map_ema.train()
        if self._use_chiunet_action:
            self.action_flow_map.train()
            self.action_flow_map_ema.train()
        else:
            self.action_decoder.train()
            self.action_decoder_ema.train()
        if self.intent_seq_encoder is not None:
            self.intent_seq_encoder.train()
            self.intent_seq_encoder_ema.train()

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str, training_state: dict = None):
        """Save all model weights and optimizer states."""
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "encoder_ema": self.encoder_ema.state_dict(),
            "intent_flow_map": self.intent_flow_map.state_dict(),
            "intent_flow_map_ema": self.intent_flow_map_ema.state_dict(),
            "intent_optimizer": self.intent_optimizer.state_dict(),
            "action_optimizer": self.action_optimizer.state_dict(),
        }
        if self._use_chiunet_action:
            checkpoint["action_flow_map"] = self.action_flow_map.state_dict()
            checkpoint["action_flow_map_ema"] = self.action_flow_map_ema.state_dict()
        else:
            checkpoint["action_decoder"] = self.action_decoder.state_dict()
            checkpoint["action_decoder_ema"] = self.action_decoder_ema.state_dict()
        if self.intent_seq_encoder is not None:
            checkpoint["intent_seq_encoder"] = self.intent_seq_encoder.state_dict()
            checkpoint["intent_seq_encoder_ema"] = self.intent_seq_encoder_ema.state_dict()
        if training_state is not None:
            checkpoint["training_state"] = training_state
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False) -> dict | None:
        """Load model weights from a checkpoint.

        Args:
            path           : path to the .pt checkpoint
            load_optimizer : also restore optimizer states

        Returns:
            training_state dict if available, else None
        """
        sd = torch.load(
            path,
            map_location=self.config.optimization.device,
            weights_only=False,
        )
        self.encoder.load_state_dict(sd["encoder"])
        self.encoder_ema.load_state_dict(sd["encoder_ema"])
        self.intent_flow_map.load_state_dict(sd["intent_flow_map"])
        self.intent_flow_map_ema.load_state_dict(sd["intent_flow_map_ema"])
        if self._use_chiunet_action:
            self.action_flow_map.load_state_dict(sd["action_flow_map"])
            self.action_flow_map_ema.load_state_dict(sd["action_flow_map_ema"])
        else:
            self.action_decoder.load_state_dict(sd["action_decoder"])
            self.action_decoder_ema.load_state_dict(sd["action_decoder_ema"])
        if self.intent_seq_encoder is not None and "intent_seq_encoder" in sd:
            self.intent_seq_encoder.load_state_dict(sd["intent_seq_encoder"])
            self.intent_seq_encoder_ema.load_state_dict(sd["intent_seq_encoder_ema"])

        if load_optimizer:
            try:
                self.intent_optimizer.load_state_dict(sd["intent_optimizer"])
                self.action_optimizer.load_state_dict(sd["action_optimizer"])
                loguru.logger.info("[FlowIntentAgent] Loaded optimizer states")
            except Exception as e:
                loguru.logger.warning(
                    f"[FlowIntentAgent] Could not load optimizer states ({e}); "
                    "continuing with fresh optimizers."
                )

        training_state = sd.get("training_state", None)
        if training_state:
            loguru.logger.info(
                f"[FlowIntentAgent] Loaded checkpoint from step "
                f"{training_state.get('n_gradient_step', 'unknown')}"
            )
        return training_state
