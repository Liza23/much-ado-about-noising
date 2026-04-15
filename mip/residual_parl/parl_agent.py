"""Residual PARL agent: learns delta_action on top of a frozen flow-intent policy.

Composition: a_exec = clip(flow_action + residual_alpha * delta, -1, 1)

Actor update (PARL — never differentiates through the critic):
  1. Sample N action candidates from actor + include base action as (N+1)-th
  2. Evaluate all with Q, keep top-K elites
  3. Refine elites via gradient ascent on Q w.r.t. action
  4. Pick the best refined action as distillation target a*
  5. Train actor via MSE loss to imitate a*

Critic update: standard TD learning (same as SAC, without entropy backup).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mip.residual_sac.networks import TanhGaussianPolicy, QEnsemble


@dataclass
class PARLConfig:
    """Hyperparameters for residual PARL."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    discount: float = 0.99
    tau: float = 0.005
    residual_alpha: float = 0.1
    action_magnitude: float = 0.1   # bounds actor output to [-mag, mag]
    predict_a_exec: bool = False
    max_grad_norm: float = 10.0
    device: str = "cuda"

    # PARL-specific
    parl_num_samples: int = 16    # N — candidates sampled from actor
    parl_num_elites: int = 4      # K — top-Q candidates kept
    parl_num_grad_steps: int = 5  # gradient ascent steps on Q w.r.t. action
    parl_step_size: float = 0.01  # step size for gradient ascent on actions

    # Update ratios
    num_critic_updates: int = 2
    num_actor_updates: int = 4

    # Critic
    critic_reduction: str = "min"  # "min" or "mean"


class ResidualPARLAgent:
    """Residual PARL that learns delta_action on top of a frozen flow-intent model.

    The actor is trained by distilling Q-optimized actions (Best-of-N + Grad-Q)
    rather than by differentiating through the critic.

    Args:
        config:       PARLConfig with hyperparameters.
        obs_dim:      Flattened observation dimension (obs_steps * per_step_dim).
        act_dim:      Per-step action dimension.
        query_freq:   Number of action steps to execute per chunk.
    """

    def __init__(
        self,
        config: PARLConfig,
        obs_dim: int,
        act_dim: int,
        query_freq: int,
        *,
        delta_dim: int | None = None,
    ):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.query_freq = query_freq
        self.act_dim_flat = query_freq * act_dim
        device = config.device

        # --- Networks (reuse SAC architecture) ---
        self.actor = TanhGaussianPolicy(
            obs_dim=obs_dim,
            act_dim_flat=self.act_dim_flat,
            hidden_dims=config.hidden_dims,
            action_mag=config.action_magnitude,
        ).to(device)

        self.critic = QEnsemble(
            obs_dim=obs_dim,
            act_dim_flat=self.act_dim_flat,
            hidden_dims=config.hidden_dims,
        ).to(device)

        self.critic_target = deepcopy(self.critic).requires_grad_(False)

        # --- Optimizers ---
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        # --- Frozen flow-intent (set externally) ---
        self.flow_intent = None

    # ------------------------------------------------------------------
    # Flow-intent interface
    # ------------------------------------------------------------------
    def set_flow_intent(self, flow_intent_agent) -> None:
        """Attach a frozen FlowIntentAgent as the base policy."""
        self.flow_intent = flow_intent_agent

    @torch.no_grad()
    def get_base_action(self, obs_tensor: torch.Tensor) -> np.ndarray:
        """Query frozen flow-intent for a base action chunk.

        Args:
            obs_tensor: (1, obs_steps, obs_dim_per_step) observation.

        Returns:
            base_actions: (query_freq, act_dim) numpy array.
        """
        action = self.flow_intent.sample(obs_tensor, use_ema=True)
        return action[0, : self.query_freq].cpu().numpy()

    # ------------------------------------------------------------------
    # Action composition
    # ------------------------------------------------------------------
    def _compose(
        self,
        base_action_flat: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Compose executed action: clip(base + alpha * delta, -1, 1)."""
        if self.config.predict_a_exec:
            return delta.clamp(-1.0, 1.0)
        return (base_action_flat + self.config.residual_alpha * delta).clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_action(
        self,
        obs_flat: np.ndarray,
        base_action_flat: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample delta from actor, compose with base_action.

        At eval (deterministic=True), uses Q-switching: returns the residual
        action only if the critic says it's better than the base action alone.

        Returns:
            a_exec: (act_dim_flat,) numpy.
            delta:  (act_dim_flat,) numpy.
            log_prob: scalar (0.0 if deterministic).
        """
        device = self.config.device
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
        base_t = torch.as_tensor(base_action_flat, dtype=torch.float32, device=device).unsqueeze(0)

        if deterministic:
            delta = self.actor.deterministic(obs_t)
            a_exec_residual = self._compose(base_t, delta)
            base_clipped = base_t.clamp(-1.0, 1.0)

            # Q-switching: use residual only if critic prefers it over base
            q_residual = self.critic.min_q(obs_t, a_exec_residual).squeeze(-1)
            q_base = self.critic.min_q(obs_t, base_clipped).squeeze(-1)
            if q_residual.item() >= q_base.item():
                a_exec = a_exec_residual
            else:
                a_exec = base_clipped
                delta = torch.zeros_like(delta)

            return (
                a_exec.squeeze(0).cpu().numpy(),
                delta.squeeze(0).cpu().numpy(),
                0.0,
            )
        else:
            delta, lp = self.actor.sample(obs_t)
            log_prob = lp.item()
            a_exec = self._compose(base_t, delta)
            return (
                a_exec.squeeze(0).cpu().numpy(),
                delta.squeeze(0).cpu().numpy(),
                log_prob,
            )

    # ------------------------------------------------------------------
    # Critic update (standard TD, no entropy backup)
    # ------------------------------------------------------------------
    def update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single critic gradient step (TD learning without entropy backup)."""
        obs = batch["obs"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        B = obs.shape[0]

        # Reconstruct a_exec from stored delta + base
        base_flat = batch["base_action"].reshape(B, -1)
        delta = batch["delta"]
        a_exec_flat = self._compose(base_flat, delta)

        # --- Target Q (no entropy term) ---
        with torch.no_grad():
            next_base_flat = batch["next_base_action"].reshape(B, -1)
            next_delta, _ = self.actor.sample(next_obs)
            next_a_exec = self._compose(next_base_flat, next_delta)

            next_qs = self.critic_target(next_obs, next_a_exec)  # (num_qs, B, 1)
            if self.config.critic_reduction == "min":
                next_q = next_qs.min(dim=0).values.squeeze(-1)
            else:
                next_q = next_qs.mean(dim=0).squeeze(-1)
            target_q = reward + (1.0 - done) * self.config.discount * next_q

        # --- Critic loss ---
        qs = self.critic(obs, a_exec_flat)  # (num_qs, B, 1)
        critic_loss = sum(
            F.mse_loss(qs[i].squeeze(-1), target_q) for i in range(self.critic.num_qs)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "q_mean": qs.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

    # ------------------------------------------------------------------
    # PARL actor update (Best-of-N + Grad-Q + BC distillation)
    # ------------------------------------------------------------------
    def update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """PARL actor update: never differentiates through the critic.

        1. Sample N deltas from actor + base action as (N+1)-th candidate
        2. Evaluate all with Q, keep top-K
        3. Refine elites via gradient ascent on Q w.r.t. action
        4. Pick best refined action → MSE distillation into actor
        """
        obs = batch["obs"]
        base_flat = batch["base_action"].reshape(obs.shape[0], -1)
        B = obs.shape[0]
        N = self.config.parl_num_samples
        K = self.config.parl_num_elites
        device = self.config.device
        alpha = self.config.residual_alpha

        # ---- Step 1: Sample N candidate deltas from actor (no grad) ----
        with torch.no_grad():
            deltas = []
            for _ in range(N):
                d, _ = self.actor.sample(obs)
                deltas.append(d)
            deltas = torch.stack(deltas, dim=0)  # (N, B, act_dim_flat)

            # Convert deltas to a_exec
            # (N, B, act_dim_flat)
            if self.config.predict_a_exec:
                a_exec_candidates = deltas.clamp(-1.0, 1.0)
            else:
                a_exec_candidates = (base_flat.unsqueeze(0) + alpha * deltas).clamp(-1.0, 1.0)

            # Add base action as (N+1)-th candidate
            base_a_exec = base_flat.clamp(-1.0, 1.0).unsqueeze(0)  # (1, B, act_dim_flat)
            a_exec_all = torch.cat([a_exec_candidates, base_a_exec], dim=0)  # (N+1, B, act_dim_flat)

            # ---- Step 2: Evaluate Q for all candidates, keep top-K ----
            # Compute Q for each candidate
            q_values = []
            for i in range(N + 1):
                qs = self.critic(obs, a_exec_all[i])  # (num_qs, B, 1)
                if self.config.critic_reduction == "min":
                    q = qs.min(dim=0).values.squeeze(-1)  # (B,)
                else:
                    q = qs.mean(dim=0).squeeze(-1)
                q_values.append(q)
            q_values = torch.stack(q_values, dim=0)  # (N+1, B)

            # Top-K per batch element
            q_T = q_values.T  # (B, N+1)
            top_k_indices = q_T.topk(K, dim=-1).indices  # (B, K)

            # Gather elite a_exec: (B, K, act_dim_flat)
            a_exec_all_T = a_exec_all.permute(1, 0, 2)  # (B, N+1, act_dim_flat)
            elite_actions = torch.gather(
                a_exec_all_T,
                dim=1,
                index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.act_dim_flat),
            )  # (B, K, act_dim_flat)

            # Track base-in-elites fraction
            base_in_elites = (top_k_indices == N).any(dim=-1).float().mean().item()

            q_before = torch.gather(q_T, dim=1, index=top_k_indices).mean().item()

        # ---- Step 3: Gradient ascent on Q w.r.t. a_exec for elites ----
        # This requires grad through the critic w.r.t. action (not params)
        refined = elite_actions.detach().clone()  # (B, K, act_dim_flat)

        for _ in range(self.config.parl_num_grad_steps):
            refined.requires_grad_(True)
            # Evaluate Q for all K elites
            q_sum = torch.tensor(0.0, device=device)
            for k in range(K):
                qs = self.critic(obs, refined[:, k, :])  # (num_qs, B, 1)
                if self.config.critic_reduction == "min":
                    q_sum = q_sum + qs.min(dim=0).values.sum()
                else:
                    q_sum = q_sum + qs.mean(dim=0).sum()

            grad = torch.autograd.grad(q_sum, refined)[0]
            refined = (refined.detach() + self.config.parl_step_size * grad).clamp(-1.0, 1.0)

        # ---- Step 4: Pick the best refined action ----
        with torch.no_grad():
            best_q = torch.full((B,), -float("inf"), device=device)
            best_a_exec = torch.zeros(B, self.act_dim_flat, device=device)
            for k in range(K):
                qs = self.critic(obs, refined[:, k, :])
                if self.config.critic_reduction == "min":
                    q = qs.min(dim=0).values.squeeze(-1)
                else:
                    q = qs.mean(dim=0).squeeze(-1)
                mask = q > best_q
                best_q = torch.where(mask, q, best_q)
                best_a_exec = torch.where(mask.unsqueeze(-1), refined[:, k, :], best_a_exec)

            # Convert best a_exec back to delta space for distillation target
            mag = self.config.action_magnitude
            if self.config.predict_a_exec:
                bc_target = best_a_exec.clamp(-mag + 1e-6, mag - 1e-6)
            else:
                bc_target_delta = (best_a_exec - base_flat) / (alpha + 1e-8)
                bc_target = bc_target_delta.clamp(-mag + 1e-6, mag - 1e-6)

        # ---- Step 5: Distill into actor via MSE loss ----
        sampled_delta, _ = self.actor.sample(obs)
        actor_loss = F.mse_loss(sampled_delta, bc_target)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "parl/q_before_refinement": q_before,
            "parl/q_after_refinement": best_q.mean().item(),
            "parl/base_in_elites_frac": base_in_elites,
            "delta_norm": sampled_delta.detach().norm(dim=-1).mean().item(),
        }

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------
    def soft_update_targets(self) -> None:
        """Polyak averaging for target critic."""
        tau = self.config.tau
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Full PARL update: multiple critic updates + multiple actor updates + target."""
        info = {}

        # Multiple critic updates
        for _ in range(self.config.num_critic_updates):
            critic_info = self.update_critic(batch)
        info.update(critic_info)  # log last critic update

        # Multiple actor updates
        for _ in range(self.config.num_actor_updates):
            actor_info = self.update_actor(batch)
        info.update(actor_info)  # log last actor update

        self.soft_update_targets()
        return info

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, load_optimizers: bool = False) -> None:
        sd = torch.load(path, map_location=self.config.device, weights_only=False)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(sd["actor_optimizer"])
            self.critic_optimizer.load_state_dict(sd["critic_optimizer"])

    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()
