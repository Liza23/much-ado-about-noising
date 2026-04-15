"""Residual SAC agent: learns delta_action on top of a frozen flow-intent policy.

Composition: a_exec = clip(flow_action + residual_alpha * delta, -1, 1)
Critic learns Q(s, a_exec).
Actor maximizes Q - alpha * log_prob(delta | s).
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
class SACConfig:
    """Hyperparameters for residual SAC."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    discount: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.1
    target_entropy: float | None = None  # None = auto: -act_dim_flat * 0.1
    residual_alpha: float = 0.1
    predict_a_exec: bool = False
    backup_entropy: bool = True
    max_grad_norm: float = 10.0
    device: str = "cuda"


class ResidualSACAgent:
    """Residual SAC that learns delta_action on top of a frozen flow-intent model.

    Args:
        config:       SACConfig with hyperparameters.
        obs_dim:      Flattened observation dimension (obs_steps * per_step_dim).
        act_dim:      Per-step action dimension.
        query_freq:   Number of action steps to execute per chunk (controls the
                      dimensionality of the SAC actor output).
    """

    def __init__(
        self,
        config: SACConfig,
        obs_dim: int,
        act_dim: int,
        query_freq: int,
    ):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.query_freq = query_freq
        self.act_dim_flat = query_freq * act_dim
        device = config.device

        # --- Networks ---
        self.actor = TanhGaussianPolicy(
            obs_dim=obs_dim,
            act_dim_flat=self.act_dim_flat,
            hidden_dims=config.hidden_dims,
        ).to(device)

        self.critic = QEnsemble(
            obs_dim=obs_dim,
            act_dim_flat=self.act_dim_flat,
            hidden_dims=config.hidden_dims,
        ).to(device)

        self.critic_target = deepcopy(self.critic).requires_grad_(False)

        # --- Entropy temperature (log_alpha) ---
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(config.init_alpha), dtype=torch.float32, device=device)
        )
        self.target_entropy = (
            config.target_entropy
            if config.target_entropy is not None
            else -self.act_dim_flat * 0.1
        )

        # --- Optimizers ---
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr
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
            obs_tensor: (1, obs_steps, obs_dim_per_step) observation for the
                        flow-intent model (already normalized).

        Returns:
            base_actions: (query_freq, act_dim) numpy array.
        """
        action = self.flow_intent.sample(obs_tensor, use_ema=True)
        # action shape: (1, horizon, act_dim) — slice to query_freq
        return action[0, : self.query_freq].cpu().numpy()

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------
    def _compose(
        self,
        base_action_flat: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Compose executed action: clip(base + alpha * delta, -1, 1).

        Both inputs are (B, act_dim_flat).
        """
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

        Args:
            obs_flat:         (obs_dim,) numpy — flattened normalized observation.
            base_action_flat: (act_dim_flat,) numpy — flattened base action chunk.
            deterministic:    If True, use mode instead of sampling.

        Returns:
            a_exec: (act_dim_flat,) numpy — composed action.
            delta:  (act_dim_flat,) numpy — raw actor output.
            log_prob: scalar log-probability (0.0 if deterministic).
        """
        device = self.config.device
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
        base_t = torch.as_tensor(base_action_flat, dtype=torch.float32, device=device).unsqueeze(0)

        if deterministic:
            delta = self.actor.deterministic(obs_t)
            log_prob = 0.0
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
    # Update steps
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single critic gradient step (TD learning).

        batch keys: obs, base_action, delta, reward, next_obs, next_base_action, done.
        """
        obs = batch["obs"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # Reconstruct a_exec from stored delta + base
        base_flat = batch["base_action"].reshape(obs.shape[0], -1)
        delta = batch["delta"]
        a_exec_flat = self._compose(base_flat, delta)

        # --- Target Q ---
        with torch.no_grad():
            next_base_flat = batch["next_base_action"].reshape(obs.shape[0], -1)
            next_delta, next_log_prob = self.actor.sample(next_obs)
            next_a_exec = self._compose(next_base_flat, next_delta)

            next_q = self.critic_target.min_q(next_obs, next_a_exec).squeeze(-1)
            if self.config.backup_entropy:
                next_q = next_q - self.alpha.detach() * next_log_prob
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

    def update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single actor gradient step (entropy-regularized Q maximization)."""
        obs = batch["obs"]
        base_flat = batch["base_action"].reshape(obs.shape[0], -1)

        delta, log_prob = self.actor.sample(obs)
        a_exec = self._compose(base_flat, delta)

        q = self.critic.min_q(obs, a_exec).squeeze(-1)

        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # --- Alpha (temperature) update ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "entropy": -log_prob.mean().item(),
            "delta_norm": delta.detach().norm(dim=-1).mean().item(),
        }

    def soft_update_targets(self) -> None:
        """Polyak averaging for target critic."""
        tau = self.config.tau
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Full SAC update: critic + actor + alpha + target."""
        info = {}
        info.update(self.update_critic(batch))
        info.update(self.update_actor(batch))
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
                "log_alpha": self.log_alpha.data,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, load_optimizers: bool = False) -> None:
        sd = torch.load(path, map_location=self.config.device, weights_only=False)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.log_alpha.data.copy_(sd["log_alpha"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(sd["actor_optimizer"])
            self.critic_optimizer.load_state_dict(sd["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(sd["alpha_optimizer"])

    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()
