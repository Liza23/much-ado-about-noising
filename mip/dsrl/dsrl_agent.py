"""DSRL-SAC agent: learns initial noise x_0 in Gaussian space to steer a frozen flow policy.

Reference: "Steering Your Diffusion Policy with Latent Space Reinforcement Learning" (arXiv 2506.15799)

For flow matching: x_0 ~ N(0,1) is the ODE starting point. The actor outputs x_0 given obs,
replacing random noise with a learned noise that steers the flow toward high-reward actions.

Composition:  x_0 = actor(obs)
              a_exec = flow_policy.sample(act_0=x_0, obs=obs, sample_mode="stochastic")

The frozen flow policy is accessed externally (set via set_flow_agent).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mip.residual_sac.networks import TanhGaussianPolicy, QEnsemble


@dataclass
class DSRLConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    # Wide MLP recommended for high-dim noise space (Ta*act_dim can be 160+)
    hidden_dims: tuple[int, ...] = field(default_factory=lambda: (1024, 1024, 1024))
    discount: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.1
    target_entropy: float | None = None  # None = auto: -noise_dim * 0.5
    # Noise magnitude: x_0 ~ N(0,1) so values typically in [-3, 3].
    # TanhGaussian squashes to [-noise_mag, noise_mag].
    noise_mag: float = 2.0
    backup_entropy: bool = True
    max_grad_norm: float = 10.0
    weight_decay: float = 1e-4
    use_layer_norm: bool = False
    # Clamp Q targets to prevent bootstrap explosion.
    # Set to reward_scale * max_episode_reward / (1 - discount).
    # For reward_scale=0.1 and max_ep_reward=50: 0.1*50/0.01=500 (loose).
    # In practice set ~5-10x expected max return for the task.
    q_target_clip: float = float("inf")
    device: str = "cuda"


class DSRLSACAgent:
    """DSRL-SAC: SAC in the noise space of a frozen flow policy.

    Args:
        config:    DSRLConfig hyperparameters.
        obs_dim:   Flattened obs dimension (obs_steps * per_step_obs_dim).
        Ta:        Action horizon of the flow policy.
        act_dim:   Per-step action dimension.
    """

    def __init__(
        self,
        config: DSRLConfig,
        obs_dim: int,
        Ta: int,
        act_dim: int,
    ):
        self.config = config
        self.obs_dim = obs_dim
        self.Ta = Ta
        self.act_dim = act_dim
        self.noise_dim = Ta * act_dim
        device = config.device

        self.actor = TanhGaussianPolicy(
            obs_dim=obs_dim,
            act_dim_flat=self.noise_dim,
            hidden_dims=config.hidden_dims,
            action_mag=config.noise_mag,
            use_layer_norm=config.use_layer_norm,
        ).to(device)

        self.critic = QEnsemble(
            obs_dim=obs_dim,
            act_dim_flat=self.noise_dim,
            hidden_dims=config.hidden_dims,
            use_layer_norm=config.use_layer_norm,
        ).to(device)

        self.critic_target = deepcopy(self.critic).requires_grad_(False)

        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(config.init_alpha), dtype=torch.float32, device=device)
        )
        self.target_entropy = (
            config.target_entropy
            if config.target_entropy is not None
            else -self.noise_dim * 0.5
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr, weight_decay=config.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        self.flow_agent = None

    def set_flow_agent(self, flow_agent) -> None:
        self.flow_agent = flow_agent

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def sample_noise(
        self,
        obs_flat: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Sample initial noise x_0 given flattened obs.

        Returns:
            noise: (noise_dim,) numpy — x_0 for the flow ODE.
            log_prob: scalar.
        """
        device = self.config.device
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
        if deterministic:
            noise = self.actor.deterministic(obs_t)
            log_prob = 0.0
        else:
            noise, lp = self.actor.sample(obs_t)
            log_prob = lp.item()
        return noise.squeeze(0).cpu().numpy(), log_prob

    def update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        noise = batch["noise"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_noise, next_log_prob = self.actor.sample(next_obs)
            next_q = self.critic_target.min_q(next_obs, next_noise).squeeze(-1)
            if self.config.backup_entropy:
                next_q = next_q - self.alpha.detach() * next_log_prob
            target_q = reward + (1.0 - done) * self.config.discount * next_q
            if self.config.q_target_clip < float("inf"):
                target_q = target_q.clamp(-self.config.q_target_clip, self.config.q_target_clip)

        qs = self.critic(obs, noise)  # (num_qs, B, 1)
        critic_loss = sum(
            F.mse_loss(qs[i].squeeze(-1), target_q) for i in range(self.critic.num_qs)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        for p in self.critic.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "q_mean": qs.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

    def update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        noise, log_prob = self.actor.sample(obs)
        q = self.critic.min_q(obs, noise).squeeze(-1)
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for p in self.actor.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "entropy": -log_prob.mean().item(),
            "noise_norm": noise.detach().norm(dim=-1).mean().item(),
        }

    def soft_update_targets(self) -> None:
        tau = self.config.tau
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        info = {}
        info.update(self.update_critic(batch))
        info.update(self.update_actor(batch))
        self.soft_update_targets()
        return info

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
