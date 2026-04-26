"""SAC policy and value networks for residual action learning.

TanhGaussianPolicy: Outputs a TanhNormal distribution for continuous actions.
QEnsemble: Twin (or N) Q-networks for value estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    activation: type[nn.Module] = nn.ReLU,
    output_activation: type[nn.Module] | None = None,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class TanhGaussianPolicy(nn.Module):
    """Squashed Gaussian policy: obs -> TanhNormal(mean, std).

    The output distribution maps to [-action_mag, action_mag] via
    Tanh followed by an affine rescaling.

    Args:
        obs_dim:      Flattened observation dimension.
        act_dim_flat: Total output action dimension (e.g. query_freq * act_dim).
        hidden_dims:  MLP hidden layer sizes.
        log_std_min:  Clamp lower bound for log_std.
        log_std_max:  Clamp upper bound for log_std.
        action_mag:   Magnitude for action rescaling (default 1.0).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim_flat: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        action_mag: float = 1.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.act_dim_flat = act_dim_flat
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_mag = action_mag

        self.trunk = _build_mlp(obs_dim, hidden_dims[-1], hidden_dims[:-1], use_layer_norm=use_layer_norm)
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim_flat)
        self.log_std_head = nn.Linear(hidden_dims[-1], act_dim_flat)

    def forward(self, obs: torch.Tensor) -> TransformedDistribution:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        base_dist = Normal(mean, std)
        # TanhTransform squashes to (-1, 1); AffineTransform rescales to (-mag, mag).
        transforms = [TanhTransform(cache_size=1)]
        if self.action_mag != 1.0:
            transforms.append(AffineTransform(loc=0.0, scale=self.action_mag))
        return TransformedDistribution(base_dist, transforms)

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample with log-probability.

        Returns:
            action:   (B, act_dim_flat) in [-action_mag, action_mag]
            log_prob: (B,) summed over action dimensions
        """
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Mode of the distribution (tanh of the mean)."""
        h = self.trunk(obs)
        mean = self.mean_head(h)
        action = torch.tanh(mean)
        if self.action_mag != 1.0:
            action = action * self.action_mag
        return action


class QNetwork(nn.Module):
    """Single Q-function: Q(obs, action) -> scalar."""

    def __init__(
        self,
        obs_dim: int,
        act_dim_flat: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.net = _build_mlp(obs_dim + act_dim_flat, 1, hidden_dims, use_layer_norm=use_layer_norm)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns Q-value of shape (B, 1)."""
        return self.net(torch.cat([obs, action], dim=-1))


class QEnsemble(nn.Module):
    """Ensemble of Q-functions for pessimistic value estimation.

    Args:
        obs_dim:      Flattened observation dimension.
        act_dim_flat: Total action dimension.
        hidden_dims:  MLP hidden layer sizes per Q-network.
        num_qs:       Number of Q-networks in the ensemble.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim_flat: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        num_qs: int = 2,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.q_nets = nn.ModuleList(
            [QNetwork(obs_dim, act_dim_flat, hidden_dims, use_layer_norm=use_layer_norm) for _ in range(num_qs)]
        )
        self.num_qs = num_qs

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Returns Q-values of shape (num_qs, B, 1)."""
        return torch.stack([q(obs, action) for q in self.q_nets], dim=0)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Pessimistic Q: min over ensemble, shape (B, 1)."""
        qs = self.forward(obs, action)
        return qs.min(dim=0).values
