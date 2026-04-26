"""Replay buffer for DSRL (noise-space transitions)."""

from __future__ import annotations

import numpy as np
import torch


class DSRLReplayBuffer:
    """Stores (obs, noise_x0, reward, next_obs, done) transitions.

    noise_x0 is the initial Gaussian x_0 passed to the flow ODE — the
    action space for the DSRL SAC actor.

    Args:
        capacity:  Maximum number of transitions.
        obs_dim:   Flattened observation dimension (obs_steps * per_step_dim).
        noise_dim: Noise dimension = Ta * act_dim.
        device:    Torch device for sampled batches.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        noise_dim: int,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.device = device
        self._size = 0
        self._ptr = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.noise = np.zeros((capacity, noise_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

    @property
    def size(self) -> int:
        return self._size

    def insert(
        self,
        obs: np.ndarray,
        noise: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self._ptr] = obs
        self.noise[self._ptr] = noise
        self.reward[self._ptr] = reward
        self.next_obs[self._ptr] = next_obs
        self.done[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idxs], device=self.device),
            "noise": torch.as_tensor(self.noise[idxs], device=self.device),
            "reward": torch.as_tensor(self.reward[idxs], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], device=self.device),
            "done": torch.as_tensor(self.done[idxs], device=self.device),
        }
