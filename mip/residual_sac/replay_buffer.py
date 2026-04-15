"""Simple numpy-backed replay buffer for online RL transitions."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity replay buffer storing (obs, base_action, delta, reward, next_obs, next_base_action, done).

    All arrays are stored as numpy. ``sample`` returns a dict of CUDA tensors.

    Args:
        capacity:           Maximum number of transitions.
        obs_dim:            Flattened observation dimension.
        base_action_shape:  Shape of the base action chunk, e.g. (chunk_len, act_dim).
        delta_dim:          Flattened delta action dimension (query_freq * act_dim).
        device:             Torch device for sampled batches.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        base_action_shape: tuple[int, ...],
        delta_dim: int,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.device = device
        self._size = 0
        self._ptr = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.base_action = np.zeros((capacity, *base_action_shape), dtype=np.float32)
        self.delta = np.zeros((capacity, delta_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_base_action = np.zeros((capacity, *base_action_shape), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

    @property
    def size(self) -> int:
        return self._size

    def insert(
        self,
        obs: np.ndarray,
        base_action: np.ndarray,
        delta: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_base_action: np.ndarray,
        done: bool,
    ) -> None:
        """Insert a single transition."""
        self.obs[self._ptr] = obs
        self.base_action[self._ptr] = base_action
        self.delta[self._ptr] = delta
        self.reward[self._ptr] = reward
        self.next_obs[self._ptr] = next_obs
        self.next_base_action[self._ptr] = next_base_action
        self.done[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random batch and return as CUDA tensors."""
        idxs = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idxs], device=self.device),
            "base_action": torch.as_tensor(self.base_action[idxs], device=self.device),
            "delta": torch.as_tensor(self.delta[idxs], device=self.device),
            "reward": torch.as_tensor(self.reward[idxs], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], device=self.device),
            "next_base_action": torch.as_tensor(self.next_base_action[idxs], device=self.device),
            "done": torch.as_tensor(self.done[idxs], device=self.device),
        }
