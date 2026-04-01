"""Learned intent predictor for intent-conditioned flow policies.

Predicts the mean future end-effector pose (mean over next act_steps) from the
current normalized observation window.  Replaces the constant-velocity proxy at
inference time, eliminating the train/inference distribution mismatch.

Input  : (B, obs_steps * base_obs_dim)  — flattened, normalized obs (no intent)
Output : (B, intent_dim)                — predicted mean future eef (normalized)
Training: MSE against ground-truth intent from the dataset
"""

import torch
import torch.nn as nn


class IntentPredictor(nn.Module):
    """Two-layer MLP that predicts mean future eef from the current obs window."""

    def __init__(self, obs_steps: int, base_obs_dim: int, intent_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = obs_steps * base_obs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, intent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_steps, base_obs_dim) — normalized obs window (no intent appended)
        Returns:
            intent: (B, intent_dim)
        """
        B = obs.shape[0]
        return self.net(obs.reshape(B, -1))
