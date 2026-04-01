"""Deterministic MLP action decoder for Config A (flow-intent + MLP-action).

Interface contract:
    obs_emb : (B, obs_steps, emb_dim) — encoded observation context from shared encoder
    intent  : (B, intent_dim)          — intent vector from flow intent model
    → action : (B, horizon, act_dim)   — action sequence

The decoder flattens obs_emb to (B, obs_steps * emb_dim), concatenates
intent, then passes through a residual MLP to produce the action sequence.

Design note: stochasticity lives entirely in the upstream flow intent model;
this decoder is deterministic given (obs_emb, intent).
"""

import torch
import torch.nn as nn


class MLPActionDecoder(nn.Module):
    """Deterministic action decoder: (obs_emb, intent) → action sequence.

    Flattens obs_emb across time steps, concatenates intent, and maps
    through a depth-n_layers MLP with LayerNorm + SiLU activations.

    Args:
        obs_steps  : int — number of observation time steps (To)
        emb_dim    : int — per-step embedding dim from the shared encoder
        intent_dim : int — intent vector dim (e.g. 7 for mean eef pos+quat)
        horizon    : int — action sequence length (Ta)
        act_dim    : int — action dim per step
        hidden_dim : int — width of all hidden layers (default: emb_dim)
        n_layers   : int — number of hidden Linear→LN→SiLU blocks
        dropout    : float — dropout rate applied in hidden blocks
    """

    def __init__(
        self,
        obs_steps: int,
        emb_dim: int,
        intent_dim: int,
        horizon: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_steps = obs_steps
        self.emb_dim = emb_dim
        self.horizon = horizon
        self.act_dim = act_dim

        # in_dim: flattened obs context + intent vector
        in_dim = obs_steps * emb_dim + intent_dim
        out_dim = horizon * act_dim

        # Input projection
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        ]
        # Hidden blocks
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        # Output head
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_emb: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_emb : (B, obs_steps, emb_dim) — context from shared encoder
            intent  : (B, intent_dim)          — intent vector (GT or sampled)
        Returns:
            action  : (B, horizon, act_dim)

        Shape assertions are placed here at the intent/action boundary so that
        any shape mismatch is caught before the MLP forward, not inside it.
        """
        # ── Shape assertions at the intent/action boundary ──────────────────
        assert obs_emb.dim() == 3, (
            f"obs_emb must be (B, obs_steps, emb_dim), got {obs_emb.shape}"
        )
        assert obs_emb.shape[1] == self.obs_steps, (
            f"obs_emb time dim must be {self.obs_steps}, got {obs_emb.shape[1]}"
        )
        assert obs_emb.shape[2] == self.emb_dim, (
            f"obs_emb emb dim must be {self.emb_dim}, got {obs_emb.shape[2]}"
        )
        assert intent.dim() == 2, (
            f"intent must be (B, intent_dim), got {intent.shape}"
        )

        B = obs_emb.shape[0]
        flat = obs_emb.reshape(B, -1)                      # (B, obs_steps * emb_dim)
        x = torch.cat([flat, intent], dim=-1)              # (B, obs_steps*emb_dim + intent_dim)
        out = self.net(x)                                  # (B, horizon * act_dim)
        return out.reshape(B, self.horizon, self.act_dim)  # (B, horizon, act_dim)
