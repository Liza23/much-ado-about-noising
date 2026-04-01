"""Per-step intent encoder for the encoded-mean intent variant.

Applies a small MLP to each raw intent step (eef_pos + eef_quat = 7D) independently,
then the caller takes the mean over steps to get the final intent embedding.

Motivation: taking the arithmetic mean of quaternions is only geometrically valid when
rotations are small. By encoding each step into a learned Euclidean space first, the
mean pooling is well-defined and the encoder can learn whatever representation is most
useful for the downstream policy.

Interface contract:
    Input  : (B, N, raw_intent_dim)  — N normalised future eef steps
    Output : (B, N, intent_emb_dim)  — per-step embeddings

    Caller is responsible for pooling over N:
        intent_emb = encoder(seq).mean(dim=1)  # (B, intent_emb_dim)
"""

import torch
import torch.nn as nn


class IntentEncoder(nn.Module):
    """Small per-step MLP: broadcasts over the N-steps dimension.

    Args:
        raw_intent_dim : dimension of each raw intent step (default 7 = pos3 + quat4)
        intent_emb_dim : output embedding dimension (default 64)
        hidden_dim     : hidden width (default 128)
    """

    def __init__(
        self,
        raw_intent_dim: int = 7,
        intent_emb_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.raw_intent_dim = raw_intent_dim
        self.intent_emb_dim = intent_emb_dim
        self.net = nn.Sequential(
            nn.Linear(raw_intent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, intent_emb_dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode each future eef step independently.

        Args:
            seq: (B, N, raw_intent_dim) — N normalised future eef steps

        Returns:
            emb: (B, N, intent_emb_dim) — per-step embeddings

        Shape assertions:
            seq.shape[-1] must equal self.raw_intent_dim
        """
        assert seq.shape[-1] == self.raw_intent_dim, (
            f"IntentEncoder: expected last dim {self.raw_intent_dim}, got {seq.shape[-1]}. "
            f"Full shape: {seq.shape}"
        )
        return self.net(seq)  # nn.Linear broadcasts over (B, N)
