"""PostBCDataset: RobomimicDataset subclass that applies posterior action perturbation.

For each sample at index idx, perturbs the normalized action:
    ã = a + alpha * ws,  ws ~ N(0, diag(variance[idx]))

where variance[idx] is the per-sample, per-step, per-dim ensemble posterior variance
precomputed by EnsemblePredictor.compute_variance() and saved to a .npy file.
"""

from __future__ import annotations

import numpy as np
import torch

from mip.datasets.robomimic_dataset import RobomimicDataset


class PostBCDataset(RobomimicDataset):
    """RobomimicDataset with posterior action perturbation (PostBC Algorithm 2).

    Args:
        variance_path: Path to .npy file of shape (N, horizon, act_dim) produced by
            EnsemblePredictor.compute_variance().
        alpha: Perturbation scale (paper uses alpha=1.0 for Robomimic Lift).
        *args, **kwargs: Passed through to RobomimicDataset.
    """

    def __init__(self, *args, variance_path: str, alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.variance = np.load(variance_path).astype(np.float32)  # (N, horizon, act_dim)
        self.alpha = alpha

    def __getitem__(self, idx: int) -> dict:
        batch = super().__getitem__(idx)
        var = torch.from_numpy(self.variance[idx])   # (horizon, act_dim)
        std = var.clamp(min=0.0).sqrt()
        ws = torch.randn_like(std) * std             # ws ~ N(0, diag(var))
        # batch["action"] shape: (dataset_horizon, act_dim); only perturb the first
        # `horizon` steps — the variance covers exactly those steps.
        h = ws.shape[0]
        batch["action"][:h] = batch["action"][:h] + self.alpha * ws
        return batch
