"""Ensemble MLP predictor for PostBC posterior variance estimation.

Reference: "Posterior Behavioral Cloning" (arXiv:2512.16911), Algorithm 1.

Trains K small MLP predictors on bootstrapped (sample-level) subsets of the
demonstration dataset.  After training, compute_variance() runs all K members
on every dataset sample and returns the element-wise variance of their action
predictions — this is the per-sample posterior covariance used by Algorithm 2
to perturb action targets during flow policy pretraining.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class EnsembleMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsemblePredictor:
    """Trains K MLP predictors on bootstrapped data; estimates posterior variance.

    Args:
        K: Number of ensemble members (paper uses K=100 for Robomimic).
        hidden_dim: Hidden layer width for each MLP.
        n_layers: Number of hidden layers.
        lr: Adam learning rate.
    """

    def __init__(self, K: int = 100, hidden_dim: int = 256, n_layers: int = 3, lr: float = 1e-3):
        self.K = K
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.members: list[EnsembleMLP] = []
        self._in_dim: int | None = None
        self._out_dim: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_data(
        self,
        dataset,
        obs_steps: int,
        horizon: int,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect all (obs, act) tensors from the dataset.

        Returns:
            obs_all: (N, obs_steps * obs_dim) normalized, flattened
            act_all: (N, horizon * act_dim) normalized, flattened
        """
        loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True,
        )
        obs_list: list[torch.Tensor] = []
        act_list: list[torch.Tensor] = []
        for batch in tqdm(loader, desc="Collecting dataset", leave=False):
            obs = batch["obs"]["state"]  # (B, dataset_horizon, obs_dim)
            act = batch["action"]        # (B, dataset_horizon, act_dim)
            obs_list.append(obs[:, :obs_steps, :].reshape(obs.shape[0], -1))
            act_list.append(act[:, :horizon, :].reshape(act.shape[0], -1))
        obs_all = torch.cat(obs_list, dim=0).float()
        act_all = torch.cat(act_list, dim=0).float()
        return obs_all, act_all

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_ensemble(
        self,
        dataset,
        obs_steps: int,
        horizon: int,
        n_epochs: int = 50,
        batch_size: int = 256,
        device: str = "cuda",
    ) -> None:
        """Train K ensemble members on bootstrapped subsets of the dataset.

        Bootstrap is at the sample level (sample N indices with replacement per member).
        Trajectory-level bootstrapping is a closer match to the paper but requires
        knowledge of episode boundaries; sample-level works well in practice.
        """
        obs_all, act_all = self._collect_data(dataset, obs_steps, horizon, device)
        N = obs_all.shape[0]
        self._in_dim = obs_all.shape[1]
        self._out_dim = act_all.shape[1]

        obs_all = obs_all.to(device)
        act_all = act_all.to(device)

        self.members = []
        for k in tqdm(range(self.K), desc="Training ensemble members"):
            model = EnsembleMLP(self._in_dim, self._out_dim, self.hidden_dim, self.n_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            # Bootstrap: sample N indices with replacement
            boot_idx = torch.randint(0, N, (N,), device=device)
            obs_boot = obs_all[boot_idx]
            act_boot = act_all[boot_idx]

            ds = TensorDataset(obs_boot, act_boot)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

            model.train()
            for _ in range(n_epochs):
                for obs_b, act_b in loader:
                    pred = model(obs_b)
                    loss = nn.functional.mse_loss(pred, act_b)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            self.members.append(model.cpu())

    def compute_variance(
        self,
        dataset,
        obs_steps: int,
        horizon: int,
        act_dim: int,
        device: str = "cuda",
        infer_batch_size: int = 512,
    ) -> np.ndarray:
        """Compute per-sample action variance across ensemble predictions.

        Returns:
            variance array of shape (N, horizon, act_dim)
        """
        if not self.members:
            raise RuntimeError("Call train_ensemble() before compute_variance().")

        obs_all, _ = self._collect_data(dataset, obs_steps, horizon, device)
        N = obs_all.shape[0]
        obs_all = obs_all.to(device)

        preds: list[np.ndarray] = []
        with torch.no_grad():
            for model in tqdm(self.members, desc="Computing ensemble predictions", leave=False):
                model.to(device)
                chunks: list[torch.Tensor] = []
                for i in range(0, N, infer_batch_size):
                    chunks.append(model(obs_all[i: i + infer_batch_size]).cpu())
                preds.append(torch.cat(chunks, dim=0).numpy())
                model.cpu()

        # preds: list of K arrays each (N, horizon*act_dim)
        preds_arr = np.stack(preds, axis=0)           # (K, N, horizon*act_dim)
        var = preds_arr.var(axis=0)                   # (N, horizon*act_dim)
        return var.reshape(N, horizon, act_dim).astype(np.float32)

    def save(self, path: str) -> None:
        if not self.members:
            raise RuntimeError("No trained members to save.")
        torch.save(
            {
                "members": [m.state_dict() for m in self.members],
                "K": self.K,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "in_dim": self._in_dim,
                "out_dim": self._out_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.K = ckpt["K"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.n_layers = ckpt["n_layers"]
        self._in_dim = ckpt["in_dim"]
        self._out_dim = ckpt["out_dim"]
        self.members = []
        for state in ckpt["members"]:
            m = EnsembleMLP(self._in_dim, self._out_dim, self.hidden_dim, self.n_layers)
            m.load_state_dict(state)
            m.eval()
            self.members.append(m)
