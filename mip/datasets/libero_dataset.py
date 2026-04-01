"""LIBERO dataset.

Loads LIBERO HDF5 demonstration files into the MIP training pipeline.
HDF5 layout mirrors robomimic: data/{demo_N}/actions, data/{demo_N}/obs/{key}.
"""

import os

import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from mip.dataset_utils import (
    MinMaxNormalizer,
    ReplayBuffer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset


def make_dataset(task_config, mode="train"):
    """Create a LIBERO dataset from config.

    Expects either task_config.dataset_path (local .hdf5 file) or a list of
    paths in task_config.dataset_paths for multi-task loading.
    """
    if hasattr(task_config, "dataset_path") and task_config.dataset_path:
        dataset_paths = [os.path.expanduser(task_config.dataset_path)]
    elif hasattr(task_config, "dataset_paths") and task_config.dataset_paths:
        dataset_paths = [os.path.expanduser(p) for p in task_config.dataset_paths]
    else:
        raise ValueError("task_config must provide dataset_path or dataset_paths")

    return LiberoDataset(
        dataset_paths=dataset_paths,
        obs_keys=task_config.obs_keys,
        horizon=task_config.horizon,
        pad_before=task_config.obs_steps - 1,
        pad_after=task_config.act_steps - 1,
        obs_steps=task_config.obs_steps,
    )


class LiberoDataset(BaseDataset):
    """Dataset for LIBERO HDF5 demonstrations.

    Args:
        dataset_paths: List of .hdf5 file paths to load.
        obs_keys: List of observation keys to concatenate into the state vector.
        horizon: Sequence length drawn per sample.
        pad_before: Left padding (obs_steps - 1).
        pad_after: Right padding (act_steps - 1).
        obs_steps: Number of observation history steps.
    """

    def __init__(
        self,
        dataset_paths: list[str],
        obs_keys: list[str],
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        obs_steps: int = 2,
    ):
        super().__init__()
        self.obs_keys = obs_keys
        self.obs_steps = obs_steps

        self.replay_buffer = ReplayBuffer.create_empty_numpy()

        for path in dataset_paths:
            logger.info(f"Loading LIBERO dataset: {path}")
            self._load_hdf5(path)

        logger.info(
            f"Loaded {self.replay_buffer.n_episodes} episodes, "
            f"{self.replay_buffer.n_steps} total steps"
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )
        self.horizon = horizon
        self.normalizer = self._get_normalizer()

    def _load_hdf5(self, path: str):
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys())
            for demo_key in tqdm(demos, desc=f"Loading {os.path.basename(path)}"):
                actions = f[f"data/{demo_key}/actions"][()].astype(np.float32)  # (T, act_dim)
                obs_parts = []
                for key in self.obs_keys:
                    arr = f[f"data/{demo_key}/obs/{key}"][()].astype(np.float32)  # (T, dim)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    obs_parts.append(arr)
                state = np.concatenate(obs_parts, axis=-1)  # (T, obs_dim)
                self.replay_buffer.add_episode({"state": state, "action": actions})

    def _get_normalizer(self):
        state_normalizer = MinMaxNormalizer(self.replay_buffer["state"][:])
        action_normalizer = MinMaxNormalizer(self.replay_buffer["action"][:])
        return {"obs": {"state": state_normalizer}, "action": action_normalizer}

    def sample_to_data(self, sample):
        state = sample["state"].astype(np.float32)
        state = self.normalizer["obs"]["state"].normalize(state)
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)
        return {"obs": {"state": state}, "action": action}

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict:
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        return dict_apply(data, torch.tensor)
