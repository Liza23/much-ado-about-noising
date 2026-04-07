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
    EmptyNormalizer,
    MinMaxNormalizer,
    ReplayBuffer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset


def make_dataset(task_config, mode="train"):
    """Create a LIBERO dataset from config."""
    if hasattr(task_config, "dataset_path") and task_config.dataset_path:
        dataset_paths = [os.path.expanduser(task_config.dataset_path)]
    elif hasattr(task_config, "dataset_paths") and task_config.dataset_paths:
        dataset_paths = [os.path.expanduser(p) for p in task_config.dataset_paths]
    else:
        raise ValueError("task_config must provide dataset_path or dataset_paths")

    intent_conditioning = getattr(task_config, "intent_conditioning", False)
    intent_horizon = getattr(task_config, "intent_horizon", task_config.act_steps)
    intent_type = getattr(task_config, "intent_type", "mean")
    intent_keys = getattr(task_config, "intent_keys", ["ee_states"])
    image_obs_keys = list(getattr(task_config, "image_obs_keys", None) or [])

    if intent_conditioning:
        pad_after = max(task_config.act_steps, intent_horizon) - 1
        dataset_horizon = max(
            task_config.horizon, task_config.obs_steps + intent_horizon
        )
    else:
        pad_after = task_config.act_steps - 1
        dataset_horizon = task_config.horizon

    return LiberoDataset(
        dataset_paths=dataset_paths,
        obs_keys=task_config.obs_keys,
        image_obs_keys=image_obs_keys,
        horizon=dataset_horizon,
        pad_before=task_config.obs_steps - 1,
        pad_after=pad_after,
        obs_steps=task_config.obs_steps,
        intent_conditioning=intent_conditioning,
        intent_horizon=intent_horizon,
        intent_type=intent_type,
        intent_keys=intent_keys,
    )


class LiberoDataset(BaseDataset):
    """Dataset for LIBERO HDF5 demonstrations.

    Args:
        dataset_paths: List of .hdf5 file paths to load.
        obs_keys: List of state observation keys to concatenate.
        image_obs_keys: List of image observation keys (e.g. ["agentview_rgb"]).
            When non-empty, obs_type becomes "image" and images are loaded
            as (T, C, H, W) uint8 alongside the state vector.
        horizon: Sequence length drawn per sample.
        pad_before: Left padding (obs_steps - 1).
        pad_after: Right padding (act_steps - 1).
        obs_steps: Number of observation history steps.
        intent_conditioning: Whether to extract intent from future eef.
        intent_horizon: Number of future steps for intent computation.
        intent_type: "mean" or "encoded_mean".
        intent_keys: HDF5 obs keys used for intent (e.g. ["ee_states"]).
    """

    def __init__(
        self,
        dataset_paths: list[str],
        obs_keys: list[str],
        image_obs_keys: list[str] | None = None,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        obs_steps: int = 2,
        intent_conditioning: bool = False,
        intent_horizon: int = 8,
        intent_type: str = "mean",
        intent_keys: list[str] | None = None,
    ):
        super().__init__()
        self.obs_keys = obs_keys
        self.image_obs_keys = image_obs_keys or []
        self.obs_type = "image" if self.image_obs_keys else "state"
        self.obs_steps = obs_steps
        self.intent_conditioning = intent_conditioning
        self.intent_horizon = intent_horizon
        self.intent_type = intent_type
        self.intent_keys = intent_keys or ["ee_states"]
        self.lowdim_keys = ["state"]
        self._state_key_dims = {}

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

        # Track where the intent source lives inside the concatenated state vector.
        self.intent_start = None
        self.intent_end = None
        offset = 0
        for key in self.obs_keys:
            key_dim = self._state_key_dims.get(key)
            if key_dim is None:
                continue
            if key in self.intent_keys and self.intent_start is None:
                self.intent_start = offset
            if key in self.intent_keys:
                self.intent_end = offset + key_dim
            offset += key_dim

        self.normalizer = self._get_normalizer()

    def _load_hdf5(self, path: str):
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys())
            for demo_key in tqdm(demos, desc=f"Loading {os.path.basename(path)}"):
                actions = f[f"data/{demo_key}/actions"][()].astype(np.float32)
                obs_parts = []
                for key in self.obs_keys:
                    arr = f[f"data/{demo_key}/obs/{key}"][()].astype(np.float32)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    if key not in self._state_key_dims:
                        self._state_key_dims[key] = int(arr.shape[-1])
                    obs_parts.append(arr)
                state = np.concatenate(obs_parts, axis=-1)

                episode = {"state": state, "action": actions}

                # Image observations: load as uint8 CHW (convert from HWC).
                for img_key in self.image_obs_keys:
                    imgs = f[f"data/{demo_key}/obs/{img_key}"][()]  # (T, H, W, C)
                    imgs = np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))  # (T, C, H, W)
                    episode[img_key] = imgs  # uint8

                if self.intent_conditioning:
                    eef_parts = []
                    for key in self.intent_keys:
                        arr = f[f"data/{demo_key}/obs/{key}"][()].astype(np.float32)
                        if arr.ndim == 1:
                            arr = arr[:, None]
                        eef_parts.append(arr)
                    episode["eef"] = np.concatenate(eef_parts, axis=-1)

                self.replay_buffer.add_episode(episode)

    def _get_normalizer(self):
        state_normalizer = MinMaxNormalizer(self.replay_buffer["state"][:])
        action_normalizer = MinMaxNormalizer(self.replay_buffer["action"][:])
        norm = {"obs": {"state": state_normalizer}, "action": action_normalizer}
        for img_key in self.image_obs_keys:
            # LIBERO images are already represented in the same [0, 1] range at
            # train and eval time, so keep the image path identity-normalized.
            norm["obs"][img_key] = EmptyNormalizer()
        if self.intent_conditioning:
            norm["eef"] = MinMaxNormalizer(self.replay_buffer["eef"][:])
        return norm

    def sample_to_data(self, sample):
        state = sample["state"].astype(np.float32)
        state = self.normalizer["obs"]["state"].normalize(state)
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        obs = {"state": state}
        # Images: (horizon, C, H, W) uint8 → float32 in [0, 1]
        for img_key in self.image_obs_keys:
            obs[img_key] = sample[img_key].astype(np.float32) / 255.0

        data = {"obs": obs, "action": action}

        if self.intent_conditioning:
            eef = sample["eef"].astype(np.float32)
            eef_normed = self.normalizer["eef"].normalize(eef)
            future_eef = eef_normed[self.obs_steps : self.obs_steps + self.intent_horizon]
            if self.intent_type == "encoded_mean":
                data["intent"] = future_eef.astype(np.float32)  # (N, eef_dim)
            else:  # "mean"
                data["intent"] = future_eef.mean(axis=0).astype(np.float32)  # (eef_dim,)

        return data

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict:
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        return dict_apply(data, torch.tensor)
