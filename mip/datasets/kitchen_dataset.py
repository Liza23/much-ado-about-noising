"""Kitchen dataset.

Author: Chaoyi Pan
Date: 2025-10-17
"""

import os
import pathlib
import zipfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from tqdm import tqdm

from mip.dataset_utils import (
    MinMaxNormalizer,
    ReplayBuffer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset
from mip.envs.kitchen.kitchen_thirdparty.kitchen_util import parse_mjl_logs

# Absolute path to Kitchen MuJoCo XML (used for FK eef computation)
_KITCHEN_XML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../envs/kitchen/kitchen_thirdparty/relay_policy_learning/adept_envs/adept_envs/franka/assets/franka_kitchen_jntpos_act_ab.xml",
)


def download_kitchen_dataset(
    dataset_filename: str = "kitchen/kitchen_demos_multitask.zip",
    repo_id: str = "ChaoyiPan/mip-dataset",
) -> str:
    """Download Kitchen dataset from HuggingFace and extract locally.

    Args:
        dataset_filename: Filename in the HuggingFace dataset repo (should be a .zip file)
        repo_id: HuggingFace repository ID

    Returns:
        Path to the extracted dataset directory
    """
    logger.info(f"Downloading Kitchen dataset from {repo_id}/{dataset_filename}")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=dataset_filename,
        repo_type="dataset",
    )
    logger.info(f"Downloaded zip file to: {zip_path}")

    # Extract the zip file in the same directory
    zip_path_obj = Path(zip_path)
    extract_dir = zip_path_obj.parent

    logger.info(f"Extracting dataset to {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Find the extracted directory
    # The zip should contain a directory with observations_seq.npy, actions_seq.npy, existence_mask.npy
    dataset_name = zip_path_obj.stem  # Remove .zip extension
    dataset_path = extract_dir / dataset_name

    if not dataset_path.exists():
        # Look for the directory that was extracted
        extracted_dirs = [
            d
            for d in extract_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if extracted_dirs:
            dataset_path = extracted_dirs[0]
        else:
            raise FileNotFoundError(f"No directory found after extracting {zip_path}")

    logger.info(f"Extracted dataset to: {dataset_path}")
    return str(dataset_path)


def make_dataset(task_config, mode="train"):
    """Create Kitchen dataset based on configuration.

    Args:
        task_config: Task configuration with dataset parameters
        mode: Dataset mode (train/val), currently unused for Kitchen

    Returns:
        Dataset instance for Kitchen task
    """
    _ = mode  # Currently unused for Kitchen, but kept for API consistency

    # Check if we should download from HuggingFace
    if hasattr(task_config, "dataset_repo") and hasattr(
        task_config, "dataset_filename"
    ):
        # Auto-download from HuggingFace (handles zip extraction)
        dataset_filename = task_config.dataset_filename

        # If filename ends with .zip, download and extract
        if dataset_filename.endswith(".zip"):
            dataset_path = download_kitchen_dataset(
                dataset_filename=dataset_filename,
                repo_id=task_config.dataset_repo,
            )
        else:
            # Direct download (backward compatibility for non-zip files)
            logger.info(
                f"Downloading dataset from {task_config.dataset_repo}/{dataset_filename}"
            )
            dataset_path = hf_hub_download(
                repo_id=task_config.dataset_repo,
                filename=dataset_filename,
                repo_type="dataset",
            )
            logger.info(f"Downloaded dataset to: {dataset_path}")
    elif hasattr(task_config, "dataset_path"):
        # Use explicit path if provided
        dataset_path = str(os.path.expanduser(task_config.dataset_path))
    else:
        raise ValueError(
            "Either dataset_repo/dataset_filename or dataset_path must be provided"
        )

    # Read intent conditioning params
    intent_conditioning = getattr(task_config, "intent_conditioning", False)
    intent_horizon = getattr(task_config, "intent_horizon", task_config.act_steps)
    intent_type = getattr(task_config, "intent_type", "mean")

    if intent_conditioning:
        pad_after = max(task_config.act_steps, intent_horizon) - 1
        dataset_horizon = max(
            task_config.horizon, task_config.obs_steps + intent_horizon
        )
    else:
        pad_after = task_config.act_steps - 1
        dataset_horizon = task_config.horizon

    if task_config.obs_type == "state":
        return KitchenMjlDataset(
            dataset_dir=f"{dataset_path}/kitchen_demos_multitask",
            horizon=dataset_horizon,
            pad_before=task_config.obs_steps - 1,
            pad_after=pad_after,
            abs_action=task_config.abs_action,
            obs_steps=task_config.obs_steps,
            intent_conditioning=intent_conditioning,
            intent_horizon=intent_horizon,
            intent_type=intent_type,
        )
    else:
        raise ValueError(f"Invalid observation type: {task_config.obs_type}")


class KitchenMjlDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=True,
        robot_noise_ratio=0.1,
        obs_steps=2,
        intent_conditioning=False,
        intent_horizon=8,
        intent_type="mean",
    ):
        super().__init__()

        self.obs_steps = obs_steps
        self.intent_conditioning = intent_conditioning
        self.intent_horizon = intent_horizon
        self.intent_type = intent_type
        # eef intent is always 7D (3D pos + 4D quat wxyz), stored separately from obs
        self.intent_start = 0
        self.intent_end = 7

        # Load MuJoCo model once for forward-kinematics eef computation
        if intent_conditioning:
            import mujoco
            from scipy.spatial.transform import Rotation as ScipyRotation

            _mj_model = mujoco.MjModel.from_xml_path(_KITCHEN_XML_PATH)
            _mj_data = mujoco.MjData(_mj_model)
            _eef_id = mujoco.mj_name2id(
                _mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
            )
            logger.info(
                f"Loaded Kitchen MuJoCo model for FK (nq={_mj_model.nq}, eef_site_id={_eef_id})"
            )

        data_directory = pathlib.Path(dataset_dir)
        robot_pos_noise_amp = np.array(
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed=42)

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i, mjl_path in enumerate(tqdm(list(data_directory.glob("*/*.mjl")))):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data["qpos"].astype(np.float32)
                obs = np.concatenate(
                    [
                        qpos[:, :9],
                        qpos[:, -21:],
                        np.zeros((len(qpos), 30), dtype=np.float32),
                    ],
                    axis=-1,
                )
                if robot_noise_ratio > 0:
                    # add observation noise to match real robot
                    noise = (
                        robot_noise_ratio
                        * robot_pos_noise_amp
                        * rng.uniform(low=-1.0, high=1.0, size=(obs.shape[0], 30))
                    )
                    obs[:, :30] += noise
                episode = {
                    "state": obs,
                    "qpos": qpos,
                    "qvel": data["qvel"].astype(np.float32),
                    "action": data["ctrl"].astype(np.float32),
                }

                if intent_conditioning:
                    # Compute eef pos + quat (wxyz) via FK for each timestep.
                    # qpos has shape (T, nq); we set data.qpos and run mj_fwdPosition.
                    T = len(qpos)
                    eef = np.zeros((T, 7), dtype=np.float32)
                    for t in range(T):
                        _mj_data.qpos[:] = qpos[t]
                        mujoco.mj_fwdPosition(_mj_model, _mj_data)
                        eef[t, :3] = _mj_data.site_xpos[_eef_id]
                        # site_xmat is (9,) row-major 3×3; convert to wxyz quaternion
                        mat = _mj_data.site_xmat[_eef_id].reshape(3, 3)
                        q_xyzw = ScipyRotation.from_matrix(mat).as_quat()  # scalar-last
                        eef[t, 3] = q_xyzw[3]   # w
                        eef[t, 4:] = q_xyzw[:3]  # xyz
                    episode["eef"] = eef

                self.replay_buffer.add_episode(episode)
            except Exception as e:
                logger.warning(f"Error parsing {mjl_path}: {e}")

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        state_normalizer = MinMaxNormalizer(
            self.replay_buffer["state"][:]
        )  # (N, obs_dim)
        action_normalizer = MinMaxNormalizer(
            self.replay_buffer["action"][:]
        )  # (N, action_dim)
        norm = {"obs": {"state": state_normalizer}, "action": action_normalizer}
        if self.intent_conditioning:
            norm["eef"] = MinMaxNormalizer(self.replay_buffer["eef"][:])  # (N, 7)
        return norm

    def sample_to_data(self, sample):
        state = sample["state"].astype(np.float32)
        state = self.normalizer["obs"]["state"].normalize(state)

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)
        data = {
            "obs": {
                "state": state,
            },
            "action": action,
        }

        if self.intent_conditioning:
            eef = sample["eef"].astype(np.float32)
            eef_normed = self.normalizer["eef"].normalize(eef)
            future_eef = eef_normed[self.obs_steps : self.obs_steps + self.intent_horizon]
            if self.intent_type == "encoded_mean":
                intent = future_eef.astype(np.float32)  # (N, 7) — encode+pool in training loop
            else:  # "mean"
                intent = future_eef.mean(axis=0).astype(np.float32)  # (7,)
            data["intent"] = intent

        return data

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data
