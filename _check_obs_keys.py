"""Verify that env wrapper + dataset produce matching obs dimensions."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

# 1) Check HDF5 dataset obs
import h5py
HDF5_PATH = os.path.expanduser(
    "~/LIBERO/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5"
)
obs_keys = ["ee_states", "gripper_states", "joint_states"]

with h5py.File(HDF5_PATH, "r") as f:
    demo = sorted(f["data"].keys())[0]
    parts = []
    for k in obs_keys:
        arr = f[f"data/{demo}/obs/{k}"][0].astype(np.float32)
        print(f"  HDF5 {k}: shape={arr.shape}")
        parts.append(arr.ravel())
    hdf5_obs = np.concatenate(parts)
    print(f"  HDF5 total obs_dim: {hdf5_obs.shape[0]}")

# 2) Check live env obs
print("\n--- Live env ---")
import mip.envs.libero._robosuite_compat  # noqa
from mip.envs.libero.libero_env_wrapper import LiberoGymWrapper

BDDL = os.path.expanduser(
    "~/LIBERO/libero/libero/bddl_files/libero_spatial/"
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl"
)
env = LiberoGymWrapper(bddl_file=BDDL, obs_keys=obs_keys)
obs, _ = env.reset()
print(f"  Env obs shape: {obs.shape}")
print(f"  Match: {hdf5_obs.shape[0] == obs.shape[0]}")
print(f"  HDF5 first 5: {hdf5_obs[:5]}")
print(f"  Env  first 5: {obs[:5]}")
env.close()
print("\nDone!")
