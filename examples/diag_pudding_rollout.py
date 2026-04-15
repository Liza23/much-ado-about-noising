"""Diagnostic rollout for pudding baseline model.

Loads the pudding model_best.pt checkpoint and runs a few steps in the LIBERO
pudding environment to observe what actions the robot produces.

Usage:
    python examples/diag_pudding_rollout.py
"""

import os
import sys

import numpy as np
import torch

# ── minimal config setup (no Hydra) ─────────────────────────────────────────
from dataclasses import dataclass, field
from mip.config import TaskConfig, OptimizationConfig, NetworkConfig, LogConfig, Config

# Build config that matches training exactly
task_cfg = TaskConfig(
    env_name="libero_object",
    env_type="state",
    obs_type="state",
    abs_action=False,
    obs_keys=["ee_states", "gripper_states", "joint_states"],
    obs_dim=15,
    act_dim=7,
    obs_steps=2,
    act_steps=8,
    horizon=16,
    num_envs=1,
    max_episode_steps=600,
)
# bddl_file is a YAML-only field not in the dataclass — attach it directly
task_cfg.bddl_file = (
    "/home/ldahiya/LIBERO/libero/libero/bddl_files/libero_object/"
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket.bddl"
)
task_cfg.dataset_path = (
    "/home/ldahiya/LIBERO/libero/datasets/libero_object/"
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5"
)

opt_cfg = OptimizationConfig(
    seed=0,
    loss_type="flow",
    loss_scale=0.1,
    lr=1e-4,
    weight_decay=1e-4,
    ema_rate=0.995,
    gradient_steps=300000,
    batch_size=256,
    sample_mode="zero",
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_compile=False,
)

net_cfg = NetworkConfig(
    network_type="mlp",
    num_layers=8,
    emb_dim=512,
    dropout=0.1,
    encoder_type="mlp",
)

log_cfg = LogConfig(
    log_dir="/data/user_data/ldahiya/libero-train/libero-object-state-pudding/baseline",
    wandb_mode="disabled",
    project="libero-experiments",
    group="libero-object-state-pudding",
    exp_name="diag",
)

config = Config(optimization=opt_cfg, network=net_cfg, task=task_cfg, log=log_cfg)

# ── dataset (for normalizer) ─────────────────────────────────────────────────
from mip.datasets.libero_dataset import make_dataset
print("Loading dataset for normalizer...")
dataset = make_dataset(task_cfg)
print(f"  Dataset loaded: {len(dataset)} samples")
obs_norm = dataset.normalizer['obs']['state']
act_norm = dataset.normalizer['action']
print(f"  obs normalizer: min={obs_norm.min}, max={obs_norm.max}")
print(f"  act normalizer: min={act_norm.min}, max={act_norm.max}")

# ── agent ────────────────────────────────────────────────────────────────────
from mip.agent import TrainingAgent
print("\nCreating agent...")
agent = TrainingAgent(config)

CKPT = "/data/user_data/ldahiya/libero-train/libero-object-state-pudding/baseline/models/model_best.pt"
print(f"Loading checkpoint: {CKPT}")
agent.load(CKPT, load_optimizer=False)
agent.eval()
print("  Checkpoint loaded OK")

device = config.optimization.device

# ── environment ──────────────────────────────────────────────────────────────
from mip.envs.libero.libero_env_wrapper import make_vec_env
print("\nCreating environment...")
eval_envs = make_vec_env(task_cfg, seed=1001)
obs, _ = eval_envs.reset()
print(f"  Initial obs shape: {obs.shape}")
# obs layout: [ee_states(6), gripper_states(2), joint_states(7)] per obs_step
# gripper_states are at indices 6:8 in each obs_step
# With obs_steps=2, obs[env_idx] shape = (obs_steps, 15)
gripper_qpos_after_warmup = obs[0, -1, 6:8]
print(f"  gripper_qpos after warm-up: {gripper_qpos_after_warmup}  (target ≈ [0.036, -0.036])")
print(f"  Initial obs (raw): {obs[0]}")

# ── rollout loop ─────────────────────────────────────────────────────────────
print("\n=== Rolling out pudding model for 10 policy steps ===")
print(f"{'Step':>4}  {'eef_pos (raw)':>30}  {'action_mean':>30}  {'action_std':>10}")

for step in range(10):
    obs_norm = dataset.normalizer["obs"]["state"].normalize(obs.astype(np.float32))
    obs_tensor = torch.tensor(obs_norm, device=device, dtype=torch.float32)

    act_0 = torch.randn(
        (task_cfg.num_envs, task_cfg.horizon, task_cfg.act_dim), device=device
    )
    with torch.no_grad():
        act_normed = agent.sample(
            act_0=act_0,
            obs={"state": obs_tensor},
            num_steps=1,
            use_ema=True,
        )

    act_normed_np = act_normed.detach().cpu().numpy()
    act = dataset.normalizer["action"].unnormalize(act_normed_np)

    # slice: obs_steps-1 .. obs_steps-1+act_steps
    start = task_cfg.obs_steps - 1
    end = start + task_cfg.act_steps
    act_slice = act[:, start:end, :]  # (1, 8, 7)

    # ee_states is first 6 dims (pos 3 + axis-angle 3)
    eef_pos_raw = obs[0, -1, :3]   # last obs_step, first 3 dims = eef pos

    print(
        f"{step*task_cfg.act_steps:>4}  "
        f"eef={eef_pos_raw}  "
        f"act_mean={act_slice[0].mean(axis=0).round(4)}  "
        f"act_std={act_slice[0].std(axis=0).round(4)}"
    )

    obs, reward, terminated, truncated, info = eval_envs.step(act_slice)

    success = False
    if "success" in info:
        s = info["success"]
        success = bool(np.asarray(s).any())
    if "_final_info" in info:
        for i in range(task_cfg.num_envs):
            if info["_final_info"][i]:
                fi = info["final_info"][i]
                if fi and "success" in fi:
                    success = bool(np.asarray(fi["success"]).any())

    if success:
        print(f"  --> SUCCESS at step {step*task_cfg.act_steps}!")
        break

print("\n=== Done. ===")
eval_envs.close()
