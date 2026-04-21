"""Reproduce the multi-task eval hang quickly (no training).

Loads a baseline checkpoint, instantiates a fresh PARL agent (random weights — we
don't care about performance), and runs exactly one `evaluate_multitask` call
with very short episodes. The goal is to see *where* it hangs during env cycling.

Usage:
    python examples/debug_residual_eval.py \
        task=libero_spatial_suite_image \
        residual_parl.flow_intent_ckpt=/path/to/baseline.pt \
        +residual_parl.flow_num_steps=9 \
        task.max_episode_steps=64 \
        log.eval_episodes=5
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import hydra
import loguru
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.agent import TrainingAgent
from mip.config import Config
from mip.residual_parl.parl_agent import ResidualPARLAgent, PARLConfig
from mip.torch_utils import set_seed


def _make_task_env(config, bddl_file, seed=0):
    from mip.envs.libero.libero_env_wrapper import make_vec_env
    old_bddl = config.task.bddl_file
    config.task.bddl_file = bddl_file
    envs = make_vec_env(config.task, seed=seed)
    config.task.bddl_file = old_bddl
    return envs


def _get_task_list(config):
    bddl_files = list(getattr(config.task, "bddl_files", None) or [])
    if not bddl_files:
        bddl_files = [config.task.bddl_file]
    tasks = []
    for bf in bddl_files:
        name = os.path.splitext(os.path.basename(bf))[0]
        tasks.append((name, bf))
    return tasks


def _build_obs_td(obs_raw, dataset, image_obs_keys, num_envs, device):
    state_norm = dataset.normalizer["obs"]["state"].normalize(
        obs_raw["state"].astype(np.float32)
    )
    obs_dict = {"state": torch.tensor(state_norm, device=device, dtype=torch.float32)}
    for k in image_obs_keys:
        obs_dict[k] = torch.tensor(
            obs_raw[k].astype(np.float32), device=device, dtype=torch.float32
        )
    return TensorDict(obs_dict, batch_size=num_envs)


@hydra.main(version_base=None, config_path="configs/", config_name="residual_parl")
def main(raw_cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)
    config = Config(
        optimization=OmegaConf.structured(raw_cfg.optimization) if "optimization" in raw_cfg else None,
        network=OmegaConf.structured(raw_cfg.network) if "network" in raw_cfg else None,
        task=OmegaConf.structured(raw_cfg.task) if "task" in raw_cfg else None,
        log=OmegaConf.structured(raw_cfg.log) if "log" in raw_cfg else None,
    )
    parl_cfg_dict = cfg_dict.get("residual_parl", {})
    flow_num_steps = int(parl_cfg_dict.get("flow_num_steps", 9))

    device = config.optimization.device
    set_seed(0)
    loguru.logger.info(f"task.env_name={config.task.env_name} obs_type={config.task.obs_type}")

    # Dataset (for normalizers only)
    from mip.datasets.libero_dataset import make_dataset
    t0 = time.time()
    dataset = make_dataset(config.task, mode="train")
    loguru.logger.info(f"Dataset loaded in {time.time()-t0:.1f}s ({len(dataset)} samples)")

    tasks = _get_task_list(config)
    image_obs_keys = list(getattr(config.task, "image_obs_keys", None) or [])
    loguru.logger.info(f"Suite has {len(tasks)} tasks; image keys = {image_obs_keys}")

    # Load baseline
    config.task.obs_dim = config.network.emb_dim
    base_ckpt = parl_cfg_dict["flow_intent_ckpt"]
    t0 = time.time()
    base_agent = TrainingAgent(config)
    base_agent.load(base_ckpt)
    for mod in [base_agent.encoder, base_agent.encoder_ema, base_agent.flow_map, base_agent.flow_map_ema]:
        for p in mod.parameters():
            p.requires_grad_(False)
    loguru.logger.info(f"Baseline loaded in {time.time()-t0:.1f}s from {base_ckpt}")

    # Random-init PARL agent (we don't care about performance)
    emb_dim = config.network.emb_dim
    obs_dim_flat = config.task.obs_steps * emb_dim
    parl_config = PARLConfig(
        device=device, residual_alpha=1.0, action_magnitude=0.1,
    )
    agent = ResidualPARLAgent(
        config=parl_config, obs_dim=obs_dim_flat,
        act_dim=config.task.act_dim, query_freq=config.task.act_steps,
    )
    loguru.logger.info(f"PARL agent created (random init), obs_dim={obs_dim_flat}")

    num_envs = config.task.num_envs
    act_steps = config.task.act_steps
    horizon = config.task.horizon
    act_dim = config.task.act_dim
    start_idx = config.task.obs_steps - 1
    end_idx = start_idx + act_steps
    max_ep = config.task.max_episode_steps
    num_eps_per_task = max(1, config.log.eval_episodes // num_envs)

    loguru.logger.info(
        f"Eval plan: {len(tasks)} tasks × {num_eps_per_task} batches × {num_envs} envs × "
        f"up to {max_ep} sim steps. total up to ~{len(tasks) * num_eps_per_task * num_envs * max_ep} steps"
    )

    # -------- The test loop --------
    agent.eval()
    for task_idx, (task_name, bddl_file) in enumerate(tasks):
        loguru.logger.info(f"[{task_idx+1}/{len(tasks)}] create envs for {task_name}")
        t_create = time.time()
        envs = _make_task_env(config, bddl_file, seed=task_idx)
        loguru.logger.info(f"  create took {time.time()-t_create:.1f}s")

        for ep_batch in range(num_eps_per_task):
            t_reset = time.time()
            obs_raw, _ = envs.reset()
            loguru.logger.info(f"  [ep {ep_batch+1}] reset took {time.time()-t_reset:.1f}s")

            t = 0
            t_loop = time.time()
            while t < max_ep:
                obs_td = _build_obs_td(obs_raw, dataset, image_obs_keys, num_envs, device)
                with torch.no_grad():
                    act_0 = torch.randn(num_envs, horizon, act_dim, device=device)
                    base_actions_norm = base_agent.sample(
                        act_0=act_0, obs=obs_td, num_steps=flow_num_steps, use_ema=True,
                    )
                    obs_emb = base_agent.encoder_ema(obs_td, None)
                    if obs_emb.dim() == 2:
                        obs_emb = obs_emb.unsqueeze(1)
                base_chunk = base_actions_norm[:, start_idx:end_idx, :]
                obs_emb_flat = obs_emb.reshape(num_envs, -1).cpu().numpy()

                all_a_exec = []
                for ei in range(num_envs):
                    base_flat = base_chunk[ei].reshape(-1).cpu().numpy()
                    a_exec, _, _ = agent.sample_action(
                        obs_emb_flat[ei], base_flat, deterministic=True
                    )
                    all_a_exec.append(a_exec)
                a_exec_flat = np.stack(all_a_exec)
                act_norm = a_exec_flat.reshape(num_envs, act_steps, act_dim)
                act = dataset.normalizer["action"].unnormalize(act_norm)

                obs_raw, reward, terminated, truncated, info = envs.step(act)
                t += act_steps
            loguru.logger.info(f"  [ep {ep_batch+1}] {t} sim steps in {time.time()-t_loop:.1f}s")

        t_close = time.time()
        envs.close()
        loguru.logger.info(f"  close took {time.time()-t_close:.1f}s")

    loguru.logger.info("Full eval completed without hanging.")


if __name__ == "__main__":
    main()
