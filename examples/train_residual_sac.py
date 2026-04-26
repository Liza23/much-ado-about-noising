"""Training pipeline for Residual SAC on top of a frozen flow-intent policy.

Architecture:
    flow_intent (frozen) -> flow_action
    SAC residual          -> delta_action
    final_action = clip(flow_action + alpha * delta_action, -1, 1)

Usage:
    python examples/train_residual_sac.py \
        task=lift_ph_state \
        residual_sac.flow_intent_ckpt=/path/to/checkpoint.pt
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
import loguru
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Set MuJoCo rendering backend before importing robomimic/mujoco modules.
os.environ.setdefault("MUJOCO_GL", "egl")

from mip.config import Config
from mip.flow_intent_agent import FlowIntentAgent
from mip.logger import Logger
from mip.residual_sac.replay_buffer import ReplayBuffer
from mip.residual_sac.sac_agent import ResidualSACAgent, SACConfig
from mip.torch_utils import set_seed


def _make_dataset_and_envs(config: Config):
    """Create dataset and envs for either LIBERO or robomimic tasks."""
    env_name = config.task.env_name
    if env_name.startswith("libero"):
        from mip.datasets.libero_dataset import make_dataset
        from mip.envs.libero.libero_env_wrapper import make_vec_env
    else:
        from mip.datasets.robomimic_dataset import make_dataset
        from mip.envs.robomimic.robomimic_env import make_vec_env

    dataset = make_dataset(config.task, mode="train")
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    return dataset, envs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_obs(obs: np.ndarray) -> np.ndarray:
    """Flatten (num_envs, obs_steps, obs_dim) -> (num_envs, obs_steps * obs_dim)."""
    ne = obs.shape[0]
    return obs.reshape(ne, -1)


def evaluate(
    sac: ResidualSACAgent,
    envs,
    dataset,
    config: Config,
    flow_intent: FlowIntentAgent,
    num_episodes: int = 10,
) -> dict[str, float]:
    """Roll out the residual policy and return success metrics."""
    sac.eval()
    num_envs = config.task.num_envs
    episode_rewards = []
    episode_success = []

    for _ in range(num_episodes // num_envs):
        obs_raw, _ = envs.reset()
        ep_reward = np.zeros(num_envs)
        t = 0

        while t < config.task.max_episode_steps:
            # Normalize obs
            obs_norm = dataset.normalizer["obs"]["state"].normalize(
                obs_raw.astype(np.float32)
            )
            obs_tensor = torch.tensor(
                obs_norm, device=config.optimization.device, dtype=torch.float32
            )  # (num_envs, obs_steps, obs_dim)

            # Query frozen flow-intent for base actions
            with torch.no_grad():
                base_actions_norm = flow_intent.sample(
                    obs_tensor, use_ema=True
                )  # (num_envs, horizon, act_dim)
            # Slice to act_steps and convert
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            base_chunk_norm = base_actions_norm[:, start:end, :]  # (num_envs, act_steps, act_dim)

            # Flatten obs for SAC
            obs_flat = flatten_obs(obs_norm)  # (num_envs, obs_steps * obs_dim)

            # Sample delta from SAC actor (deterministic at eval)
            all_a_exec = []
            for ei in range(num_envs):
                base_flat = base_chunk_norm[ei].reshape(-1).cpu().numpy()
                a_exec, _delta, _lp = sac.sample_action(
                    obs_flat[ei], base_flat, deterministic=True
                )
                all_a_exec.append(a_exec)
            a_exec_flat = np.stack(all_a_exec)  # (num_envs, act_dim_flat)

            # Reshape to (num_envs, act_steps, act_dim) and unnormalize
            act_norm = a_exec_flat.reshape(num_envs, config.task.act_steps, config.task.act_dim)
            act = dataset.normalizer["action"].unnormalize(act_norm)

            # Undo rotation transform for robomimic (not needed for LIBERO)
            if hasattr(dataset, "undo_transform_action") and config.task.abs_action and config.task.env_name in [
                "can", "lift", "square", "tool_hang", "transport",
            ]:
                act = dataset.undo_transform_action(act)

            obs_raw, reward, terminated, truncated, info = envs.step(act)
            ep_reward += reward
            t += config.task.act_steps

        success = [1.0 if r > 0 else 0.0 for r in ep_reward]
        episode_rewards.extend(ep_reward.tolist())
        episode_success.extend(success)

    sac.train()
    return {
        "eval/mean_reward": np.mean(episode_rewards),
        "eval/mean_success": np.mean(episode_success),
        "eval/num_episodes": len(episode_success),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs/", config_name="residual_sac")
def main(raw_cfg: DictConfig) -> None:
    # ---- Resolve OmegaConf -> typed Config --------------------------------
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)

    # Build the MIP Config from the nested dict (task, network, optimization, log)
    config = Config(
        optimization=OmegaConf.structured(raw_cfg.optimization)
        if "optimization" in raw_cfg
        else None,
        network=OmegaConf.structured(raw_cfg.network)
        if "network" in raw_cfg
        else None,
        task=OmegaConf.structured(raw_cfg.task)
        if "task" in raw_cfg
        else None,
        log=OmegaConf.structured(raw_cfg.log)
        if "log" in raw_cfg
        else None,
    )
    sac_cfg_dict = cfg_dict.get("residual_sac", {})

    device = config.optimization.device
    set_seed(config.optimization.seed)
    loguru.logger.info(f"Config:\n{OmegaConf.to_yaml(raw_cfg)}")
    config.log.log_dir = cfg_dict["log"]["log_dir"]

    # ---- Dataset + Environment (auto-selects LIBERO vs robomimic) ----------
    dataset, envs = _make_dataset_and_envs(config)
    loguru.logger.info(f"Dataset loaded ({len(dataset)} samples)")
    loguru.logger.info(f"Envs created: {config.task.env_name} x {config.task.num_envs}")

    # ---- Frozen flow-intent -----------------------------------------------
    flow_intent_ckpt = sac_cfg_dict["flow_intent_ckpt"]
    assert flow_intent_ckpt, "residual_sac.flow_intent_ckpt must be set"

    # Set obs_dim from actual dataset so the encoder matches the checkpoint
    base_obs_dim = dataset[0]["obs"]["state"].shape[-1]
    config.task.obs_dim = base_obs_dim

    # The flow-intent agent needs the full MIP config for architecture.
    # We reuse the same config (task + network) that was used for pretraining.
    flow_intent = FlowIntentAgent(config)
    flow_intent.load(flow_intent_ckpt)
    # Freeze everything
    for mod in [flow_intent.encoder, flow_intent.intent_flow_map]:
        for p in mod.parameters():
            p.requires_grad_(False)
    if flow_intent.action_decoder is not None:
        for p in flow_intent.action_decoder.parameters():
            p.requires_grad_(False)
    if flow_intent.action_flow_map is not None:
        for p in flow_intent.action_flow_map.parameters():
            p.requires_grad_(False)
    loguru.logger.info(f"Frozen flow-intent loaded from {flow_intent_ckpt}")

    # ---- SAC agent --------------------------------------------------------
    obs_dim_per_step = dataset[0]["obs"]["state"].shape[-1]
    obs_dim_flat = config.task.obs_steps * obs_dim_per_step
    act_dim = config.task.act_dim
    act_steps = config.task.act_steps

    sac_config = SACConfig(
        actor_lr=sac_cfg_dict.get("actor_lr", 3e-4),
        critic_lr=sac_cfg_dict.get("critic_lr", 3e-4),
        alpha_lr=sac_cfg_dict.get("alpha_lr", 3e-4),
        hidden_dims=tuple(sac_cfg_dict.get("hidden_dims", [256, 256, 256])),
        discount=sac_cfg_dict.get("discount", 0.99),
        tau=sac_cfg_dict.get("tau", 0.005),
        init_alpha=sac_cfg_dict.get("init_alpha", 0.1),
        residual_alpha=sac_cfg_dict.get("residual_alpha", 1.0),
        predict_a_exec=sac_cfg_dict.get("predict_a_exec", False),
        backup_entropy=sac_cfg_dict.get("backup_entropy", True),
        device=device,
    )
    target_entropy = sac_cfg_dict.get("target_entropy", "auto")
    if target_entropy != "auto":
        sac_config.target_entropy = float(target_entropy)

    sac = ResidualSACAgent(
        config=sac_config,
        obs_dim=obs_dim_flat,
        act_dim=act_dim,
        query_freq=act_steps,
    )
    sac.set_flow_intent(flow_intent)
    loguru.logger.info(
        f"ResidualSACAgent created: obs_dim={obs_dim_flat}, act_dim={act_dim}, "
        f"query_freq={act_steps}, alpha={sac_config.residual_alpha}"
    )

    # ---- Replay buffer ----------------------------------------------------
    buffer_capacity = sac_cfg_dict.get("buffer_capacity", 1_000_000)
    buffer = ReplayBuffer(
        capacity=buffer_capacity,
        obs_dim=obs_dim_flat,
        base_action_shape=(act_steps, act_dim),
        delta_dim=act_steps * act_dim,
        device=device,
    )

    # ---- Logger -----------------------------------------------------------
    logger = Logger(config)

    # ---- Training loop ----------------------------------------------------
    total_env_steps = sac_cfg_dict.get("total_env_steps", 1_000_000)
    warmup_steps = sac_cfg_dict.get("warmup_steps", 1000)
    batch_size = sac_cfg_dict.get("batch_size", 256)
    num_updates = sac_cfg_dict.get("num_updates_per_step", 1)
    eval_freq = config.log.eval_freq
    log_freq = config.log.log_freq
    save_freq = config.log.save_freq
    eval_episodes = config.log.eval_episodes

    save_dir = Path(config.log.log_dir) / "residual_sac_checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    obs_raw, _ = envs.reset()
    # Flow-intent query: get initial base actions
    obs_norm = dataset.normalizer["obs"]["state"].normalize(obs_raw.astype(np.float32))
    obs_tensor = torch.tensor(obs_norm, device=device, dtype=torch.float32)

    with torch.no_grad():
        base_actions_norm = flow_intent.sample(obs_tensor, use_ema=True)
    start_idx = config.task.obs_steps - 1
    end_idx = start_idx + act_steps
    base_chunk_norm = base_actions_norm[:, start_idx:end_idx, :].cpu().numpy()

    global_step = 0
    episode_count = 0
    ep_reward = np.zeros(config.task.num_envs)
    ep_len = np.zeros(config.task.num_envs, dtype=int)
    best_success = -1.0

    loguru.logger.info(f"Starting training for {total_env_steps} env steps")

    while global_step < total_env_steps:
        t_start = time.time()

        # --- Action selection ---
        obs_flat = flatten_obs(obs_norm)  # (num_envs, obs_steps * obs_dim)

        all_a_exec = []
        all_delta = []
        for ei in range(config.task.num_envs):
            base_flat = base_chunk_norm[ei].reshape(-1)
            if global_step < warmup_steps:
                # Small random perturbations during warmup — keep base policy intact
                delta = np.random.normal(0, 0.1, size=(act_steps * act_dim,)).astype(np.float32).clip(-1, 1)
                if sac_config.predict_a_exec:
                    a_exec = (base_flat + sac_config.residual_alpha * delta).clip(-1, 1)
                else:
                    a_exec = (base_flat + sac_config.residual_alpha * delta).clip(-1, 1)
                all_a_exec.append(a_exec)
                all_delta.append(delta)
            else:
                a_exec, delta, _lp = sac.sample_action(obs_flat[ei], base_flat)
                all_a_exec.append(a_exec)
                all_delta.append(delta)

        a_exec_flat = np.stack(all_a_exec)  # (num_envs, act_dim_flat)
        delta_flat = np.stack(all_delta)

        # --- Env step ---
        act_norm = a_exec_flat.reshape(config.task.num_envs, act_steps, act_dim)
        act = dataset.normalizer["action"].unnormalize(act_norm)
        if config.task.abs_action and config.task.env_name in [
            "can", "lift", "square", "tool_hang", "transport",
        ]:
            act = dataset.undo_transform_action(act)

        next_obs_raw, reward, terminated, truncated, info = envs.step(act)
        done = terminated | truncated
        ep_reward += reward
        ep_len += act_steps
        global_step += act_steps * config.task.num_envs

        # Normalize next obs and get next base actions
        next_obs_norm = dataset.normalizer["obs"]["state"].normalize(
            next_obs_raw.astype(np.float32)
        )
        next_obs_tensor = torch.tensor(next_obs_norm, device=device, dtype=torch.float32)
        with torch.no_grad():
            next_base_norm = flow_intent.sample(next_obs_tensor, use_ema=True)
        next_base_chunk = next_base_norm[:, start_idx:end_idx, :].cpu().numpy()

        # --- Store transitions ---
        next_obs_flat = flatten_obs(next_obs_norm)
        for ei in range(config.task.num_envs):
            buffer.insert(
                obs=obs_flat[ei],
                base_action=base_chunk_norm[ei],
                delta=delta_flat[ei],
                reward=float(reward[ei]),
                next_obs=next_obs_flat[ei],
                next_base_action=next_base_chunk[ei],
                done=bool(done[ei]),
            )

        # --- Handle episode resets ---
        for ei in range(config.task.num_envs):
            if done[ei]:
                episode_count += 1
                if global_step >= warmup_steps and global_step % log_freq < act_steps * config.task.num_envs:
                    logger.log({
                        "step": global_step,
                        "train/ep_reward": ep_reward[ei],
                        "train/ep_length": int(ep_len[ei]),
                        "train/episode": episode_count,
                    }, category="train")
                ep_reward[ei] = 0.0
                ep_len[ei] = 0

        # Advance state
        obs_raw = next_obs_raw
        obs_norm = next_obs_norm
        obs_tensor = next_obs_tensor
        base_chunk_norm = next_base_chunk

        # --- SAC updates ---
        if buffer.size >= warmup_steps:
            update_info = {}
            for _ in range(num_updates):
                batch = buffer.sample(batch_size)
                info_step = sac.update(batch)
                # Accumulate for logging
                for k, v in info_step.items():
                    update_info[k] = update_info.get(k, 0.0) + v / num_updates

            if global_step % log_freq < act_steps * config.task.num_envs:
                update_info["step"] = global_step
                update_info["train/buffer_size"] = buffer.size
                update_info["train/global_step"] = global_step
                logger.log(update_info, category="train")

        # --- Evaluation ---
        if global_step % eval_freq < act_steps * config.task.num_envs:
            eval_metrics = evaluate(
                sac, envs, dataset, config, flow_intent,
                num_episodes=eval_episodes,
            )
            eval_metrics["step"] = global_step
            logger.log(eval_metrics, category="eval")
            loguru.logger.info(
                f"Step {global_step}: success={eval_metrics['eval/mean_success']:.2f}, "
                f"reward={eval_metrics['eval/mean_reward']:.2f}"
            )

            if eval_metrics["eval/mean_success"] > best_success:
                best_success = eval_metrics["eval/mean_success"]
                sac.save(str(save_dir / "best.pt"))
                loguru.logger.info(f"New best success: {best_success:.2f}")

            # Re-reset env after eval
            obs_raw, _ = envs.reset()
            obs_norm = dataset.normalizer["obs"]["state"].normalize(
                obs_raw.astype(np.float32)
            )
            obs_tensor = torch.tensor(obs_norm, device=device, dtype=torch.float32)
            with torch.no_grad():
                base_actions_norm = flow_intent.sample(obs_tensor, use_ema=True)
            base_chunk_norm = base_actions_norm[:, start_idx:end_idx, :].cpu().numpy()
            ep_reward[:] = 0.0
            ep_len[:] = 0

        # --- Periodic save ---
        if global_step % save_freq < act_steps * config.task.num_envs:
            sac.save(str(save_dir / f"step_{global_step}.pt"))

    # Final save
    sac.save(str(save_dir / "final.pt"))
    loguru.logger.info(f"Training complete. Best success: {best_success:.2f}")


if __name__ == "__main__":
    main()
