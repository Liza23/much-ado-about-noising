"""Multi-task Residual PARL: one residual policy for an entire LIBERO suite.

Each episode samples a random task from the suite. Evaluation reports per-task
and suite-mean success. The frozen flow-intent must also be a multi-task
checkpoint trained on the same suite.

Usage:
    python examples/train_residual_parl_multitask.py \
        task=libero_10_state_flow_intent \
        residual_parl.flow_intent_ckpt=/path/to/multi_task_checkpoint.pt
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

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.config import Config
from mip.flow_intent_agent import FlowIntentAgent
from mip.logger import Logger
from mip.residual_sac.replay_buffer import ReplayBuffer
from mip.residual_parl.parl_agent import ResidualPARLAgent, PARLConfig
from mip.torch_utils import set_seed


# ---------------------------------------------------------------------------
# Multi-task env helpers
# ---------------------------------------------------------------------------

def _make_task_env(config, bddl_file, seed=0):
    """Create a vectorized env for a single task."""
    from mip.envs.libero.libero_env_wrapper import make_vec_env
    # Temporarily override bddl_file
    old_bddl = config.task.bddl_file
    config.task.bddl_file = bddl_file
    envs = make_vec_env(config.task, seed=seed)
    config.task.bddl_file = old_bddl
    return envs


def _get_task_list(config):
    """Extract list of (task_name, bddl_file) from config."""
    bddl_files = list(getattr(config.task, "bddl_files", None) or [])
    if not bddl_files:
        # Single task fallback
        bddl_files = [config.task.bddl_file]
    tasks = []
    for bf in bddl_files:
        name = os.path.splitext(os.path.basename(bf))[0]
        tasks.append((name, bf))
    return tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_obs(obs: np.ndarray) -> np.ndarray:
    """Flatten (num_envs, obs_steps, obs_dim) -> (num_envs, obs_steps * obs_dim)."""
    return obs.reshape(obs.shape[0], -1)


def evaluate_multitask(
    agent: ResidualPARLAgent,
    tasks: list[tuple[str, str]],
    dataset,
    config: Config,
    flow_intent: FlowIntentAgent,
    num_episodes_per_task: int = 10,
) -> dict[str, float]:
    """Evaluate on all tasks in the suite."""
    agent.eval()
    device = config.optimization.device
    num_envs = config.task.num_envs
    act_steps = config.task.act_steps
    start_idx = config.task.obs_steps - 1
    end_idx = start_idx + act_steps

    all_success = []
    per_task_success = {}

    for task_name, bddl_file in tasks:
        envs = _make_task_env(config, bddl_file, seed=config.optimization.seed + 999)
        task_success = []

        for _ in range(num_episodes_per_task // num_envs):
            obs_raw, _ = envs.reset()
            ep_reward = np.zeros(num_envs)
            t = 0

            while t < config.task.max_episode_steps:
                obs_norm = dataset.normalizer["obs"]["state"].normalize(
                    obs_raw.astype(np.float32)
                )
                obs_tensor = torch.tensor(obs_norm, device=device, dtype=torch.float32)

                with torch.no_grad():
                    base_actions_norm = flow_intent.sample(obs_tensor, use_ema=True)
                base_chunk = base_actions_norm[:, start_idx:end_idx, :]

                obs_flat = flatten_obs(obs_norm)
                all_a_exec = []
                for ei in range(num_envs):
                    base_flat = base_chunk[ei].reshape(-1).cpu().numpy()
                    a_exec, _, _ = agent.sample_action(
                        obs_flat[ei], base_flat, deterministic=True
                    )
                    all_a_exec.append(a_exec)

                a_exec_flat = np.stack(all_a_exec)
                act_norm = a_exec_flat.reshape(num_envs, act_steps, config.task.act_dim)
                act = dataset.normalizer["action"].unnormalize(act_norm)

                obs_raw, reward, terminated, truncated, info = envs.step(act)
                ep_reward += reward
                t += act_steps

            success = [1.0 if r > 0 else 0.0 for r in ep_reward]
            task_success.extend(success)

        envs.close()
        mean_s = np.mean(task_success)
        per_task_success[task_name] = mean_s
        all_success.extend(task_success)
        loguru.logger.info(f"  [{task_name}] success={mean_s:.3f}")

    agent.train()

    metrics = {
        "eval/mean_success": np.mean(all_success),
        "eval/mean_reward": np.mean(all_success),  # binary reward
        "eval/num_tasks": len(tasks),
    }
    for task_name, s in per_task_success.items():
        metrics[f"eval/task/{task_name}"] = s
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs/", config_name="residual_parl")
def main(raw_cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)

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
    parl_cfg_dict = cfg_dict.get("residual_parl", {})

    device = config.optimization.device
    set_seed(config.optimization.seed)
    loguru.logger.info(f"Config:\n{OmegaConf.to_yaml(raw_cfg)}")

    # ---- Dataset (multi-task: loads all datasets for normalizers) ----------
    from mip.datasets.libero_dataset import make_dataset
    dataset = make_dataset(config.task, mode="train")
    loguru.logger.info(f"Dataset loaded ({len(dataset)} samples)")

    # ---- Task list ---------------------------------------------------------
    tasks = _get_task_list(config)
    loguru.logger.info(f"Suite has {len(tasks)} tasks: {[t[0] for t in tasks]}")

    # ---- Frozen flow-intent ------------------------------------------------
    flow_intent_ckpt = parl_cfg_dict["flow_intent_ckpt"]
    assert flow_intent_ckpt, "residual_parl.flow_intent_ckpt must be set"

    flow_intent = FlowIntentAgent(config)
    flow_intent.load(flow_intent_ckpt)
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

    # ---- PARL agent --------------------------------------------------------
    obs_dim_per_step = dataset[0]["obs"]["state"].shape[-1]
    obs_dim_flat = config.task.obs_steps * obs_dim_per_step
    act_dim = config.task.act_dim
    act_steps = config.task.act_steps

    parl_config = PARLConfig(
        actor_lr=parl_cfg_dict.get("actor_lr", 3e-4),
        critic_lr=parl_cfg_dict.get("critic_lr", 3e-4),
        hidden_dims=tuple(parl_cfg_dict.get("hidden_dims", [256, 256, 256])),
        discount=parl_cfg_dict.get("discount", 0.99),
        tau=parl_cfg_dict.get("tau", 0.005),
        residual_alpha=parl_cfg_dict.get("residual_alpha", 1.0),
        action_magnitude=parl_cfg_dict.get("action_magnitude", 0.1),
        predict_a_exec=parl_cfg_dict.get("predict_a_exec", False),
        device=device,
        parl_num_samples=parl_cfg_dict.get("parl_num_samples", 16),
        parl_num_elites=parl_cfg_dict.get("parl_num_elites", 4),
        parl_num_grad_steps=parl_cfg_dict.get("parl_num_grad_steps", 5),
        parl_step_size=parl_cfg_dict.get("parl_step_size", 0.01),
        num_critic_updates=parl_cfg_dict.get("num_critic_updates", 2),
        num_actor_updates=parl_cfg_dict.get("num_actor_updates", 4),
        critic_reduction=parl_cfg_dict.get("critic_reduction", "min"),
    )

    agent = ResidualPARLAgent(
        config=parl_config,
        obs_dim=obs_dim_flat,
        act_dim=act_dim,
        query_freq=act_steps,
    )
    agent.set_flow_intent(flow_intent)
    loguru.logger.info(
        f"ResidualPARLAgent created: obs_dim={obs_dim_flat}, act_dim={act_dim}, "
        f"query_freq={act_steps}, alpha={parl_config.residual_alpha}, "
        f"action_mag={parl_config.action_magnitude}"
    )

    # ---- Replay buffer (shared across tasks) -------------------------------
    buffer_capacity = parl_cfg_dict.get("buffer_capacity", 1_000_000)
    buffer = ReplayBuffer(
        capacity=buffer_capacity,
        obs_dim=obs_dim_flat,
        base_action_shape=(act_steps, act_dim),
        delta_dim=act_steps * act_dim,
        device=device,
    )

    # ---- Logger ------------------------------------------------------------
    logger = Logger(config)

    # ---- Training config ---------------------------------------------------
    total_env_steps = parl_cfg_dict.get("total_env_steps", 1_000_000)
    warmup_steps = parl_cfg_dict.get("warmup_steps", 5000)
    batch_size = parl_cfg_dict.get("batch_size", 256)
    num_updates = parl_cfg_dict.get("num_updates_per_step", 1)
    eval_freq = config.log.eval_freq
    log_freq = config.log.log_freq
    save_freq = config.log.save_freq
    eval_episodes = config.log.eval_episodes
    num_envs = config.task.num_envs
    start_idx = config.task.obs_steps - 1
    end_idx = start_idx + act_steps

    save_dir = Path(config.log.log_dir) / "residual_parl_checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop: cycle tasks each episode ---------------------------
    global_step = 0
    episode_count = 0
    best_success = -1.0

    loguru.logger.info(f"Starting multi-task training for {total_env_steps} env steps")

    # Pick initial task
    rng = np.random.RandomState(config.optimization.seed)
    current_task_idx = rng.randint(len(tasks))
    current_task_name, current_bddl = tasks[current_task_idx]
    envs = _make_task_env(config, current_bddl, seed=config.optimization.seed)
    loguru.logger.info(f"Initial task: {current_task_name}")

    obs_raw, _ = envs.reset()
    obs_norm = dataset.normalizer["obs"]["state"].normalize(obs_raw.astype(np.float32))
    obs_tensor = torch.tensor(obs_norm, device=device, dtype=torch.float32)
    with torch.no_grad():
        base_actions_norm = flow_intent.sample(obs_tensor, use_ema=True)
    base_chunk_norm = base_actions_norm[:, start_idx:end_idx, :].cpu().numpy()

    ep_reward = np.zeros(num_envs)
    ep_len = np.zeros(num_envs, dtype=int)
    # Track which envs need task switch on next reset
    env_done_count = np.zeros(num_envs, dtype=int)

    while global_step < total_env_steps:
        # --- Action selection ---
        obs_flat = flatten_obs(obs_norm)

        all_a_exec = []
        all_delta = []
        for ei in range(num_envs):
            base_flat = base_chunk_norm[ei].reshape(-1)
            if global_step < warmup_steps:
                mag = parl_config.action_magnitude
                delta = rng.normal(0, mag * 0.5, size=(act_steps * act_dim,)).astype(np.float32).clip(-mag, mag)
                a_exec = (base_flat + parl_config.residual_alpha * delta).clip(-1, 1)
                all_a_exec.append(a_exec)
                all_delta.append(delta)
            else:
                a_exec, delta, _lp = agent.sample_action(obs_flat[ei], base_flat)
                all_a_exec.append(a_exec)
                all_delta.append(delta)

        a_exec_flat = np.stack(all_a_exec)
        delta_flat = np.stack(all_delta)

        # --- Env step ---
        act_norm = a_exec_flat.reshape(num_envs, act_steps, act_dim)
        act = dataset.normalizer["action"].unnormalize(act_norm)

        next_obs_raw, reward, terminated, truncated, info = envs.step(act)
        done = terminated | truncated
        ep_reward += reward
        ep_len += act_steps
        global_step += act_steps * num_envs

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
        for ei in range(num_envs):
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
        any_done = False
        for ei in range(num_envs):
            if done[ei]:
                episode_count += 1
                env_done_count[ei] += 1
                if global_step >= warmup_steps and global_step % log_freq < act_steps * num_envs:
                    logger.log({
                        "step": global_step,
                        "train/ep_reward": ep_reward[ei],
                        "train/ep_length": int(ep_len[ei]),
                        "train/episode": episode_count,
                    }, category="train")
                ep_reward[ei] = 0.0
                ep_len[ei] = 0
                any_done = True

        # Switch task when all envs in the current batch have finished at least one episode
        if any_done and env_done_count.min() >= 1:
            envs.close()
            current_task_idx = rng.randint(len(tasks))
            current_task_name, current_bddl = tasks[current_task_idx]
            envs = _make_task_env(config, current_bddl, seed=config.optimization.seed + global_step)

            next_obs_raw, _ = envs.reset()
            next_obs_norm = dataset.normalizer["obs"]["state"].normalize(
                next_obs_raw.astype(np.float32)
            )
            next_obs_tensor = torch.tensor(next_obs_norm, device=device, dtype=torch.float32)
            with torch.no_grad():
                next_base_norm = flow_intent.sample(next_obs_tensor, use_ema=True)
            next_base_chunk = next_base_norm[:, start_idx:end_idx, :].cpu().numpy()
            ep_reward[:] = 0.0
            ep_len[:] = 0
            env_done_count[:] = 0

        # Advance state
        obs_raw = next_obs_raw
        obs_norm = next_obs_norm
        obs_tensor = next_obs_tensor
        base_chunk_norm = next_base_chunk

        # --- PARL updates ---
        if buffer.size >= warmup_steps:
            update_info = {}
            for _ in range(num_updates):
                batch = buffer.sample(batch_size)
                info_step = agent.update(batch)
                for k, v in info_step.items():
                    update_info[k] = update_info.get(k, 0.0) + v / num_updates

            if global_step % log_freq < act_steps * num_envs:
                update_info["step"] = global_step
                update_info["train/buffer_size"] = buffer.size
                update_info["train/global_step"] = global_step
                logger.log(update_info, category="train")

        # --- Evaluation (all tasks) ---
        if global_step % eval_freq < act_steps * num_envs:
            eval_metrics = evaluate_multitask(
                agent, tasks, dataset, config, flow_intent,
                num_episodes_per_task=eval_episodes,
            )
            eval_metrics["step"] = global_step
            logger.log(eval_metrics, category="eval")
            loguru.logger.info(
                f"Step {global_step}: suite_success={eval_metrics['eval/mean_success']:.3f}"
            )

            if eval_metrics["eval/mean_success"] > best_success:
                best_success = eval_metrics["eval/mean_success"]
                agent.save(str(save_dir / "best.pt"))
                loguru.logger.info(f"New best suite success: {best_success:.3f}")

            # Re-init env after eval (current env was not closed)
            envs.close()
            current_task_idx = rng.randint(len(tasks))
            current_task_name, current_bddl = tasks[current_task_idx]
            envs = _make_task_env(config, current_bddl, seed=config.optimization.seed + global_step)
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
            env_done_count[:] = 0

        # --- Periodic save ---
        if global_step % save_freq < act_steps * num_envs:
            agent.save(str(save_dir / f"step_{global_step}.pt"))

    envs.close()
    agent.save(str(save_dir / "final.pt"))
    loguru.logger.info(f"Training complete. Best suite success: {best_success:.3f}")


if __name__ == "__main__":
    main()
