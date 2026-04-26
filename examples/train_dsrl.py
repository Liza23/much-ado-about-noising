"""DSRL-SAC training: learns initial noise x_0 in Gaussian space to steer a frozen flow policy.

Reference: "Steering Your Diffusion Policy with Latent Space Reinforcement Learning" (arXiv 2506.15799)

Architecture:
    flow_policy (frozen) receives x_0 chosen by the DSRL actor instead of random noise.

Supports:
    Exp 1/2 - DSRL on plain flow policy:
        task=lift_mh_state, network=mlp
        noise_dim = Ta * act_dim  (e.g. 10*10=100)
        injects into action ODE: flow_agent.sample(act_0=x0, ...)

    Exp 3   - DSRL on flow-intent policy (FlowIntentAgent):
        task=lift_mh_state_flow_intent, network=mlp_flow_intent
        noise_dim = intent_dim  (7)
        injects into intent ODE: flow_agent.sample_with_intent_noise(obs, x0)

Usage:
    python examples/train_dsrl.py \\
        task=lift_mh_state \\
        network=mlp \\
        dsrl.flow_ckpt=/path/to/baseline/model_best.pt

    python examples/train_dsrl.py \\
        task=lift_mh_state_flow_intent \\
        network=mlp_flow_intent \\
        dsrl.flow_ckpt=/path/to/flow_intent/model_best.pt
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import loguru
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.config import Config
from mip.dsrl.dsrl_agent import DSRLConfig, DSRLSACAgent
from mip.dsrl.replay_buffer import DSRLReplayBuffer
from mip.logger import Logger
from mip.torch_utils import set_seed


def _make_dataset_and_envs(config: Config):
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


def _flatten(obs: np.ndarray) -> np.ndarray:
    """(num_envs, obs_steps, obs_dim) -> (num_envs, obs_steps * obs_dim)."""
    return obs.reshape(obs.shape[0], -1)


def _load_flow_agent(config: Config, ckpt_path: str):
    """Load and freeze the appropriate flow agent based on network arch_variant."""
    arch = getattr(config.network, "arch_variant", "flow_action")
    if arch == "flow_intent":
        from mip.flow_intent_agent import FlowIntentAgent
        agent = FlowIntentAgent(config)
        agent.load(ckpt_path)
        for mod in [agent.encoder, agent.intent_flow_map]:
            for p in mod.parameters():
                p.requires_grad_(False)
        if agent.action_decoder is not None:
            for p in agent.action_decoder.parameters():
                p.requires_grad_(False)
        if agent.action_flow_map is not None:
            for p in agent.action_flow_map.parameters():
                p.requires_grad_(False)
        is_flow_intent = True
    else:
        from mip.agent import TrainingAgent
        agent = TrainingAgent(config)
        agent.load(ckpt_path)
        for p in agent.flow_map.parameters():
            p.requires_grad_(False)
        for p in agent.encoder.parameters():
            p.requires_grad_(False)
        is_flow_intent = False
    loguru.logger.info(f"Frozen {'FlowIntentAgent' if is_flow_intent else 'TrainingAgent'} loaded from {ckpt_path}")
    return agent, is_flow_intent


def _run_flow(
    flow_agent,
    is_flow_intent: bool,
    obs_norm: np.ndarray,
    noise_batch: np.ndarray,
    config: Config,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen flow agent with injected noise.

    Returns:
        obs_tensor: (num_envs, obs_steps, obs_dim) — for SAC obs_flat computation
        a_exec_norm: (num_envs, Ta, act_dim)
    """
    num_envs = obs_norm.shape[0]
    obs_tensor = torch.tensor(obs_norm, device=device, dtype=torch.float32)

    if is_flow_intent:
        intent_dim = config.task.intent_dim
        intent_noise = torch.tensor(
            noise_batch.reshape(num_envs, 1, intent_dim), device=device, dtype=torch.float32
        )
        a_exec_norm = flow_agent.sample_with_intent_noise(obs_tensor, intent_noise, use_ema=True)
    else:
        Ta = config.task.horizon
        act_dim = config.task.act_dim
        act_0 = torch.tensor(
            noise_batch.reshape(num_envs, Ta, act_dim), device=device, dtype=torch.float32
        )
        with torch.no_grad():
            a_exec_norm = flow_agent.sample(
                act_0=act_0, obs=obs_tensor, use_ema=True, sample_mode="stochastic"
            )

    return obs_tensor, a_exec_norm


def evaluate(
    dsrl: DSRLSACAgent,
    flow_agent,
    is_flow_intent: bool,
    envs,
    dataset,
    config: Config,
    noise_dim: int,
    num_episodes: int = 10,
) -> dict[str, float]:
    dsrl.eval()
    num_envs = config.task.num_envs
    episode_rewards, episode_success = [], []
    act_steps = config.task.act_steps
    device = config.optimization.device

    for _ in range(num_episodes // num_envs):
        obs_raw, _ = envs.reset()
        ep_reward = np.zeros(num_envs)
        t = 0
        while t < config.task.max_episode_steps:
            obs_norm = dataset.normalizer["obs"]["state"].normalize(obs_raw.astype(np.float32))
            obs_flat = _flatten(obs_norm)

            all_noise = []
            for ei in range(num_envs):
                noise, _ = dsrl.sample_noise(obs_flat[ei], deterministic=True)
                all_noise.append(noise)
            noise_batch = np.stack(all_noise)

            _, a_exec_norm = _run_flow(flow_agent, is_flow_intent, obs_norm, noise_batch, config, device)

            start = config.task.obs_steps - 1
            chunk_norm = a_exec_norm[:, start:start + act_steps, :]
            act = dataset.normalizer["action"].unnormalize(chunk_norm.cpu().numpy())
            if config.task.abs_action and config.task.env_name in [
                "can", "lift", "square", "tool_hang", "transport"
            ]:
                act = dataset.undo_transform_action(act)

            obs_raw, reward, terminated, truncated, info = envs.step(act)
            ep_reward += reward
            t += act_steps

        success = [1.0 if r > 0 else 0.0 for r in ep_reward]
        episode_rewards.extend(ep_reward.tolist())
        episode_success.extend(success)

    dsrl.train()
    return {
        "eval/mean_reward": np.mean(episode_rewards),
        "eval/mean_success": np.mean(episode_success),
        "eval/num_episodes": len(episode_success),
    }


@hydra.main(version_base=None, config_path="configs/", config_name="dsrl")
def main(raw_cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)
    config = Config(
        optimization=OmegaConf.structured(raw_cfg.optimization) if "optimization" in raw_cfg else None,
        network=OmegaConf.structured(raw_cfg.network) if "network" in raw_cfg else None,
        task=OmegaConf.structured(raw_cfg.task) if "task" in raw_cfg else None,
        log=OmegaConf.structured(raw_cfg.log) if "log" in raw_cfg else None,
    )
    dsrl_cfg = cfg_dict.get("dsrl", {})
    device = config.optimization.device
    set_seed(config.optimization.seed)
    loguru.logger.info(f"Config:\n{OmegaConf.to_yaml(raw_cfg)}")

    # Resolve interpolated log_dir before Logger is constructed
    config.log.log_dir = cfg_dict["log"]["log_dir"]

    dataset, envs = _make_dataset_and_envs(config)
    loguru.logger.info(f"Dataset loaded ({len(dataset)} samples)")

    sample0 = dataset[0]
    base_obs_dim = sample0["obs"]["state"].shape[-1]
    config.task.obs_dim = base_obs_dim

    # ---- Frozen flow policy -----------------------------------------------
    flow_ckpt = dsrl_cfg.get("flow_ckpt", "")
    assert flow_ckpt, "dsrl.flow_ckpt must be set"
    flow_agent, is_flow_intent = _load_flow_agent(config, flow_ckpt)

    # ---- DSRL dimensions --------------------------------------------------
    # FlowIntentAgent: SAC acts in intent noise space (intent_dim=7, much smaller)
    # TrainingAgent:   SAC acts in action noise space (Ta * act_dim)
    Ta = config.task.horizon
    act_dim = config.task.act_dim
    act_steps = config.task.act_steps
    obs_steps = config.task.obs_steps

    if is_flow_intent:
        noise_dim = config.task.intent_dim  # 7
    else:
        noise_dim = Ta * act_dim            # e.g. 100

    obs_dim_flat = obs_steps * base_obs_dim  # SAC uses raw base obs

    dsrl_config = DSRLConfig(
        actor_lr=dsrl_cfg.get("actor_lr", 3e-4),
        critic_lr=dsrl_cfg.get("critic_lr", 3e-4),
        alpha_lr=dsrl_cfg.get("alpha_lr", 3e-4),
        hidden_dims=tuple(dsrl_cfg.get("hidden_dims", [1024, 1024, 1024])),
        discount=dsrl_cfg.get("discount", 0.99),
        tau=dsrl_cfg.get("tau", 0.005),
        init_alpha=dsrl_cfg.get("init_alpha", 0.1),
        noise_mag=dsrl_cfg.get("noise_mag", 2.0),
        backup_entropy=dsrl_cfg.get("backup_entropy", True),
        q_target_clip=float(dsrl_cfg.get("q_target_clip", float("inf"))),
        device=device,
    )
    target_entropy = dsrl_cfg.get("target_entropy", "auto")
    if target_entropy != "auto":
        dsrl_config.target_entropy = float(target_entropy)

    dsrl = DSRLSACAgent(config=dsrl_config, obs_dim=obs_dim_flat, Ta=1, act_dim=noise_dim)
    dsrl.set_flow_agent(flow_agent)
    loguru.logger.info(
        f"DSRLSACAgent: obs_dim={obs_dim_flat}, noise_dim={noise_dim}, "
        f"is_flow_intent={is_flow_intent}"
    )

    # ---- Replay buffer -----------------------------------------------------
    buffer = DSRLReplayBuffer(
        capacity=dsrl_cfg.get("buffer_capacity", 1_000_000),
        obs_dim=obs_dim_flat,
        noise_dim=noise_dim,
        device=device,
    )

    logger = Logger(config)

    total_env_steps = dsrl_cfg.get("total_env_steps", 1_000_000)
    warmup_steps = dsrl_cfg.get("warmup_steps", 1000)
    batch_size = dsrl_cfg.get("batch_size", 256)
    num_updates = dsrl_cfg.get("num_updates_per_step", 1)
    reward_scale = dsrl_cfg.get("reward_scale", 1.0)
    eval_freq = config.log.eval_freq
    log_freq = config.log.log_freq
    save_freq = config.log.save_freq
    eval_episodes = config.log.eval_episodes

    save_dir = Path(config.log.log_dir) / "dsrl_checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    obs_raw, _ = envs.reset()
    obs_norm = dataset.normalizer["obs"]["state"].normalize(obs_raw.astype(np.float32))
    obs_flat = _flatten(obs_norm)

    global_step = 0
    episode_count = 0
    ep_reward = np.zeros(config.task.num_envs)
    ep_len = np.zeros(config.task.num_envs, dtype=int)
    best_success = -1.0

    loguru.logger.info(f"Starting DSRL training for {total_env_steps} env steps")

    while global_step < total_env_steps:
        # --- Sample noise from actor (or random during warmup) ---
        all_noise = []
        for ei in range(config.task.num_envs):
            if global_step < warmup_steps:
                noise = np.random.uniform(
                    -dsrl_config.noise_mag, dsrl_config.noise_mag, size=(noise_dim,)
                ).astype(np.float32)
            else:
                noise, _ = dsrl.sample_noise(obs_flat[ei])
            all_noise.append(noise)
        noise_batch = np.stack(all_noise)  # (num_envs, noise_dim)

        # --- Run frozen flow policy with injected noise ---
        _, a_exec_norm = _run_flow(flow_agent, is_flow_intent, obs_norm, noise_batch, config, device)

        start = obs_steps - 1
        chunk_norm = a_exec_norm[:, start:start + act_steps, :].cpu().numpy()
        act = dataset.normalizer["action"].unnormalize(chunk_norm)
        if config.task.abs_action and config.task.env_name in [
            "can", "lift", "square", "tool_hang", "transport"
        ]:
            act = dataset.undo_transform_action(act)

        next_obs_raw, reward, terminated, truncated, info = envs.step(act)
        done = terminated | truncated
        ep_reward += reward
        ep_len += act_steps
        global_step += act_steps * config.task.num_envs

        # --- Next obs ---
        next_obs_norm = dataset.normalizer["obs"]["state"].normalize(next_obs_raw.astype(np.float32))
        next_obs_flat = _flatten(next_obs_norm)

        # --- Store transitions ---
        for ei in range(config.task.num_envs):
            buffer.insert(
                obs=obs_flat[ei],
                noise=noise_batch[ei],
                reward=float(reward[ei]) * reward_scale,
                next_obs=next_obs_flat[ei],
                done=bool(done[ei]),
            )

        # --- Episode logging ---
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

        obs_norm = next_obs_norm
        obs_flat = next_obs_flat

        # --- SAC updates ---
        if buffer.size >= warmup_steps:
            update_info = {}
            for _ in range(num_updates):
                batch = buffer.sample(batch_size)
                info_step = dsrl.update(batch)
                for k, v in info_step.items():
                    update_info[k] = update_info.get(k, 0.0) + v / num_updates

            if global_step % log_freq < act_steps * config.task.num_envs:
                update_info["step"] = global_step
                update_info["train/buffer_size"] = buffer.size
                logger.log(update_info, category="train")

        # --- Evaluation ---
        if global_step % eval_freq < act_steps * config.task.num_envs:
            eval_metrics = evaluate(
                dsrl, flow_agent, is_flow_intent, envs, dataset, config,
                noise_dim=noise_dim, num_episodes=eval_episodes,
            )
            eval_metrics["step"] = global_step
            logger.log(eval_metrics, category="eval")
            loguru.logger.info(
                f"Step {global_step}: success={eval_metrics['eval/mean_success']:.2f}, "
                f"reward={eval_metrics['eval/mean_reward']:.2f}"
            )
            if eval_metrics["eval/mean_success"] > best_success:
                best_success = eval_metrics["eval/mean_success"]
                dsrl.save(str(save_dir / "best.pt"))
                loguru.logger.info(f"New best success: {best_success:.2f}")

            obs_raw, _ = envs.reset()
            obs_norm = dataset.normalizer["obs"]["state"].normalize(obs_raw.astype(np.float32))
            obs_flat = _flatten(obs_norm)
            ep_reward[:] = 0.0
            ep_len[:] = 0

        if global_step % save_freq < act_steps * config.task.num_envs:
            dsrl.save(str(save_dir / f"step_{global_step}.pt"))

    dsrl.save(str(save_dir / "final.pt"))
    loguru.logger.info(f"Training complete. Best success: {best_success:.2f}")


if __name__ == "__main__":
    main()
