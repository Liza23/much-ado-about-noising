"""Training pipeline for LIBERO dataset.

Usage:
    python examples/train_libero.py task=libero_spatial_state \
        task.dataset_path=~/datasets/libero_spatial/pick_up_the_black_bowl_on_the_plate_demo.hdf5 \
        task.bddl_file=~/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_on_the_plate.bddl
"""

import os
import time

import hydra
import loguru
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.agent import TrainingAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.libero_dataset import make_dataset
from mip.envs.libero import make_vec_env
from mip.logger import Logger, compute_average_metrics, update_best_metrics
from mip.samplers import get_default_step_list
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import set_seed

torch.set_float32_matmul_precision("high")


def train(config: Config, envs, dataset, agent, logger, resume_state=None):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    loop_loader = loop_dataloader(dataloader)

    lr_scheduler = CosineAnnealingLR(
        agent.optimizer, T_max=config.optimization.gradient_steps
    )
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=config.optimization.gradient_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        rampup_ratio=config.optimization.rampup_ratio,
        min_value=config.optimization.min_value,
        max_value=config.optimization.max_value,
    )

    start_step = 0
    best_metrics = {}
    eval_history = []
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        loguru.logger.info(f"Resuming from step {start_step}")
        for _ in range(start_step):
            lr_scheduler.step()

    info_list = []
    start_time = time.time()

    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        batch = next(loop_loader)

        obs = batch["obs"]["state"].to(config.optimization.device)
        obs = obs[:, : config.task.obs_steps, :]  # (B, obs_steps, obs_dim)
        act = batch["action"].to(config.optimization.device)
        act = act[:, : config.task.horizon, :]  # (B, horizon, act_dim)

        delta_t_scalar = warmup_scheduler(n_gradient_step)
        delta_t = torch.full(
            (act.shape[0],), delta_t_scalar, device=config.optimization.device
        )

        info = agent.update(act, obs, delta_t)
        lr_scheduler.step()

        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.item()
        info_list.append(info)

        if (n_gradient_step + 1) % config.log.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([d[key] for d in info_list])
                except (KeyError, TypeError, ValueError):
                    metrics[key] = np.nan
            logger.log(metrics, category="train")
            info_list = []

        if (n_gradient_step + 1) % config.log.save_freq == 0:
            loguru.logger.info("Saving latest checkpoint...")
            logger.save_agent(agent=agent, identifier="latest")

        if (n_gradient_step + 1) % config.log.eval_freq == 0:
            loguru.logger.info("Evaluating...")
            agent.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics.update(evaluate(config, envs, dataset, agent, logger, num_steps))

            old_best = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            eval_history.append(metrics.copy())
            avg_metrics = compute_average_metrics(eval_history)

            primary_key = f"mean_success_{num_steps_list[0]}"
            if primary_key in metrics:
                is_new_best = (
                    primary_key not in old_best
                    or metrics[primary_key] > old_best[primary_key]
                )
                if is_new_best:
                    success_rate = metrics[primary_key]
                    loguru.logger.info(f"New best! {primary_key} = {success_rate:.4f}")
                    logger.save_agent(agent=agent, identifier="best")
                    ckpt_name = (
                        f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
                        f"{config.optimization.loss_type}_{config.network.network_type}_"
                        f"{config.network.emb_dim}_seed{config.optimization.seed}"
                    )
                    training_state = {
                        "n_gradient_step": n_gradient_step,
                        "best_metrics": best_metrics,
                        "eval_history": eval_history,
                    }
                    logger.save_global_checkpoint(
                        agent, ckpt_name, success_rate, training_state=training_state
                    )

            for key, value in best_metrics.items():
                metrics[f"best_{key}"] = value
            for key, value in avg_metrics.items():
                metrics[key] = value
            loguru.logger.info("Best metrics so far:")
            for key, value in best_metrics.items():
                loguru.logger.info(f"  {key}: {value:.4f}")

            logger.log(metrics, category="eval")
            agent.train()


def evaluate(config: Config, envs, dataset, agent, logger, num_steps: int = 1):
    episode_rewards = []
    episode_steps = []
    episode_success = []

    for _ in range(config.log.eval_episodes // config.task.num_envs):
        ep_reward = [0.0] * config.task.num_envs
        obs, _ = envs.reset()
        t = 0

        while t < config.task.max_episode_steps:
            obs = obs.astype(np.float32)
            obs = dataset.normalizer["obs"]["state"].normalize(obs)
            obs_tensor = torch.tensor(
                obs, device=config.optimization.device, dtype=torch.float32
            )

            act_0 = torch.randn(
                (config.task.num_envs, config.task.horizon, config.task.act_dim),
                device=config.optimization.device,
            )
            with torch.no_grad():
                act_normed = agent.sample(
                    act_0=act_0,
                    obs={"state": obs_tensor},
                    num_steps=num_steps,
                    use_ema=True,
                )

            act_normed = act_normed.detach().cpu().numpy()
            act = dataset.normalizer["action"].unnormalize(act_normed)

            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]

            obs, reward, terminated, truncated, info = envs.step(act)
            ep_reward = [ep_reward[i] + reward[i] for i in range(config.task.num_envs)]
            t += config.task.act_steps

        # success: check info["success"] or "final_info"
        success = [0] * config.task.num_envs
        if "success" in info:
            for i in range(config.task.num_envs):
                success[i] = int(info["success"][i]) if hasattr(info["success"], "__len__") else int(info["success"])
        elif "_final_info" in info:
            for i in range(config.task.num_envs):
                if info["_final_info"][i]:
                    fi = info["final_info"][i]
                    if fi and "success" in fi:
                        success[i] = int(fi["success"])

        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    mean_success = float(np.nanmean(episode_success))
    loguru.logger.info(
        f"Nstep={num_steps} | mean_reward={np.nanmean(episode_rewards):.3f} | "
        f"mean_success={mean_success:.3f}"
    )
    return {
        f"mean_step_{num_steps}": float(np.nanmean(episode_steps)),
        f"mean_reward_{num_steps}": float(np.nanmean(episode_rewards)),
        f"mean_success_{num_steps}": mean_success,
    }


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    set_seed(config.optimization.seed)
    logger = Logger(config)
    loguru.logger.info("Logger ready")

    config.task.save_video = config.log.save_video
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    # Set obs_dim from actual env observation shape
    config.task.obs_dim = obs.shape[-1]
    loguru.logger.info(f"obs_dim resolved to {config.task.obs_dim}")

    dataset = make_dataset(config.task)
    loguru.logger.info(f"Dataset: {dataset}")

    agent = TrainingAgent(config)

    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    elif config.mode == "train" and config.optimization.auto_resume:
        ckpt_name = (
            f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_seed{config.optimization.seed}"
        )
        ckpt_path = logger.find_latest_checkpoint(ckpt_name)
        if ckpt_path:
            loguru.logger.info(f"Auto-resuming from {ckpt_path}")
            resume_state = agent.load(str(ckpt_path), load_optimizer=True)

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
    elif config.mode == "eval":
        agent.eval()
        num_steps_list = get_default_step_list(config.optimization.loss_type)
        metrics = {"step": 0}
        for num_steps in num_steps_list:
            metrics.update(evaluate(config, envs, dataset, agent, logger, num_steps))
        for key, val in metrics.items():
            if "success" in key:
                loguru.logger.info(f"{key}: {val}")
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    main()
