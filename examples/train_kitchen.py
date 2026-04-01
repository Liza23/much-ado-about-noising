"""Training pipeline for Kitchen dataset.

Author: Chaoyi Pan
Date: 2025-10-17
"""

import time

import hydra
import loguru
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from mip.agent import TrainingAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.kitchen_dataset import make_dataset
from mip.envs.kitchen import make_vec_env
from mip.flow_intent_agent import FlowIntentAgent
from mip.intent_encoder import IntentEncoder
from mip.intent_predictor import IntentPredictor
from mip.logger import Logger, compute_average_metrics, update_best_metrics
from mip.samplers import get_default_step_list
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import set_seed


def train(config: Config, envs, dataset, agent, logger, resume_state=None,
          intent_predictor=None, intent_encoder=None):
    """Standalone training function.

    Args:
        config: Configuration for training
        envs: Environment
        dataset: Training dataset
        agent: Agent to train
        logger: Logger for metrics
        resume_state: Optional dict with training state to resume from
        intent_predictor: Optional IntentPredictor trained jointly with the policy
        intent_encoder: Optional IntentEncoder (only for encoded_mean, Config B).
    """
    arch_variant = (
        getattr(config.task, "arch_variant", None)
        or getattr(config.network, "arch_variant", "flow_action")
    )
    intent_conditioning = getattr(config.task, "intent_conditioning", False)

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4 if config.task.obs_type == "state" else 8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    loop_loader = loop_dataloader(dataloader)

    # LR scheduler: Config A exposes intent_optimizer + action_optimizer;
    # Config B exposes a single optimizer.
    _main_opt = agent.intent_optimizer if arch_variant == "flow_intent" else agent.optimizer
    lr_scheduler = CosineAnnealingLR(
        _main_opt, T_max=config.optimization.gradient_steps,
    )
    action_lr_scheduler = (
        CosineAnnealingLR(agent.action_optimizer, T_max=config.optimization.gradient_steps)
        if arch_variant == "flow_intent" else None
    )

    # Intent predictor optimizer (Config B only)
    intent_pred_optimizer = None
    intent_pred_lr_scheduler = None
    if intent_predictor is not None:
        _pred_params = list(intent_predictor.parameters())
        if intent_encoder is not None:
            _pred_params += list(intent_encoder.parameters())
        intent_pred_optimizer = torch.optim.Adam(
            _pred_params, lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )
        intent_pred_lr_scheduler = CosineAnnealingLR(
            intent_pred_optimizer, T_max=config.optimization.gradient_steps
        )

    # warmup scheduler (mainly for flow map learning)
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=config.optimization.gradient_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        rampup_ratio=config.optimization.rampup_ratio,
        min_value=config.optimization.min_value,
        max_value=config.optimization.max_value,
    )

    # Resume from checkpoint if available
    start_step = 0
    best_metrics = {}
    eval_history = []
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        loguru.logger.info(f"Resuming training from step {start_step}")
        loguru.logger.info(f"Restored best metrics: {best_metrics}")

        for _ in range(start_step):
            lr_scheduler.step()
            if intent_pred_lr_scheduler is not None:
                intent_pred_lr_scheduler.step()
            if action_lr_scheduler is not None:
                action_lr_scheduler.step()

    # Encoder warmup: freeze IntentEncoder after intent_encoder_warmup_steps.
    _enc_warmup_steps = getattr(config.task, "intent_encoder_warmup_steps", 0)
    _encoder_frozen = False

    info_list = []
    start_time = time.time()
    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        # Freeze IntentEncoder after warmup (only for encoded_mean + warmup_steps > 0).
        if (intent_encoder is not None
                and _enc_warmup_steps > 0
                and not _encoder_frozen
                and n_gradient_step >= _enc_warmup_steps):
            for p in intent_encoder.parameters():
                p.requires_grad = False
            _encoder_frozen = True
            loguru.logger.info(
                f"[encoded_mean] IntentEncoder frozen at step {n_gradient_step}."
            )

        # get batch from dataloader
        batch = next(loop_loader)

        # preprocess data
        if config.task.obs_type == "state":
            obs = batch["obs"]["state"].to(config.optimization.device)
            obs = obs[:, : config.task.obs_steps, :]  # (B, obs_steps, obs_dim)
        else:
            raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

        base_obs = obs  # keep ref before intent is appended

        # Intent conditioning (Config B only — Config A passes intent via agent.update)
        _joint_hl_loss = None
        if (intent_conditioning
                and "intent" in batch
                and arch_variant != "flow_intent"):
            training_mode = getattr(config.task, "intent_training_mode", "independent")

            _raw_intent = batch["intent"].to(config.optimization.device)
            if (getattr(config.task, "intent_type", "mean") == "encoded_mean"
                    and intent_encoder is not None):
                intent_gt_encoded = intent_encoder(_raw_intent).mean(dim=1)
            else:
                intent_gt_encoded = _raw_intent

            if training_mode == "joint" and intent_predictor is not None:
                intent_pred_pre = intent_predictor(base_obs)
                hl_loss_pre = F.mse_loss(intent_pred_pre, intent_gt_encoded)
                intent_pred_optimizer.zero_grad()
                hl_loss_pre.backward()
                intent_pred_optimizer.step()
                intent_pred_lr_scheduler.step()
                intent = intent_pred_pre.detach()
                _joint_hl_loss = hl_loss_pre.item()
            else:
                intent = intent_gt_encoded.detach()

            intent_expanded = intent.unsqueeze(1).expand(
                -1, config.task.obs_steps, -1
            )
            obs = torch.cat([obs, intent_expanded], dim=-1)

        act = batch["action"].to(config.optimization.device)
        act = act[:, : config.task.horizon, :]  # (B, horizon, act_dim)

        # update diffusion
        delta_t_scalar = warmup_scheduler(n_gradient_step)
        batch_size = act.shape[0]
        delta_t = torch.full(
            (batch_size,), delta_t_scalar, device=config.optimization.device
        )
        if arch_variant == "flow_intent":
            _intent_gt = batch["intent"].to(config.optimization.device)
            info = agent.update(act, base_obs, delta_t, intent_gt=_intent_gt)
            lr_scheduler.step()
            action_lr_scheduler.step()
        else:
            info = agent.update(act, obs, delta_t)
            lr_scheduler.step()

        # Update HL predictor (independent mode — joint mode already updated above).
        if intent_predictor is not None and "intent" in batch:
            _tm = getattr(config.task, "intent_training_mode", "independent")
            if _tm == "joint":
                info["intent_pred_mse"] = _joint_hl_loss
            else:
                _raw_intent_ind = batch["intent"].to(config.optimization.device)
                if (getattr(config.task, "intent_type", "mean") == "encoded_mean"
                        and intent_encoder is not None):
                    intent_gt_ind = intent_encoder(_raw_intent_ind).mean(dim=1)
                else:
                    intent_gt_ind = _raw_intent_ind
                intent_pred = intent_predictor(base_obs)
                intent_pred_loss = F.mse_loss(intent_pred, intent_gt_ind)
                intent_pred_optimizer.zero_grad()
                intent_pred_loss.backward()
                intent_pred_optimizer.step()
                intent_pred_lr_scheduler.step()
                info["intent_pred_mse"] = intent_pred_loss.item()

        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.item()
        info_list.append(info)

        # log metrics
        if (n_gradient_step + 1) % config.log.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([info[key] for info in info_list])
                except (KeyError, TypeError, ValueError):
                    metrics[key] = np.nan
            logger.log(metrics, category="train")
            info_list = []

        if (n_gradient_step + 1) % config.log.save_freq == 0:
            loguru.logger.info("Save model...")
            logger.save_agent(agent=agent, identifier="latest")

        if (n_gradient_step + 1) % config.log.eval_freq == 0:
            loguru.logger.info("Evaluate model...")
            agent.eval()
            if intent_predictor is not None:
                intent_predictor.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics.update(
                    evaluate(config, envs, dataset, agent, logger, num_steps,
                             intent_predictor=intent_predictor,
                             arch_variant=arch_variant)
                )

            # Update best metrics and average metrics
            old_best_metrics = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            eval_history.append(metrics.copy())
            avg_metrics = compute_average_metrics(eval_history)

            # Check if this is a new best model based on p4 success rate for kitchen
            primary_metric_key = f"p4_{num_steps_list[0]}"
            if primary_metric_key in metrics:
                is_new_best = (
                    primary_metric_key not in old_best_metrics
                    or metrics[primary_metric_key]
                    > old_best_metrics[primary_metric_key]
                )
                if is_new_best:
                    success_rate = metrics[primary_metric_key]
                    loguru.logger.info(
                        f"New best model! {primary_metric_key} = {success_rate:.4f}"
                    )
                    logger.save_agent(agent=agent, identifier="best")

                    checkpoint_base_name = (
                        f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
                        f"{config.optimization.loss_type}_{config.network.network_type}_"
                        f"{config.network.emb_dim}_seed{config.optimization.seed}"
                    )
                    if intent_conditioning:
                        checkpoint_base_name += "_intent"
                    if arch_variant == "flow_intent":
                        checkpoint_base_name += "_flow_intent"
                    else:
                        if getattr(config.task, "intent_predictor", False):
                            checkpoint_base_name += "_learned"
                        if getattr(config.task, "intent_training_mode", "independent") == "joint":
                            checkpoint_base_name += "_joint"
                    if getattr(config.task, "intent_type", "mean") == "encoded_mean":
                        checkpoint_base_name += "_emb"
                    training_state = {
                        "n_gradient_step": n_gradient_step,
                        "best_metrics": best_metrics,
                        "eval_history": eval_history,
                        "intent_predictor_state": (
                            intent_predictor.state_dict() if intent_predictor is not None else None
                        ),
                        "intent_encoder_state": (
                            intent_encoder.state_dict() if intent_encoder is not None else None
                        ),
                    }
                    logger.save_global_checkpoint(
                        agent,
                        checkpoint_base_name,
                        success_rate,
                        training_state=training_state,
                    )

            # Add best and average metrics to current metrics for logging
            for key, value in best_metrics.items():
                metrics[f"best_{key}"] = value
            for key, value in avg_metrics.items():
                metrics[key] = value

            # Print best and average metrics
            loguru.logger.info("Best metrics so far:")
            for key, value in best_metrics.items():
                loguru.logger.info(f"  {key}: {value:.4f}")
            if avg_metrics:
                loguru.logger.info("Average metrics (last 5 evals):")
                for key, value in avg_metrics.items():
                    loguru.logger.info(f"  {key}: {value:.4f}")

            logger.log(metrics, category="eval")
            agent.train()
            if intent_predictor is not None:
                intent_predictor.train()
            if intent_encoder is not None:
                intent_encoder.train()


def evaluate(config: Config, envs, dataset, agent, logger, num_steps=1,
             intent_predictor=None, arch_variant="flow_action"):
    """Standalone inference function to evaluate a trained agent and optionally save a video.

    Args:
        config: Configuration object containing evaluation parameters
        envs: Environment
        dataset: Dataset
        agent: Trained agent
        logger: Logger for metrics
        num_steps: Number of steps for sampling
        intent_predictor: Optional IntentPredictor for eval-time intent conditioning
        arch_variant: "flow_intent" (Config A) or "flow_action" (Config B / baseline)

    Returns:
        dict: Metrics including mean step, reward, success rate, and kitchen-specific metrics
    """
    intent_conditioning = getattr(config.task, "intent_conditioning", False)

    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_kit_success = []

    for i in range(config.log.eval_episodes // config.task.num_envs):
        ep_reward = [0.0] * config.task.num_envs
        obs, _ = envs.reset()
        t = 0

        # Track task completions for each environment (Gymnasium-Robotics returns this in info)
        max_tasks_completed = [0] * config.task.num_envs

        # initialize video stream
        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < config.task.max_episode_steps:
            if config.task.obs_type == "state":
                obs = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                obs = dataset.normalizer["obs"]["state"].normalize(obs)
                obs_tensor = torch.tensor(
                    obs, device=config.optimization.device, dtype=torch.float32
                )  # (num_envs, obs_steps, obs_dim)

                # Intent conditioning at eval
                if intent_conditioning and arch_variant != "flow_intent":
                    if intent_predictor is not None:
                        with torch.no_grad():
                            intent_proxy = intent_predictor(obs_tensor)
                    elif getattr(config.task, "intent_type", "mean") == "encoded_mean":
                        raise RuntimeError(
                            "intent_type=encoded_mean requires intent_predictor=true. "
                            "No valid CV proxy in embedding space."
                        )
                    else:
                        raise RuntimeError(
                            "Kitchen eval requires intent_predictor for Config B. "
                            "eef is not part of obs — no constant-velocity proxy available."
                        )
                    intent_expanded = intent_proxy.unsqueeze(1).expand(
                        -1, config.task.obs_steps, -1
                    )
                    obs_tensor = torch.cat([obs_tensor, intent_expanded], dim=-1)

                obs_for_sample = obs_tensor
            else:
                raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

            if arch_variant == "flow_intent":
                act_normed = agent.sample(
                    obs=obs_for_sample,
                    use_ema=True,
                    num_steps=num_steps,
                )
            else:
                act_0 = torch.randn(
                    (config.task.num_envs, config.task.horizon, config.task.act_dim),
                    device=config.optimization.device,
                )
                act_normed = agent.sample(
                    act_0=act_0,
                    obs=obs_for_sample,
                    num_steps=num_steps,
                    use_ema=True,
                )

            # unnormalize prediction
            act_normed = (
                act_normed.detach().to("cpu").numpy()
            )  # (num_envs, horizon, action_dim)
            act = dataset.normalizer["action"].unnormalize(act_normed)

            # get action by slicing from start to end
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]

            obs, reward, terminated, truncated, info = envs.step(act)
            _ = terminated | truncated  # Track done status
            ep_reward += reward
            t += config.task.act_steps

            # Update task completion counts from info
            # SyncVectorEnv returns info as dict-of-arrays: info["key"][env_idx]
            if "completed_tasks" in info:
                for env_idx in range(config.task.num_envs):
                    completed = info["completed_tasks"][env_idx]
                    # completed is a list of sets (one per obs step); take the last
                    last = completed[-1] if isinstance(completed, (list, np.ndarray)) else completed
                    max_tasks_completed[env_idx] = max(
                        max_tasks_completed[env_idx], len(last)
                    )
            elif "_final_info" in info:
                for env_idx in range(config.task.num_envs):
                    if info["_final_info"][env_idx]:
                        final_info = info["final_info"][env_idx]
                        if final_info and "completed_tasks" in final_info:
                            completed = final_info["completed_tasks"]
                            last = completed[-1] if isinstance(completed, (list, np.ndarray)) else completed
                            max_tasks_completed[env_idx] = max(
                                max_tasks_completed[env_idx], len(last)
                            )

        # Kitchen-specific: compute task completion metrics
        kit_success = []
        for num in max_tasks_completed:
            sublist = [1 if i < num else 0 for i in range(7)]
            kit_success.append(sublist)
        success = [1 if num >= 4 else 0 for num in max_tasks_completed]

        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)
        episode_kit_success.append(kit_success)

    loguru.logger.info(
        f"Nstep: {num_steps} Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}"
    )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    # Add kitchen-specific metrics (p1-p7: percentage of episodes completing 1-7 tasks)
    mean_kit_success = np.mean(np.array(episode_kit_success), axis=(0, 1))
    kit_metrics = {}
    for i in range(7):
        kit_metrics[f"p{i + 1}_{num_steps}"] = mean_kit_success[i]
    metrics.update(kit_metrics)
    loguru.logger.info(f"Kit metrics: {kit_metrics}")

    return metrics


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Main pipeline function that calls the appropriate standalone function based on mode."""
    # general config setup
    set_seed(config.optimization.seed)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    # env setup
    config.task.save_video = config.log.save_video
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    _ = envs.reset()
    loguru.logger.info("Finished setting up env")

    # Resolve arch_variant (Config A puts it on task yaml as convenience alias)
    arch_variant = (
        getattr(config.task, "arch_variant", None)
        or getattr(config.network, "arch_variant", "flow_action")
    )
    intent_conditioning = getattr(config.task, "intent_conditioning", False)
    base_obs_dim = config.task.obs_dim  # 60 for kitchen state

    # Intent conditioning: bump obs_dim for Config B (intent appended to obs).
    # Config A (flow_intent) keeps base obs_dim — intent handled by separate flow model.
    if intent_conditioning:
        _intent_type = getattr(config.task, "intent_type", "mean")
        if arch_variant != "flow_intent":
            _cond_dim = (
                getattr(config.task, "intent_emb_dim", 64)
                if _intent_type == "encoded_mean"
                else config.task.intent_dim
            )
            config.task.obs_dim = base_obs_dim + _cond_dim
            loguru.logger.info(
                f"Intent conditioning [Config B]: obs_dim {base_obs_dim} -> {config.task.obs_dim} "
                f"(+{_cond_dim} intent dims, type={_intent_type})"
            )
        else:
            loguru.logger.info(
                f"Intent conditioning [Config A]: obs_dim stays {base_obs_dim} "
                f"(intent handled by flow model, not appended to obs)"
            )

    # dataset setup
    dataset = make_dataset(config.task)
    loguru.logger.info("Finished setting up dataset")

    # Agent creation: Config A uses FlowIntentAgent, Config B / baseline uses TrainingAgent
    if arch_variant == "flow_intent":
        agent = FlowIntentAgent(config)
        loguru.logger.info("[main] Using Config-A agent: FlowIntentAgent")
    else:
        agent = TrainingAgent(config)

    # Instantiate learned intent predictor and encoder (Config B only).
    # Config A handles intent via the flow intent model inside FlowIntentAgent.
    intent_predictor = None
    intent_encoder = None
    if (intent_conditioning
            and getattr(config.task, "intent_predictor", False)
            and arch_variant != "flow_intent"):
        _intent_type_inst = getattr(config.task, "intent_type", "mean")
        _pred_out_dim = (
            getattr(config.task, "intent_emb_dim", 64)
            if _intent_type_inst == "encoded_mean"
            else config.task.intent_dim
        )
        intent_predictor = IntentPredictor(
            obs_steps=config.task.obs_steps,
            base_obs_dim=base_obs_dim,
            intent_dim=_pred_out_dim,
        ).to(config.optimization.device)
        n_params = sum(p.numel() for p in intent_predictor.parameters())
        loguru.logger.info(
            f"IntentPredictor: {n_params/1e3:.1f}K params "
            f"(in={config.task.obs_steps * base_obs_dim}, out={_pred_out_dim})"
        )

        if _intent_type_inst == "encoded_mean":
            _intent_emb_dim = getattr(config.task, "intent_emb_dim", 64)
            intent_encoder = IntentEncoder(
                raw_intent_dim=config.task.intent_dim,
                intent_emb_dim=_intent_emb_dim,
            ).to(config.optimization.device)
            enc_params = sum(p.numel() for p in intent_encoder.parameters())
            loguru.logger.info(
                f"IntentEncoder (encoded_mean): {enc_params/1e3:.1f}K params "
                f"(in={config.task.intent_dim}, out={_intent_emb_dim})"
            )

    resume_state = None

    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    elif config.mode == "train" and config.optimization.auto_resume:
        checkpoint_base_name = (
            f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_seed{config.optimization.seed}"
        )
        if intent_conditioning:
            checkpoint_base_name += "_intent"
        if arch_variant == "flow_intent":
            checkpoint_base_name += "_flow_intent"
        else:
            if getattr(config.task, "intent_predictor", False):
                checkpoint_base_name += "_learned"
            if getattr(config.task, "intent_training_mode", "independent") == "joint":
                checkpoint_base_name += "_joint"
        if getattr(config.task, "intent_type", "mean") == "encoded_mean":
            checkpoint_base_name += "_emb"
        checkpoint_path = logger.find_latest_checkpoint(checkpoint_base_name)
        if checkpoint_path:
            loguru.logger.info(f"Found checkpoint to resume from: {checkpoint_path}")
            loguru.logger.info("Loading checkpoint with optimizer state...")
            resume_state = agent.load(str(checkpoint_path), load_optimizer=True)
        else:
            loguru.logger.info("No checkpoint found, starting training from scratch")
    elif config.mode == "train" and not config.optimization.auto_resume:
        loguru.logger.info("Auto-resume disabled, starting training from scratch")

    # Restore intent predictor and encoder weights from checkpoint
    if intent_predictor is not None and resume_state is not None:
        pred_state = resume_state.get("intent_predictor_state")
        if pred_state is not None:
            intent_predictor.load_state_dict(pred_state)
            loguru.logger.info("Restored IntentPredictor weights from checkpoint")
    if intent_encoder is not None and resume_state is not None:
        enc_state = resume_state.get("intent_encoder_state")
        if enc_state is not None:
            intent_encoder.load_state_dict(enc_state)
            loguru.logger.info("Restored IntentEncoder weights from checkpoint")

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state,
              intent_predictor=intent_predictor, intent_encoder=intent_encoder)
    elif config.mode == "eval":
        agent.eval()
        if intent_predictor is not None:
            intent_predictor.eval()
        if intent_encoder is not None:
            intent_encoder.eval()

        num_steps_list = get_default_step_list(config.optimization.loss_type)
        for num_steps in num_steps_list:
            metrics = {"step": num_steps}
            metrics.update(evaluate(config, envs, dataset, agent, logger, num_steps,
                                    intent_predictor=intent_predictor,
                                    arch_variant=arch_variant))
            metrics["step"] = int(metrics["step"])
            logger.log(metrics, category="eval")

        # print result in easy to read format
        for key, val in metrics.items():
            if "mean_success" in key or key.startswith("p"):
                loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
