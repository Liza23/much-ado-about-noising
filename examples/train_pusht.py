"""Training pipeline for PushT dataset.

Author: Chaoyi Pan
Date: 2025-10-15
"""

import os
import time

import hydra
import loguru
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

import torch.nn.functional as F

from mip.agent import TrainingAgent
from mip.config import Config
from mip.flow_intent_agent import FlowIntentAgent  # Config A
from mip.intent_predictor import IntentPredictor
from mip.dataset_utils import loop_dataloader
from mip.datasets.pusht_dataset import make_dataset
from mip.envs.pusht import make_vec_env
from mip.logger import Logger, compute_average_metrics, update_best_metrics
from mip.samplers import get_default_step_list
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import set_seed
from mip.visualizations import visualize_geometric_predictions, visualize_vla_predictions, visualize_vla_debug


# ── DDP helpers ──────────────────────────────────────────────────────────
def setup_ddp():
    """Initialize distributed training if launched with torchrun."""
    if "RANK" not in os.environ:
        # Not launched with torchrun → single GPU
        return 0, 1, False

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size, True


def cleanup_ddp(is_distributed):
    if is_distributed:
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def get_net(agent):
    """Get underlying network, unwrapping DDP if needed."""
    flow_map = agent.flow_map
    if hasattr(flow_map, 'module'):
        flow_map = flow_map.module
    net = flow_map.net
    if hasattr(net, 'module'):
        net = net.module
    return net


def get_ema_net(agent):
    """Get underlying EMA network, unwrapping DDP if needed."""
    flow_map = agent.flow_map_ema
    if hasattr(flow_map, 'module'):
        flow_map = flow_map.module
    net = flow_map.net
    if hasattr(net, 'module'):
        net = net.module
    return net


def train(config: Config, envs, dataset, agent, logger, resume_state=None,
          rank=0, world_size=1, is_distributed=False, intent_predictor=None):
    """Standalone training function.

    Args:
        config: Configuration for training
        envs: Environment
        dataset: Training dataset
        agent: Agent to train
        logger: Logger for metrics
        resume_state: Optional dict with training state to resume from
        rank: Local rank for DDP
        world_size: Total number of processes
        is_distributed: Whether running with DDP
    """
    # dataloader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True) if is_distributed else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4 if config.task.obs_type in ["state", "keypoint"] else 8,
        shuffle=(sampler is None),
        sampler=sampler,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True,
    )
    loop_loader = loop_dataloader(dataloader)

    arch_variant = getattr(config.network, "arch_variant", "flow_action")

    # lr scheduler — Config A uses intent_optimizer; Config B uses optimizer
    _main_opt = agent.intent_optimizer if arch_variant == "flow_intent" else agent.optimizer
    lr_scheduler = CosineAnnealingLR(
        _main_opt, T_max=config.optimization.gradient_steps
    )
    action_lr_scheduler = (
        CosineAnnealingLR(agent.action_optimizer, T_max=config.optimization.gradient_steps)
        if arch_variant == "flow_intent" else None
    )

    # IntentPredictor optimizer + scheduler (Config B joint mode only)
    intent_pred_optimizer = None
    intent_pred_lr_scheduler = None
    if intent_predictor is not None:
        intent_pred_optimizer = torch.optim.AdamW(intent_predictor.parameters(), lr=1e-4)
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

        # Fast-forward the lr_scheduler to the correct step
        for _ in range(start_step):
            lr_scheduler.step()
            if action_lr_scheduler is not None:
                action_lr_scheduler.step()
            if intent_pred_lr_scheduler is not None:
                intent_pred_lr_scheduler.step()

    info_list = []
    start_time = time.time()
    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        # get batch from dataloader
        batch = next(loop_loader)

        # preprocess data
        _joint_hl_loss = None
        if config.task.obs_type == "image":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs[k] = obs_batch[k][:, : config.task.obs_steps, :].to(
                    config.optimization.device
                )
            # Config B: use IntentPredictor (joint) or GT intent (independent)
            if getattr(config.task, "intent_conditioning", False) and arch_variant != "flow_intent":
                training_mode = getattr(config.task, "intent_training_mode", "independent")
                intent_gt = batch["intent"].to(config.optimization.device)  # (B, intent_dim)
                base_obs_img = obs["agent_pos"]  # (B, obs_steps, 2) — predictor input
                if training_mode == "joint" and intent_predictor is not None:
                    intent_pred = intent_predictor(base_obs_img)
                    pred_loss = F.mse_loss(intent_pred, intent_gt)
                    intent_pred_optimizer.zero_grad()
                    pred_loss.backward()
                    intent_pred_optimizer.step()
                    intent_pred_lr_scheduler.step()
                    intent = intent_pred.detach()
                    _joint_hl_loss = pred_loss.item()
                else:
                    intent = intent_gt
                intent_rep = intent.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
                obs["agent_pos"] = torch.cat([obs["agent_pos"], intent_rep], dim=-1)
        elif config.task.obs_type == "state":
            obs = batch["obs"]["state"].to(config.optimization.device)
            obs = obs[:, : config.task.obs_steps, :]  # (B, obs_horizon, obs_dim)
            base_obs = obs  # keep ref before intent appended (predictor input)
            # Config B: use IntentPredictor (joint) or GT intent (independent)
            if getattr(config.task, "intent_conditioning", False) and arch_variant != "flow_intent":
                training_mode = getattr(config.task, "intent_training_mode", "independent")
                intent_gt = batch["intent"].to(config.optimization.device)  # (B, intent_dim)
                if training_mode == "joint" and intent_predictor is not None:
                    intent_pred = intent_predictor(base_obs)
                    pred_loss = F.mse_loss(intent_pred, intent_gt)
                    intent_pred_optimizer.zero_grad()
                    pred_loss.backward()
                    intent_pred_optimizer.step()
                    intent_pred_lr_scheduler.step()
                    intent = intent_pred.detach()
                    _joint_hl_loss = pred_loss.item()
                else:
                    intent = intent_gt
                intent_rep = intent.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
                obs = torch.cat([obs, intent_rep], dim=-1)
        elif config.task.obs_type == "keypoint":
            obs_batch = batch["obs"]
            obs = {}
            for k in obs_batch:
                obs_data = obs_batch[k].to(config.optimization.device)
                obs[k] = obs_data[
                    :, : config.task.obs_steps, :
                ]  # (B, obs_horizon, obs_dim)
        else:
            raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

        act = batch["action"].to(config.optimization.device)
        act = act[:, : config.task.horizon, :]  # (B, horizon, act_dim)

        # Pass raw render data to VLA network (if available) — not applicable to FlowIntentAgent
        render_state = batch.get("render_state", None)
        if render_state is not None and arch_variant != "flow_intent":
            net = get_net(agent)
            if hasattr(net, 'set_render_data'):
                net.set_render_data(
                    render_state.to(config.optimization.device),
                    batch["render_action"].to(config.optimization.device),  # GT for draft loss
                )
            if hasattr(net, 'clear_cache'):
                net.clear_cache()

        # update diffusion
        delta_t_scalar = warmup_scheduler(n_gradient_step)
        batch_size = act.shape[0]
        delta_t = torch.full(
            (batch_size,), delta_t_scalar, device=config.optimization.device
        )
        if arch_variant == "flow_intent":
            intent_gt = batch["intent"].to(config.optimization.device)  # (B, intent_dim)
            info = agent.update(act, obs, delta_t, intent_gt=intent_gt)
        else:
            info = agent.update(act, obs, delta_t)
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.item()
        lr_scheduler.step()
        if action_lr_scheduler is not None:
            action_lr_scheduler.step()

        # Independent-mode predictor update (joint mode already did it above)
        if intent_predictor is not None and "intent" in batch:
            _tm = getattr(config.task, "intent_training_mode", "independent")
            if _tm == "joint":
                info["intent_pred_mse"] = _joint_hl_loss
            else:
                _ig = batch["intent"].to(config.optimization.device)
                _pred_input = base_obs if config.task.obs_type == "state" else obs_batch["agent_pos"][:, :config.task.obs_steps].to(config.optimization.device)
                _ip = intent_predictor(_pred_input)
                _pl = F.mse_loss(_ip, _ig)
                intent_pred_optimizer.zero_grad()
                _pl.backward()
                intent_pred_optimizer.step()
                intent_pred_lr_scheduler.step()
                info["intent_pred_mse"] = _pl.item()

        info_list.append(info)

        # log metrics (rank 0 only)
        if is_main(rank) and (n_gradient_step + 1) % config.log.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([info[key] for info in info_list])
                except (KeyError, TypeError, ValueError) as e:
                    loguru.logger.error(f"Error calculating {key}: {e}")
                    metrics[key] = np.nan

            # Log geometric/VLA features if available — not applicable to FlowIntentAgent
            _train_net = get_net(agent) if arch_variant != "flow_intent" else None
            if _train_net is not None and hasattr(_train_net, '_last_geom_features') and _train_net._last_geom_features is not None:
                geom_feats = _train_net._last_geom_features
                metrics['geom/dist_to_block'] = geom_feats['dist_to_block'].mean().item()
                metrics['geom/will_contact'] = geom_feats['will_contact'].mean().item()
                metrics['geom/penetration_depth'] = geom_feats['penetration_depth'].mean().item()
                metrics['geom/push_quality'] = geom_feats['contact_normal_alignment'].mean().item()
                metrics['geom/num_contacts'] = geom_feats['num_contact_points'].mean().item()

                # Create and log visualizations every 2 log intervals (more frequent)
                if (n_gradient_step + 1) % (config.log.log_freq * 2) == 0:
                    try:
                        import wandb
                        vis_img = visualize_geometric_predictions(
                            obs_batch=_train_net._last_condition,
                            action_pred_batch=_train_net._last_action_pred,
                            geom_features_batch=_train_net._last_geom_features,
                            num_samples=4
                        )
                        metrics['visualizations/geometric_predictions'] = wandb.Image(vis_img)
                        loguru.logger.info(f"Step {n_gradient_step + 1}: Created geometric visualization")
                    except Exception as e:
                        loguru.logger.warning(f"Step {n_gradient_step + 1}: Failed to create visualization: {e}")

            # Log VLA debug visualizations (for vla_crossattn_policy)
            if _train_net is not None and hasattr(_train_net, '_last_attn_weights') and _train_net._last_attn_weights is not None:
                if (n_gradient_step + 1) % (config.log.log_freq * 2) == 0:
                    try:
                        import wandb
                        debug_img = visualize_vla_debug(
                            net=_train_net,
                            sampled_actions=None,
                            gt_actions=None,
                            render_state=getattr(_train_net, '_render_state', None),
                            num_samples=4,
                        )
                        metrics['visualizations/vla_train_debug'] = wandb.Image(debug_img)
                        loguru.logger.info(f"Step {n_gradient_step + 1}: Created VLA debug visualization")
                    except Exception as e:
                        loguru.logger.warning(f"Step {n_gradient_step + 1}: Failed to create VLA debug visualization: {e}")

            logger.log(metrics, category="train")
            info_list = []

        if is_main(rank) and (n_gradient_step + 1) % config.log.save_freq == 0:
            loguru.logger.info("Save model...")
            logger.save_agent(agent=agent, identifier="latest")

            # Create and log visualizations (for geometric_policy, not applicable to FlowIntentAgent)
            _save_net = get_net(agent) if arch_variant != "flow_intent" else None
            if _save_net is not None and hasattr(_save_net, '_last_geom_features') and _save_net._last_geom_features is not None:
                try:
                    import wandb
                    if wandb.run is not None:
                        vis_img = visualize_geometric_predictions(
                            obs_batch=_save_net._last_condition,
                            action_pred_batch=_save_net._last_action_pred,
                            geom_features_batch=_save_net._last_geom_features,
                            num_samples=4
                        )
                        logger.log({"visualizations/geometric_predictions": wandb.Image(vis_img)}, category="train")
                        loguru.logger.info("Logged geometric predictions visualization")
                except Exception as e:
                    loguru.logger.warning(f"Failed to create visualization: {e}")

        if is_main(rank) and (n_gradient_step + 1) % config.log.eval_freq == 0:
            loguru.logger.info("Evaluate model...")
            # Clear training render data before eval (eval loop sets live env state)
            _nets_to_clear = (
                [get_net(agent), get_ema_net(agent)]
                if arch_variant != "flow_intent" else []
            )
            for _net in _nets_to_clear:
                if hasattr(_net, 'set_render_data'):
                    _net.set_render_data(None, None)
                if hasattr(_net, 'clear_cache'):
                    _net.clear_cache()
            agent.eval()
            if intent_predictor is not None:
                intent_predictor.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics.update(
                    evaluate(config, envs, dataset, agent, logger, num_steps,
                             intent_predictor=intent_predictor)
                )

            # Update best metrics and average metrics
            old_best_metrics = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            eval_history.append(metrics.copy())
            avg_metrics = compute_average_metrics(eval_history)

            # Check if this is a new best model based on success rate
            # Use the first num_steps in the list as the primary metric
            primary_metric_key = f"mean_success_{num_steps_list[0]}"
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
                    # Save to local models directory
                    logger.save_agent(agent=agent, identifier="best")

                    # Save to global checkpoints directory with success rate comparison
                    # Include training state for resuming
                    renderer_suffix = ""
                    renderer_type = getattr(config.task, 'renderer_type', 'physics_step')
                    if renderer_type != "physics_step":
                        renderer_suffix = f"_{renderer_type}"
                    if arch_variant == "flow_intent":
                        _variant_suffix = "_flow_intent"
                    elif getattr(config.task, "intent_conditioning", False):
                        _variant_suffix = "_hierarchical_emb"
                        if getattr(config.task, "intent_predictor", False):
                            _variant_suffix += "_learned"
                    else:
                        _variant_suffix = ""
                    checkpoint_base_name = (
                        f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
                        f"{config.optimization.loss_type}_{config.network.network_type}_"
                        f"{config.network.emb_dim}_seed{config.optimization.seed}{renderer_suffix}{_variant_suffix}"
                    )
                    training_state = {
                        "n_gradient_step": n_gradient_step,
                        "best_metrics": best_metrics,
                        "eval_history": eval_history,
                        "intent_predictor_state": (
                            intent_predictor.state_dict() if intent_predictor is not None else None
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


def evaluate(config: Config, envs, dataset, agent, logger, num_steps=1, intent_predictor=None):
    """Standalone inference function to evaluate a trained agent and optionally save a video.

    Args:
        config: Configuration object containing evaluation parameters
        envs: Environment
        dataset: Dataset
        agent: Trained agent
        logger: Logger for metrics
        num_steps: Number of steps for sampling

    Returns:
        dict: Metrics including mean step, reward, and success rate
    """
    arch_variant = getattr(config.network, "arch_variant", "flow_action")

    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []

    for i in range(config.log.eval_episodes // config.task.num_envs):
        step_reward = []
        ep_reward = [0.0] * config.task.num_envs
        # NOTE: update env seed, the original envs is update seed so reset is broken
        for j in range(len(envs.envs)):
            envs.envs[j].seed(config.optimization.seed + i * config.task.num_envs + j)
        obs, _ = envs.reset()
        t = 0
        state_history = None  # Will be [num_envs, To, 5] for temporal state

        # initialize video stream
        if config.log.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < config.task.max_episode_steps:
            _flow_obs = None  # raw obs tensor for Config A (set in state branch)
            if config.task.obs_type == "state":
                obs = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # normalize obs
                obs_norm = dataset.normalizer["obs"]["state"].normalize(obs)
                obs_tensor = torch.tensor(
                    obs_norm, device=config.optimization.device, dtype=torch.float32
                )  # (num_envs, obs_steps, obs_dim)
                # Config A (flow_intent): FlowIntentAgent samples intent internally — obs unchanged
                _flow_obs = obs_tensor  # save raw tensor for flow_intent sampling
                if getattr(config.task, "intent_conditioning", False) and arch_variant != "flow_intent":
                    if intent_predictor is not None:
                        with torch.no_grad():
                            intent_proxy = intent_predictor(obs_tensor)  # (num_envs, intent_dim)
                    else:
                        # CV proxy fallback (no predictor)
                        _intent_idx = dataset.intent_indices
                        pos_now  = obs_norm[:, -1, _intent_idx]
                        pos_prev = obs_norm[:, -2, _intent_idx]
                        velocity = pos_now - pos_prev
                        half_h = (config.task.intent_horizon + 1) / 2.0
                        intent_proxy = torch.tensor(
                            pos_now + half_h * velocity,
                            device=config.optimization.device, dtype=torch.float32
                        )
                    intent_rep = intent_proxy.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
                    obs_tensor = torch.cat([obs_tensor, intent_rep], dim=-1)
                obs = {"state": obs_tensor}
            elif config.task.obs_type == "keypoint":
                obs_raw = obs.astype(np.float32)  # (num_envs, obs_steps, 20)
                # Split into keypoint and agent_pos
                keypoint_obs = obs_raw[:, :, :18]  # (num_envs, obs_steps, 18)
                agent_pos_obs = obs_raw[:, :, 18:20]  # (num_envs, obs_steps, 2)

                # Normalize
                nkeypoint = (
                    dataset.normalizer["obs"]["keypoint"]
                    .normalize(keypoint_obs.reshape(-1, 2))
                    .reshape(config.task.num_envs, config.task.obs_steps, 18)
                )
                nagent_pos = dataset.normalizer["obs"]["agent_pos"].normalize(
                    agent_pos_obs
                )

                obs = {
                    "keypoint": torch.tensor(
                        nkeypoint,
                        device=config.optimization.device,
                        dtype=torch.float32,
                    ),
                    "agent_pos": torch.tensor(
                        nagent_pos,
                        device=config.optimization.device,
                        dtype=torch.float32,
                    ),
                }
            elif config.task.obs_type == "image":
                obs_raw = obs
                obs = {}
                for k in obs_raw:
                    obs[k] = obs_raw[k].astype(
                        np.float32
                    )  # (num_envs, obs_steps, obs_dim)
                    obs[k] = dataset.normalizer["obs"][k].normalize(obs[k])
                    obs[k] = torch.tensor(
                        obs[k], device=config.optimization.device, dtype=torch.float32
                    )  # (num_envs, obs_steps, obs_dim)
                _flow_obs = obs  # Config A: pass full image obs dict to FlowIntentAgent
                # Config B intent: use IntentPredictor or CV proxy fallback
                if getattr(config.task, "intent_conditioning", False) and arch_variant != "flow_intent":
                    if intent_predictor is not None:
                        with torch.no_grad():
                            intent_proxy = intent_predictor(obs["agent_pos"])  # (num_envs, intent_dim)
                    else:
                        agent_pos_np = obs["agent_pos"].cpu().numpy()
                        pos_now  = agent_pos_np[:, -1, :]
                        pos_prev = agent_pos_np[:, -2, :]
                        velocity = pos_now - pos_prev
                        half_h = (config.task.intent_horizon + 1) / 2.0
                        intent_proxy = torch.tensor(
                            pos_now + half_h * velocity,
                            device=config.optimization.device, dtype=torch.float32
                        )
                    intent_rep = intent_proxy.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
                    obs["agent_pos"] = torch.cat([obs["agent_pos"], intent_rep], dim=-1)
            else:
                raise ValueError(f"Invalid obs_type: {config.task.obs_type}")

            # Extract raw pymunk state for VLA eval conditioning.
            # Maintain temporal state history [num_envs, To, 5] for cross-attention.
            # Config A (FlowIntentAgent) has no flow_map_ema — skip this block.
            ema_net = get_ema_net(agent) if arch_variant != "flow_intent" else None
            if ema_net is not None and hasattr(ema_net, 'set_render_data'):
                raw_states = []
                for env_wrapper in envs.envs:
                    base = env_wrapper.env.env  # MultiStep → VideoRecord → PushTImageEnv
                    state_vec = np.array([
                        base.agent.position[0], base.agent.position[1],
                        base.block.position[0], base.block.position[1],
                        base.block.angle,
                    ], dtype=np.float32)
                    raw_states.append(state_vec)
                raw_states_array = np.stack(raw_states)  # [num_envs, 5]
                if state_history is None:
                    # Initialize with repeated first frame
                    state_history = np.stack(
                        [raw_states_array] * config.task.obs_steps, axis=1
                    )  # [num_envs, To, 5]
                else:
                    # Shift left and append new frame
                    state_history = np.roll(state_history, -1, axis=1)
                    state_history[:, -1, :] = raw_states_array
                render_state = torch.tensor(
                    state_history, device=config.optimization.device
                )  # [num_envs, To, 5]
                ema_net.set_render_data(render_state, None)  # No GT during eval
                if hasattr(ema_net, 'clear_cache'):
                    ema_net.clear_cache()

            # run sampling (num_envs, horizon, action_dim)
            if arch_variant == "flow_intent":
                # Config A: FlowIntentAgent samples intent internally; needs raw tensor
                act_normed = agent.sample(
                    obs=_flow_obs,
                    num_steps=num_steps,
                    use_ema=True,
                )
            else:
                act_0 = torch.randn(
                    (config.task.num_envs, config.task.horizon, config.task.act_dim),
                    device=config.optimization.device,
                )
                act_normed = agent.sample(
                    act_0=act_0,
                    obs=obs,
                    num_steps=num_steps,
                    use_ema=True,
                )

            # unnormalize prediction
            act_normed = (
                act_normed.detach().to("cpu").numpy()
            )  # (num_envs, horizon, action_dim)
            act = dataset.normalizer["action"].unnormalize(act_normed)

            # --- Debug visualization: capture first step of first episode (VLA only) ---
            if i == 0 and t == 0 and arch_variant != "flow_intent":
                try:
                    import wandb
                    ema_net_vis = get_ema_net(agent)
                    if wandb.run is not None and getattr(ema_net_vis, '_last_current_images', None) is not None:
                        debug_img = visualize_vla_debug(
                            net=ema_net_vis,
                            sampled_actions=torch.tensor(act[:4]),  # unnormalized pixel coords
                            gt_actions=None,
                            render_state=getattr(ema_net_vis, '_render_state', None),
                            num_samples=min(4, config.task.num_envs),
                        )
                        wandb.log({
                            f"eval_debug/vla_debug_nstep{num_steps}": wandb.Image(debug_img),
                        }, commit=False)
                        loguru.logger.info(f"Logged VLA debug visualization (Nstep={num_steps})")
                except Exception as e:
                    loguru.logger.warning(f"Failed to create VLA debug visualization: {e}")

            # get action by slicing from start to end
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]

            obs, reward, terminated, truncated, _ = envs.step(act)
            _ = terminated | truncated  # Track done status
            ep_reward += reward
            step_reward.append(reward)
            t += config.task.act_steps

        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    loguru.logger.info(
        f"Nstep: {num_steps} Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}"
    )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    return metrics


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Main pipeline function that calls the appropriate standalone function based on mode."""
    # ── DDP setup ──
    rank, world_size, is_distributed = setup_ddp()
    if is_distributed:
        config.optimization.device = f"cuda:{rank}"
        loguru.logger.info(f"[Rank {rank}/{world_size}] DDP initialized on {config.optimization.device}")

    # general config setup
    set_seed(config.optimization.seed + rank)  # different seed per rank
    if is_main(rank):
        logger = Logger(config)
        loguru.logger.info("Finished setting up logger")
    else:
        logger = None

    # env setup (only rank 0 needs vec envs for eval)
    if is_main(rank):
        envs = make_vec_env(config.task, seed=config.optimization.seed)
        obs, _ = envs.reset()
        loguru.logger.info("Finished setting up env")
    else:
        envs = None

    # dataset setup (all ranks need the dataset)
    # Rank 0 loads first to extract zarr archive; others wait to avoid race conditions
    if is_distributed:
        if is_main(rank):
            dataset = make_dataset(config.task)
            loguru.logger.info("Finished setting up dataset")
        dist.barrier()  # Wait for rank 0 to finish extraction
        if not is_main(rank):
            dataset = make_dataset(config.task)
    else:
        dataset = make_dataset(config.task)
        loguru.logger.info("Finished setting up dataset")

    # Intent conditioning: bump obs_dim to include intent dims (Config B, state obs only).
    # For image obs, intent is concatenated to agent_pos (lowdim input) via shape_meta,
    # so obs_dim (encoder output embedding dim) must NOT be modified.
    _arch_variant = getattr(config.network, "arch_variant", "flow_action")
    if (getattr(config.task, "intent_conditioning", False)
            and _arch_variant != "flow_intent"
            and config.task.obs_type == "state"):
        _cond_dim = config.task.intent_dim  # cv_proxy always uses raw intent_dim
        config.task.obs_dim = config.task.obs_dim + _cond_dim
        loguru.logger.info(
            f"Intent conditioning [Config B/PushT state]: obs_dim -> {config.task.obs_dim} (+{_cond_dim})"
        )

    # All ranks need a renderer env when using image rendering (for consistent DDP forward pass)
    renderer_env = None
    if getattr(config.task, 'use_image_renderer', False) and config.network.network_type in [
        "vla_crossattn_policy", "image_crossattn_policy"
    ]:
        from mip.envs.pusht.pusht_env_wrapper import make_pusht_env
        renderer_env_fn = make_pusht_env(config.task, idx=rank, render=False,
                                          seed=config.optimization.seed + rank)
        renderer_env = renderer_env_fn()
        renderer_env.reset()  # Must reset to initialize pymunk bodies (agent, block)
        loguru.logger.info(f"[Rank {rank}] Created renderer env")

    # Create agent — prefer renderer_env (all ranks), fall back to vec env's first env (rank 0 only)
    single_env = renderer_env if renderer_env is not None else (
        envs.envs[0] if envs is not None and hasattr(envs, 'envs') else None
    )
    if _arch_variant == "flow_intent":
        agent = FlowIntentAgent(config)
        loguru.logger.info("[main] Using Config-A agent: FlowIntentAgent")
    else:
        agent = TrainingAgent(config, env=single_env)

    # Set action normalizer on render-augmented networks (needed to unnormalize draft to pixel coords)
    # Config A (FlowIntentAgent) has no flow_map / get_net(), so skip this block.
    if _arch_variant != "flow_intent":
        for _net in [get_net(agent), get_ema_net(agent)]:
            if hasattr(_net, 'set_action_normalizer'):
                _net.set_action_normalizer(dataset.normalizer['action'])
                loguru.logger.info("[Train] Action normalizer injected into RenderAugmentedNetwork")

    # Instantiate IntentPredictor for Config B (hierarchical_emb) if requested.
    # Config A (FlowIntentAgent) handles intent via its own flow model — no predictor needed.
    intent_predictor = None
    if (getattr(config.task, "intent_conditioning", False)
            and getattr(config.task, "intent_predictor", False)
            and _arch_variant != "flow_intent"):
        if config.task.obs_type == "state":
            _base_obs_dim = config.task.obs_dim  # before intent bump (raw obs)
        else:
            # For image: predictor takes raw agent_pos (2D), not the post-concat 4D
            _base_obs_dim = config.task.intent_dim  # agent_pos raw dim == intent_dim == 2
        intent_predictor = IntentPredictor(
            obs_steps=config.task.obs_steps,
            base_obs_dim=_base_obs_dim,
            intent_dim=config.task.intent_dim,
        ).to(config.optimization.device)
        n_params = sum(p.numel() for p in intent_predictor.parameters())
        loguru.logger.info(
            f"IntentPredictor: {n_params/1e3:.1f}K params "
            f"(in={config.task.obs_steps * _base_obs_dim}, out={config.task.intent_dim})"
        )

    resume_state = None

    pretrained_ckpt = getattr(config.optimization, 'pretrained_ckpt', None)
    if pretrained_ckpt and pretrained_ckpt != "None":
        if is_main(rank):
            loguru.logger.info(f"[Finetune] Loading pretrained base weights from {pretrained_ckpt}")
        agent.load_pretrained(pretrained_ckpt)

    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    elif config.mode == "train" and config.optimization.auto_resume:
        # Automatically look for checkpoint to resume from
        renderer_suffix = ""
        renderer_type = getattr(config.task, 'renderer_type', 'physics_step')
        if renderer_type != "physics_step":
            renderer_suffix = f"_{renderer_type}"
        if _arch_variant == "flow_intent":
            _variant_suffix = "_flow_intent"
        elif getattr(config.task, "intent_conditioning", False):
            _variant_suffix = "_hierarchical_emb"
            if getattr(config.task, "intent_predictor", False):
                _variant_suffix += "_learned"
        else:
            _variant_suffix = ""
        checkpoint_base_name = (
            f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_seed{config.optimization.seed}{renderer_suffix}{_variant_suffix}"
        )
        if is_main(rank) and logger is not None:
            checkpoint_path = logger.find_latest_checkpoint(checkpoint_base_name)
        else:
            checkpoint_path = None

        # Broadcast checkpoint path from rank 0
        if is_distributed:
            paths = [checkpoint_path]
            dist.broadcast_object_list(paths, src=0)
            checkpoint_path = paths[0]

        if checkpoint_path:
            if is_main(rank):
                loguru.logger.info(f"Found checkpoint to resume from: {checkpoint_path}")
                loguru.logger.info("Loading checkpoint with optimizer state...")
            resume_state = agent.load(str(checkpoint_path), load_optimizer=True)
        else:
            if is_main(rank):
                loguru.logger.info("No checkpoint found, starting training from scratch")
    elif config.mode == "train" and not config.optimization.auto_resume:
        if is_main(rank):
            loguru.logger.info("Auto-resume disabled, starting training from scratch")

    # Restore IntentPredictor weights from checkpoint if available
    if intent_predictor is not None and resume_state is not None:
        pred_state = resume_state.get("intent_predictor_state")
        if pred_state is not None:
            intent_predictor.load_state_dict(pred_state)
            loguru.logger.info("Restored IntentPredictor weights from checkpoint")

    # ── Wrap with DDP ──
    # Wrap the inner network (flow_map.net) instead of FlowMap itself,
    # so FlowMap's custom methods (get_velocity, jvp_*, etc.) remain accessible.
    # Config A (FlowIntentAgent) has intent_flow_map.net instead of flow_map.net.
    if is_distributed:
        if _arch_variant == "flow_intent":
            agent.intent_flow_map.net = DDP(agent.intent_flow_map.net, device_ids=[rank],
                                            find_unused_parameters=True)
            agent.action_decoder = DDP(agent.action_decoder, device_ids=[rank])
        else:
            agent.flow_map.net = DDP(agent.flow_map.net, device_ids=[rank],
                                     find_unused_parameters=True)
        # Only wrap encoder if it has trainable parameters (IdentityEncoder has none)
        encoder_params = sum(1 for p in agent.encoder.parameters() if p.requires_grad)
        if encoder_params > 0:
            agent.encoder = DDP(agent.encoder, device_ids=[rank])
        if is_main(rank):
            loguru.logger.info(f"Wrapped networks with DDP ({world_size} GPUs, encoder params: {encoder_params})")

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state,
              rank=rank, world_size=world_size, is_distributed=is_distributed,
              intent_predictor=intent_predictor)
    elif config.mode == "eval":
        if is_main(rank):
            agent.eval()
            if intent_predictor is not None:
                intent_predictor.eval()
            num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics = {"step": num_steps}
                metrics.update(evaluate(config, envs, dataset, agent, logger, num_steps,
                                        intent_predictor=intent_predictor))
                logger.log(metrics, category="eval")

            for key, val in metrics.items():
                if "mean_success" in key:
                    loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")

    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    main()
