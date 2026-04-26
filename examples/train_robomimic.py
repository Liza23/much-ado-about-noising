"""Training pipeline for robomimic dataset.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import os
import time
from contextlib import contextmanager

import io

import hydra
import loguru
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

# Set MuJoCo rendering backend before importing any robomimic/mujoco modules.
# Respect caller-provided MUJOCO_GL; default to EGL for headless GPU rendering.
os.environ.setdefault("MUJOCO_GL", "egl")  # noqa: E402

# Import mip modules after setting environment variables
from mip.agent import TrainingAgent  # noqa: E402
from mip.config import Config  # noqa: E402
from mip.flow_intent_agent import FlowIntentAgent  # noqa: E402  [Config A]
from mip.intent_encoder import IntentEncoder  # noqa: E402
from mip.intent_predictor import IntentPredictor  # noqa: E402
from mip.dataset_utils import loop_dataloader  # noqa: E402
from mip.datasets.robomimic_dataset import make_dataset  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
from mip.logger import (  # noqa: E402
    Logger,
    compute_average_metrics,
    update_best_metrics,
)
from mip.samplers import get_default_step_list  # noqa: E402
from mip.scheduler import WarmupAnnealingScheduler  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, record_dict: dict):
    """Context manager for timing code sections.

    Args:
        section: Name of the section being timed
        record_dict: Dictionary to store timing results
    """
    start = time.perf_counter()
    yield
    record_dict[section].append(time.perf_counter() - start)


def get_net(agent):
    """Unwrap DDP / FlowMap to reach the raw nn.Module network."""
    net = agent.flow_map.net
    if hasattr(net, 'module'):
        net = net.module
    return net


def get_ema_net(agent):
    """Unwrap DDP / FlowMap EMA to reach the raw nn.Module network."""
    net = agent.flow_map_ema.net
    if hasattr(net, 'module'):
        net = net.module
    return net


def _get_sim_state(env) -> np.ndarray | None:
    """Unwrap env wrapper stack to get the flat MuJoCo sim state.

    Used to inject render_state into RenderAugmentedNetwork at eval time so the
    network receives the same render-augmented conditioning it saw during training.
    Returns None if the env is not a robomimic env or get_state() is unavailable.
    """
    try:
        from robomimic.envs.env_robosuite import EnvRobosuite
        current = env
        while not isinstance(current, EnvRobosuite):
            if hasattr(current, 'env'):
                current = current.env
            else:
                return None
        return current.get_state()["states"].astype(np.float32)
    except Exception:
        return None


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
            Co-trained with intent_predictor; its params are added to the same optimizer.
    """
    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=4 if config.task.obs_type == "state" else 8,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True,
        # IMPORTANT: drop_last=True is required for CUDA graphs (static shapes)
        drop_last=True,
    )
    loop_loader = loop_dataloader(dataloader)

    # Config A vs Config B: choose the right optimizer(s) for LR scheduling.
    arch_variant = getattr(config.network, "arch_variant", "flow_action")

    # lr scheduler
    # Config A exposes intent_optimizer (encoder + intent flow).
    # Config B exposes optimizer (encoder + flow map).
    _main_opt = agent.intent_optimizer if arch_variant.startswith("flow_intent") else agent.optimizer
    lr_scheduler = CosineAnnealingLR(
        _main_opt,
        T_max=config.optimization.gradient_steps,
    )
    # Config A also needs a scheduler for the separate action_optimizer.
    action_lr_scheduler = (
        CosineAnnealingLR(agent.action_optimizer, T_max=config.optimization.gradient_steps)
        if arch_variant.startswith("flow_intent") else None
    )

    # Intent predictor optimizer (trained jointly with the policy).
    # For encoded_mean, intent_encoder params are included so both are co-trained
    # by the intent MSE loss, allowing the encoder to adapt its embedding space.
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
        raw_history = resume_state.get("eval_history", [])
        # Strip any non-primitive values (e.g. wandb.Image) that can't be pickled on re-save.
        eval_history = [
            {k: v for k, v in entry.items() if isinstance(v, (int, float, bool, str))}
            for entry in raw_history
        ]
        loguru.logger.info(f"Resuming training from step {start_step}")
        loguru.logger.info(f"Restored best metrics: {best_metrics}")

        # Fast-forward the lr_scheduler to the correct step
        for _ in range(start_step):
            lr_scheduler.step()
            if intent_pred_lr_scheduler is not None:
                intent_pred_lr_scheduler.step()
            if action_lr_scheduler is not None:
                action_lr_scheduler.step()

    info_list = []
    start_time = time.time()

    # Performance tracking
    perf_times = {
        "data_load": [],
        "preprocess": [],
        "update": [],
        "total_step": [],
    }

    # Encoder warmup: freeze IntentEncoder after intent_encoder_warmup_steps.
    # Set once and never unfreeze — the embedding space is stable after warmup.
    _enc_warmup_steps = getattr(config.task, "intent_encoder_warmup_steps", 0)
    _encoder_frozen = False  # track so we only log the freeze event once

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
                f"[encoded_mean] IntentEncoder frozen at step {n_gradient_step}. "
                f"Embedding space is now fixed; flow/action models train on stable 64D intent."
            )

        with timed("total_step", perf_times):
            # get batch from dataloader
            with timed("data_load", perf_times):
                batch = next(loop_loader)

            # preprocess data
            with timed("preprocess", perf_times):
                from tensordict import TensorDict

                if config.task.obs_type == "image":
                    obs_batch = batch["obs"]
                    obs_dict = {}
                    for k in obs_batch:
                        obs_dict[k] = obs_batch[k][:, : config.task.obs_steps, :].to(
                            config.optimization.device
                        )
                    batch_size = next(iter(obs_dict.values())).shape[0]
                    # base_obs for image: TensorDict without intent key (used by Config A agent.update)
                    base_obs = TensorDict(obs_dict, batch_size=batch_size)

                    # Image intent conditioning (Config B only — Config A never appends intent to obs).
                    # Concatenate lowdim keys → base_obs for predictor; compute intent
                    # using the same encoded_mean / joint / independent logic as state obs.
                    # After computing `intent` (B, cond_dim), add as "intent" key in obs_dict.
                    # MultiImageObsEncoder processes it as an additional low_dim input via shape_meta.
                    if (getattr(config.task, "intent_conditioning", False)
                            and "intent" in batch
                            and not arch_variant.startswith("flow_intent")):
                        training_mode = getattr(config.task, "intent_training_mode", "independent")
                        _lowdim_parts = [obs_dict[k] for k in dataset.lowdim_keys]
                        # Overwrite base_obs with lowdim-only tensor for predictor input.
                        # (Config B predictor takes lowdim features, not full image TensorDict)
                        base_obs = torch.cat(_lowdim_parts, dim=-1)  # (B, obs_steps, lowdim_dim)

                        _raw_intent = batch["intent"].to(config.optimization.device)
                        if (getattr(config.task, "intent_type", "mean") == "encoded_mean"
                                and intent_encoder is not None):
                            intent_gt_encoded = intent_encoder(_raw_intent).mean(dim=1)
                        else:
                            intent_gt_encoded = _raw_intent

                        if training_mode == "joint" and intent_predictor is not None:
                            intent_pred_pre = intent_predictor(base_obs)
                            hl_loss_pre = torch.nn.functional.mse_loss(intent_pred_pre, intent_gt_encoded)
                            intent_pred_optimizer.zero_grad()
                            hl_loss_pre.backward()
                            intent_pred_optimizer.step()
                            intent_pred_lr_scheduler.step()
                            intent = intent_pred_pre.detach()
                            _joint_hl_loss = hl_loss_pre.item()
                        else:
                            intent = intent_gt_encoded.detach()
                            _joint_hl_loss = None

                        # Add intent as a new low_dim key; encoder picks it up via shape_meta
                        intent_expanded = intent.unsqueeze(1).expand(-1, config.task.obs_steps, -1).contiguous()
                        obs_dict["intent"] = intent_expanded  # (B, obs_steps, cond_dim)

                    # Convert to TensorDict for consistent handling throughout pipeline
                    obs = TensorDict(obs_dict, batch_size=batch_size)
                elif config.task.obs_type == "state":
                    obs = batch["obs"]["state"].to(config.optimization.device)
                    obs = obs[
                        :, : config.task.obs_steps, :
                    ]  # (B, obs_steps, base_obs_dim)
                    base_obs = obs  # keep ref before intent is appended (used by HL predictor)

                    # Intent conditioning: concatenate intent to each obs step.
                    # "independent": flow policy trains on GT intent from the dataset.
                    # "joint": HL predictor is updated first (MSE vs GT), then its
                    #   output is detached and used to condition the flow policy.
                    #   This matches the user spec order (HL first, flow second) and
                    #   avoids running the predictor twice per step.
                    # Config A (flow_intent): skip this block entirely — FlowIntentAgent
                    #   never appends intent to obs; it handles intent via a flow model.
                    if (getattr(config.task, "intent_conditioning", False)
                            and "intent" in batch
                            and not arch_variant.startswith("flow_intent")):
                        training_mode = getattr(config.task, "intent_training_mode", "independent")

                        # encoded_mean pre-processing: encode per-step sequence → mean-pool.
                        # batch["intent"] is (B, N, 7); result is (B, intent_emb_dim).
                        # Gradients flow through intent_encoder from the MSE loss below.
                        # For all other intent types, intent_gt == batch["intent"] unchanged.
                        _raw_intent = batch["intent"].to(config.optimization.device)
                        if (getattr(config.task, "intent_type", "mean") == "encoded_mean"
                                and intent_encoder is not None):
                            # (B, N, 7) → (B, N, emb) → (B, emb); no detach — encoder trained by MSE
                            intent_gt_encoded = intent_encoder(_raw_intent).mean(dim=1)
                        else:
                            intent_gt_encoded = _raw_intent  # (B, intent_dim) as before

                        if training_mode == "joint" and intent_predictor is not None:
                            # Step 1: HL predictor forward + update
                            intent_pred_pre = intent_predictor(base_obs)
                            hl_loss_pre = torch.nn.functional.mse_loss(intent_pred_pre, intent_gt_encoded)
                            intent_pred_optimizer.zero_grad()
                            hl_loss_pre.backward()
                            intent_pred_optimizer.step()
                            intent_pred_lr_scheduler.step()
                            # Step 2: detach and use for flow policy conditioning
                            intent = intent_pred_pre.detach()
                            # stash loss so it's logged below (skip duplicate HL update)
                            _joint_hl_loss = hl_loss_pre.item()
                        else:
                            intent = intent_gt_encoded.detach()  # GT (detached so action flow gets no encoder grad)
                            _joint_hl_loss = None
                        intent_expanded = intent.unsqueeze(1).expand(
                            -1, config.task.obs_steps, -1
                        )  # (B, obs_steps, intent_dim or intent_emb_dim)
                        obs = torch.cat([obs, intent_expanded], dim=-1)
                        # obs is now (B, obs_steps, base_obs_dim + _cond_dim)

                act = batch["action"].to(config.optimization.device)
                act = act[:, : config.task.horizon, :]  # (B, horizon, act_dim)

                # Inject render state for render-augmented training (Config B only).
                # Config A (FlowIntentAgent) has no flow_map / get_net().
                net = get_net(agent) if not arch_variant.startswith("flow_intent") else None
                if net is not None and hasattr(net, 'set_render_data') and "render_state" in batch:
                    render_state = batch["render_state"].to(config.optimization.device)
                    # GT action at step 0 (normalized) for draft_head supervision
                    render_action_gt = batch["action"][:, 0, :].to(config.optimization.device)
                    # Expand render_state to [B, 1, state_dim] to match set_render_data interface
                    net.set_render_data(
                        render_state=render_state.unsqueeze(1),
                        render_action_gt=render_action_gt,
                    )

            # update diffusion
            with timed("update", perf_times):
                delta_t_scalar = warmup_scheduler(n_gradient_step)
                batch_size = act.shape[0]
                delta_t = torch.full(
                    (batch_size,), delta_t_scalar, device=config.optimization.device
                )
                if arch_variant.startswith("flow_intent"):
                    # Config A: agent handles both flow-intent and MLP-action updates.
                    # Use base_obs (no intent appended); GT intent passed explicitly.
                    _intent_gt = batch["intent"].to(config.optimization.device)
                    info = agent.update(act, base_obs, delta_t, intent_gt=_intent_gt)
                    lr_scheduler.step()
                    action_lr_scheduler.step()
                else:
                    # Config B: standard flow-action update (obs may have intent appended).
                    info = agent.update(act, obs, delta_t)
                    lr_scheduler.step()

            # Update HL predictor (independent mode only — joint mode already did it above).
            if intent_predictor is not None and "intent" in batch:
                _tm = getattr(config.task, "intent_training_mode", "independent")
                if _tm == "joint":
                    # Already updated in preprocess; just log the stashed loss.
                    info["intent_pred_mse"] = _joint_hl_loss
                else:
                    _raw_intent_ind = batch["intent"].to(config.optimization.device)
                    if (getattr(config.task, "intent_type", "mean") == "encoded_mean"
                            and intent_encoder is not None):
                        intent_gt_ind = intent_encoder(_raw_intent_ind).mean(dim=1)
                    else:
                        intent_gt_ind = _raw_intent_ind
                    intent_pred = intent_predictor(base_obs)
                    intent_pred_loss = torch.nn.functional.mse_loss(intent_pred, intent_gt_ind)
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
        if ((n_gradient_step + 1) % config.log.log_freq) == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
                "intent_encoder_frozen": int(_encoder_frozen),  # 0→1 transition visible in wandb
            }
            for key in info:
                try:
                    metrics[key] = np.nanmean([info[key] for info in info_list])
                except Exception as e:
                    loguru.logger.error(f"Error calculating {key}: {e}")
                    metrics[key] = np.nan

            # Add performance metrics
            if perf_times["total_step"]:
                metrics["perf/data_load_ms"] = (
                    np.mean(perf_times["data_load"][-config.log.log_freq :]) * 1000
                )
                metrics["perf/preprocess_ms"] = (
                    np.mean(perf_times["preprocess"][-config.log.log_freq :]) * 1000
                )
                metrics["perf/update_ms"] = (
                    np.mean(perf_times["update"][-config.log.log_freq :]) * 1000
                )
                metrics["perf/total_step_ms"] = (
                    np.mean(perf_times["total_step"][-config.log.log_freq :]) * 1000
                )
                metrics["perf/steps_per_sec"] = 1.0 / np.mean(
                    perf_times["total_step"][-config.log.log_freq :]
                )

            logger.log(metrics, category="train")
            info_list = []

        if ((n_gradient_step + 1) % config.log.save_freq) == 0:
            loguru.logger.info("Save model...")
            latest_training_state = {
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
            logger.save_agent(agent=agent, identifier="latest", training_state=latest_training_state)

        if ((n_gradient_step + 1) % config.log.eval_freq) == 0:
            loguru.logger.info("Evaluate model...")
            agent.eval()
            if intent_predictor is not None:
                intent_predictor.eval()
            metrics = {"step": n_gradient_step}
            if getattr(config.log, 'eval_nsteps', 0):
                num_steps_list = [config.log.eval_nsteps]
            else:
                num_steps_list = get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics.update(eval(config, envs, dataset, agent, logger, num_steps,
                                    intent_predictor=intent_predictor))

            # Update best metrics and average metrics
            old_best_metrics = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            # Strip wandb.Image objects before storing — they hold a wandb Run reference
            # that can't be pickled by torch.save (causes AttributeError in __getstate__).
            eval_history.append({k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str))})
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
                    checkpoint_base_name = (
                        f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
                        f"{config.optimization.loss_type}_{config.network.network_type}_"
                        f"{config.network.emb_dim}_h{config.task.horizon}_seed{config.optimization.seed}"
                    )
                    if getattr(config.network, 'use_render_augmentation', False):
                        checkpoint_base_name += "_render"
                    if getattr(config.task, 'intent_conditioning', False):
                        checkpoint_base_name += "_intent"
                    if arch_variant.startswith("flow_intent"):
                        checkpoint_base_name += "_flow_intent"  # Config A
                        if arch_variant == "flow_intent_mip":
                            checkpoint_base_name += "_mip"
                    else:
                        if getattr(config.task, 'intent_predictor', False):
                            checkpoint_base_name += "_learned"
                        if getattr(config.task, 'intent_training_mode', 'independent') == 'joint':
                            checkpoint_base_name += "_joint"
                    if getattr(config.task, 'intent_type', 'mean') == 'encoded_mean':
                        checkpoint_base_name += "_emb"  # encoded_mean has different model shapes
                    if getattr(config.task, 'intent_key_groups', None) is not None:
                        checkpoint_base_name += "_dual"
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
                    # Also save HL predictor as an independent checkpoint so it
                    # can be loaded/evaluated without the flow policy.
                    if intent_predictor is not None:
                        import pathlib
                        hl_path = pathlib.Path("checkpoints") / f"{checkpoint_base_name}_hl_policy.pt"
                        torch.save(intent_predictor.state_dict(), hl_path)
                        loguru.logger.info(f"Saved HL predictor to {hl_path}")

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


def _fig_to_wandb_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    import wandb
    return wandb.Image(img)


def _make_eval_figures(traj_data: dict, intent_data: dict, action_data: dict,
                       multi_action_data: dict | None = None) -> dict:
    """Generate lightweight eval-time figures. Returns dict of wandb.Image objects."""
    figures = {}

    # ── 1. EEF Trajectories (2D projections) ─────────────────────────────────
    methods  = list(traj_data.keys())
    n        = len(methods)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10), squeeze=False)
    proj_labels = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    for col, method in enumerate(methods):
        trajs, successes = traj_data[method]["trajs"], traj_data[method]["success"]
        for row, (xl, yl, xi, yi) in enumerate(proj_labels):
            ax = axes[row][col]
            for traj, suc in zip(trajs, successes):
                ax.plot(traj[:, xi], traj[:, yi],
                        color="steelblue" if suc else "tomato",
                        alpha=0.5 if suc else 0.2, lw=0.8)
            if row == 0:
                ax.set_title(method, fontsize=9)
            ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("EEF Trajectories (blue=success  red=failure)", fontweight="bold")
    plt.tight_layout()
    figures["viz/trajectories"] = _fig_to_wandb_image(fig)
    plt.close(fig)

    # ── 2. Intent prediction error histogram ─────────────────────────────────
    intent_methods = {k: v for k, v in intent_data.items() if v}
    if intent_methods:
        fig, axes = plt.subplots(1, len(intent_methods),
                                 figsize=(5 * len(intent_methods), 4), squeeze=False)
        for col, (method, errors) in enumerate(intent_methods.items()):
            ax = axes[0][col]
            ax.hist(errors, bins=25, color="steelblue", alpha=0.8, edgecolor="white")
            ax.axvline(float(np.mean(errors)), color="tomato", lw=2,
                       label=f"mean={np.mean(errors):.3f}")
            ax.set_title(method, fontsize=9)
            ax.set_xlabel("L2 error (predicted vs future mean eef xyz)")
            ax.legend(fontsize=7)
        fig.suptitle("Intent Prediction Error", fontweight="bold")
        plt.tight_layout()
        figures["viz/intent_accuracy"] = _fig_to_wandb_image(fig)
        plt.close(fig)

    # ── 3. Action variance per dimension ─────────────────────────────────────
    if action_data:
        act_dim_labels = ["dx", "dy", "dz", "r0", "r1", "r2", "r3", "r4", "r5", "grip"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(action_data)))
        n_dims = next(iter(action_data.values())).shape[1]
        x = np.arange(n_dims)
        w = 0.8 / max(len(action_data), 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        for mi, (method, acts) in enumerate(action_data.items()):
            ax.bar(x + mi * w, acts.var(axis=0), w, label=method,
                   color=colors[mi], alpha=0.8)
        ax.set_xticks(x + w * len(action_data) / 2)
        ax.set_xticklabels(act_dim_labels[:n_dims], rotation=30, fontsize=8)
        ax.set_ylabel("Variance"); ax.set_title("Action variance per dimension")
        ax.legend(fontsize=7)
        plt.tight_layout()
        figures["viz/action_variance"] = _fig_to_wandb_image(fig)
        plt.close(fig)

    # ── 4. Action multimodality: fan plot + UMAP/PCA embedding ───────────────
    if multi_action_data:
        for method, multi_chunks in multi_action_data.items():
            # multi_chunks: list of (K, act_steps, act_dim)
            if not multi_chunks:
                continue
            K = multi_chunks[0].shape[0]
            act_steps = multi_chunks[0].shape[1]
            act_dim = multi_chunks[0].shape[2]
            n_episodes = len(multi_chunks)
            colors = plt.cm.viridis(np.linspace(0, 1, n_episodes))

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: fan plot — integrate dx/dy to get xy trajectory per sample
            ax = axes[0]
            for ep_i, chunks in enumerate(multi_chunks):
                # chunks: (K, act_steps, act_dim); dx=0, dy=1
                for k in range(K):
                    xy = np.cumsum(chunks[k, :, :2], axis=0)  # (act_steps, 2)
                    ax.plot(xy[:, 0], xy[:, 1],
                            color=colors[ep_i], alpha=0.4, lw=0.8)
            ax.set_xlabel("Δx (integrated)"); ax.set_ylabel("Δy (integrated)")
            ax.set_title(f"Action fan ({K} samples/step, {n_episodes} eps)")
            sm = plt.cm.ScalarMappable(cmap="viridis",
                                       norm=plt.Normalize(0, n_episodes - 1))
            plt.colorbar(sm, ax=ax, label="episode")

            # Right: UMAP / PCA of all K*N samples
            ax2 = axes[1]
            all_chunks = np.concatenate(multi_chunks, axis=0)  # (N*K, act_steps, act_dim)
            X = all_chunks.reshape(len(all_chunks), -1)
            ep_labels = np.repeat(np.arange(n_episodes), K)
            try:
                import umap as umap_lib
                reducer = umap_lib.UMAP(n_components=2, random_state=0, n_neighbors=min(15, len(X) - 1))
                emb = reducer.fit_transform(X)
                embed_method = "UMAP"
            except Exception:
                from sklearn.decomposition import PCA
                emb = PCA(n_components=2).fit_transform(X)
                embed_method = "PCA"
            sc = ax2.scatter(emb[:, 0], emb[:, 1],
                             c=ep_labels, cmap="viridis", s=10, alpha=0.7)
            plt.colorbar(sc, ax=ax2, label="episode")
            ax2.set_title(f"Action diversity — {embed_method} ({K}×{n_episodes} chunks)")
            ax2.set_xlabel(f"{embed_method} 1"); ax2.set_ylabel(f"{embed_method} 2")

            fig.suptitle(f"Action Multimodality — {method}", fontweight="bold")
            plt.tight_layout()
            figures["viz/action_multimodality"] = _fig_to_wandb_image(fig)
            plt.close(fig)
            break  # only first method for now

    return figures


def eval(config: Config, envs, dataset, agent, logger, num_steps=1,
         intent_predictor=None):
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
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_kit_success = []

    # Trajectory / intent / action data for wandb visualizations
    _viz_trajs    = []   # list of (T, 3) eef_pos arrays
    _viz_success  = []   # list of bool
    _viz_intent_errors = []  # list of floats (L2 error, intent methods only)
    _viz_all_actions   = []  # list of (chunk_size, act_dim) arrays
    _viz_multi_actions = []  # list of (K, act_steps, act_dim) — K samples from same obs
    _viz_intent_start  = getattr(dataset, "intent_start", None)
    _viz_has_intent    = getattr(config.task, "intent_conditioning", False)
    _viz_intent_hz     = getattr(config.task, "intent_horizon",
                                 getattr(config.task, "act_steps", 8))

    # Performance tracking for inference
    inference_times = {
        "normalize": [],
        "sample": [],
        "unnormalize": [],
        "env_step": [],
    }
    video_env = None
    if config.log.save_video:
        if hasattr(envs, "envs") and len(envs.envs) > 0:
            video_env = envs.envs[0]
        else:
            loguru.logger.warning(
                "Video saving is enabled but this vector env backend does not expose "
                "sub-env handles; skipping eval video recording."
            )

    arch_variant = getattr(config.network, "arch_variant", "flow_action")

    for i in range(config.log.eval_episodes // config.task.num_envs):
        ep_reward = [0.0] * config.task.num_envs
        obs, _ = envs.reset()
        t = 0
        _ep_eef = []   # collect eef_pos for env 0 this episode

        # initialize video stream
        if video_env is not None:
            logger.video_init(video_env, enable=True, video_id=str(i))  # save videos

        while t < config.task.max_episode_steps:
            with timed("normalize", inference_times):
                if config.task.obs_type == "state":
                    obs = obs.astype(np.float32)  # (num_envs, obs_steps, base_obs_dim)
                    # normalize obs
                    obs = dataset.normalizer["obs"]["state"].normalize(obs)
                    obs_tensor = torch.tensor(
                        obs, device=config.optimization.device, dtype=torch.float32
                    )  # (num_envs, obs_steps, base_obs_dim)

                    # Collect eef_pos for env 0 (unnormalized world-frame xyz)
                    if _viz_intent_start is not None:
                        obs_unnorm = dataset.normalizer["obs"]["state"].unnormalize(obs)
                        _ep_eef.append(obs_unnorm[0, -1, _viz_intent_start:_viz_intent_start + 3].copy())

                    # Intent conditioning at eval:
                    # Config B: use learned predictor or CV proxy then append to obs.
                    # Config A: FlowIntentAgent samples intent internally via ODE;
                    #           obs is passed without any intent appended.
                    if getattr(config.task, "intent_conditioning", False) and not arch_variant.startswith("flow_intent"):
                        if intent_predictor is not None:
                            # Learned predictor: MLP trained to predict intent in embedding space.
                            # For encoded_mean the predictor directly outputs (num_envs, intent_emb_dim).
                            # For other types it outputs (num_envs, intent_dim). No IntentEncoder needed
                            # at inference — the predictor was trained to predict the embedding directly.
                            with torch.no_grad():
                                intent_proxy = intent_predictor(obs_tensor)  # (num_envs, cond_dim)
                        elif getattr(config.task, "intent_type", "mean") == "encoded_mean":
                            raise RuntimeError(
                                "intent_type=encoded_mean requires intent_predictor=true. "
                                "There is no valid constant-velocity proxy in embedding space."
                            )
                        else:
                            # Constant-velocity projection fallback
                            intent_start = dataset.intent_start
                            intent_end = dataset.intent_end
                            eef_now  = obs_tensor[:, -1, intent_start:intent_end]
                            eef_prev = obs_tensor[:, -2, intent_start:intent_end]
                            velocity = eef_now - eef_prev
                            if getattr(config.task, "intent_type", "mean") == "sequence":
                                # Project each of the N future steps individually → (num_envs, N*7)
                                ks = torch.arange(1, config.task.intent_horizon + 1,
                                                  dtype=torch.float32,
                                                  device=config.optimization.device)
                                # (N, 1) * (1, 7) → (N, 7), then prepend batch dim
                                future_steps = (eef_now.unsqueeze(1)
                                                + ks.view(-1, 1) * velocity.unsqueeze(1))
                                intent_proxy = future_steps.reshape(obs_tensor.shape[0], -1)
                            else:
                                half_horizon = (config.task.intent_horizon + 1) / 2.0
                                intent_proxy = eef_now + half_horizon * velocity
                        intent_expanded = intent_proxy.unsqueeze(1).expand(
                            -1, config.task.obs_steps, -1
                        )  # (num_envs, obs_steps, intent_dim)
                        obs_tensor = torch.cat([obs_tensor, intent_expanded], dim=-1)
                        # obs_tensor: (num_envs, obs_steps, base_obs_dim + intent_dim)

                    # Config A: keep obs_tensor as-is (no intent appended).
                    # The obs dict below is only used for Config B's agent.sample().
                    obs = {"state": obs_tensor}
                else:  # image-based observation
                    obs_raw = obs
                    obs = {}
                    for k in obs_raw:
                        obs[k] = obs_raw[k].astype(
                            np.float32
                        )  # (num_envs, obs_steps, obs_dim)
                        obs[k] = dataset.normalizer["obs"][k].normalize(obs[k])
                        obs[k] = torch.tensor(
                            obs[k],
                            device=config.optimization.device,
                            dtype=torch.float32,
                        )  # (num_envs, obs_steps, obs_dim)

                    # Image intent conditioning at eval (Config B only).
                    # Config A (FlowIntentAgent) samples intent internally — obs passed as-is.
                    if getattr(config.task, "intent_conditioning", False) and not arch_variant.startswith("flow_intent"):
                        _lowdim_parts_eval = [obs[k] for k in dataset.lowdim_keys]
                        lowdim_tensor = torch.cat(_lowdim_parts_eval, dim=-1)
                        # (num_envs, obs_steps, lowdim_dim)
                        if intent_predictor is not None:
                            with torch.no_grad():
                                intent_proxy = intent_predictor(lowdim_tensor)
                                # (num_envs, cond_dim)
                        else:
                            # CV proxy using eef slice (only valid for non-encoded_mean types)
                            if getattr(config.task, "intent_type", "mean") == "encoded_mean":
                                raise RuntimeError(
                                    "intent_type=encoded_mean requires intent_predictor=true. "
                                    "No valid CV proxy in embedding space."
                                )
                            eef_now  = lowdim_tensor[:, -1, dataset.intent_start:dataset.intent_end]
                            eef_prev = lowdim_tensor[:, -2, dataset.intent_start:dataset.intent_end]
                            velocity = eef_now - eef_prev
                            half_h = (config.task.intent_horizon + 1) / 2.0
                            intent_proxy = eef_now + half_h * velocity
                        intent_expanded = intent_proxy.unsqueeze(1).expand(
                            -1, config.task.obs_steps, -1
                        )  # (num_envs, obs_steps, cond_dim)
                        obs["intent"] = intent_expanded.contiguous()  # encoder picks this up via shape_meta

                act_0 = torch.randn(
                    (config.task.num_envs, config.task.horizon, config.task.act_dim),
                    device=config.optimization.device,
                )

            # Inject current sim states into RenderAugmentedNetwork so renders
            # happen at eval time (same conditioning distribution as training).
            # Config A (FlowIntentAgent) has no flow_map_ema / get_ema_net().
            ema_net = get_ema_net(agent) if not arch_variant.startswith("flow_intent") else None
            if ema_net is not None and hasattr(ema_net, 'set_render_data') and hasattr(envs, 'envs'):
                sim_states = [_get_sim_state(e) for e in envs.envs]
                # Only inject if ALL envs succeeded — partial batches cause a shape
                # mismatch between render_state [K] and obs label [num_envs] in fusion.
                if all(s is not None for s in sim_states):
                    render_state_t = torch.tensor(
                        np.stack(sim_states), dtype=torch.float32,
                        device=config.optimization.device,
                    )
                    ema_net.set_render_data(
                        render_state=render_state_t.unsqueeze(1),
                        render_action_gt=None,
                    )

            # run sampling (num_envs, horizon, action_dim)
            with timed("sample", inference_times):
                if arch_variant.startswith("flow_intent"):
                    # Config A: FlowIntentAgent samples intent then decodes action.
                    # State obs: obs_tensor (num_envs, obs_steps, base_obs_dim) — no intent appended.
                    # Image obs: obs dict (TensorDict-compatible) — no "intent" key added.
                    _flow_obs = obs_tensor if config.task.obs_type == "state" else obs
                    act_normed = agent.sample(
                        obs=_flow_obs,
                        use_ema=True,
                        num_steps=num_steps,
                    )
                else:
                    # Config B: standard flow-action sampling.
                    act_normed = agent.sample(
                        act_0=act_0,
                        obs=obs,
                        num_steps=num_steps,
                        use_ema=True,
                    )

            # Multimodality: sample K action chunks from the same obs at step 0.
            # Config A (flow_intent): multimodality comes from different intent samples.
            # Config B (baseline/hierarchical): multimodality comes from different action noise.
            _MULTI_K = 10
            if t == 0 and config.task.obs_type == "state":
                _multi_samples = []
                with torch.no_grad():
                    for _ in range(_MULTI_K):
                        if arch_variant.startswith("flow_intent"):
                            # Config A: each call draws fresh intent noise → different action.
                            _an_k = agent.sample(
                                obs=obs_tensor, use_ema=True, num_steps=num_steps,
                            )
                        else:
                            _act0_k = torch.randn(
                                (config.task.num_envs, config.task.horizon, config.task.act_dim),
                                device=config.optimization.device,
                            )
                            _an_k = agent.sample(
                                act_0=_act0_k, obs=obs, num_steps=num_steps, use_ema=True,
                            )
                        _an_k = _an_k.detach().cpu().numpy()
                        _ak = dataset.normalizer["action"].unnormalize(_an_k)
                        _s = config.task.obs_steps - 1
                        _multi_samples.append(_ak[0, _s:_s + config.task.act_steps])
                _viz_multi_actions.append(np.stack(_multi_samples))  # (K, act_steps, act_dim)

            # unnormalize prediction
            with timed("unnormalize", inference_times):
                act_normed = (
                    act_normed.detach().to("cpu").numpy()
                )  # (num_envs, horizon, action_dim)
                act = dataset.normalizer["action"].unnormalize(act_normed)

                # get action by slicing from start to end
                start = config.task.obs_steps - 1
                end = start + config.task.act_steps
                act = act[:, start:end, :]
                # Collect action chunk for env 0 (unnormalized)
                _viz_all_actions.append(act[0].copy())

                if config.task.abs_action and config.task.env_name in [
                    "can",
                    "lift",
                    "square",
                    "tool_hang",
                    "transport",
                ]:
                    act = dataset.undo_transform_action(act)

            with timed("env_step", inference_times):
                obs, reward, terminated, truncated, info = envs.step(act)
                _ = terminated | truncated
                ep_reward += reward
                t += config.task.act_steps
        success = [1.0 if s > 0 else 0.0 for s in ep_reward]

        # Store trajectory for env 0
        if _ep_eef:
            traj = np.array(_ep_eef)
            _viz_trajs.append(traj)
            _viz_success.append(bool(success[0]))
            # Retrospective intent error: compare predicted intent to actual future mean eef
            if _viz_has_intent and intent_predictor is not None and len(traj) > _viz_intent_hz:
                for step_i in range(len(traj) - _viz_intent_hz):
                    gt_mean = traj[step_i + 1: step_i + 1 + _viz_intent_hz, :3].mean(axis=0)
                    # intent_proxy was already computed per-step; use stored predicted xyz
                    # We only have access to the GT future here, store errors from traj alone
                    # (predicted-vs-actual comparison requires capturing intent_proxy per step;
                    # here we store traj-based GT mean for post-hoc use via eval_visualize.py)
                    _viz_intent_errors.append(np.linalg.norm(traj[step_i, :3] - gt_mean))

        # evaluate kitchen
        kit_success = []
        if "kitchen" in config.task.env_name:
            task_completion_counts = [
                len(info[i]["completed_tasks"][0]) for i in range(config.task.num_envs)
            ]
            for num in task_completion_counts:
                sublist = [1 if i < num else 0 for i in range(7)]
                kit_success.append(sublist)
            # Use p4 success rate as the main success metric for kitchen environments
            success = [1 if num >= 4 else 0 for num in task_completion_counts]

        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)
        episode_kit_success.append(kit_success)
    # Log performance metrics
    loguru.logger.info(
        f"Nstep: {num_steps} Mean step: {np.nanmean(episode_steps)} "
        f"Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}"
    )

    # Calculate inference performance
    if inference_times["sample"]:
        loguru.logger.info(
            f"Inference perf - Normalize: {np.mean(inference_times['normalize']) * 1000:.2f}ms, "
            f"Sample: {np.mean(inference_times['sample']) * 1000:.2f}ms, "
            f"Unnormalize: {np.mean(inference_times['unnormalize']) * 1000:.2f}ms, "
            f"Env step: {np.mean(inference_times['env_step']) * 1000:.2f}ms"
        )

    metrics = {
        f"mean_step_{num_steps}": np.nanmean(episode_steps),
        f"mean_reward_{num_steps}": np.nanmean(episode_rewards),
        f"mean_success_{num_steps}": np.nanmean(episode_success),
    }

    # Add inference performance metrics
    if inference_times["sample"]:
        metrics[f"perf/inference_normalize_ms_{num_steps}"] = (
            np.mean(inference_times["normalize"]) * 1000
        )
        metrics[f"perf/inference_sample_ms_{num_steps}"] = (
            np.mean(inference_times["sample"]) * 1000
        )
        metrics[f"perf/inference_unnormalize_ms_{num_steps}"] = (
            np.mean(inference_times["unnormalize"]) * 1000
        )
        metrics[f"perf/inference_env_step_ms_{num_steps}"] = (
            np.mean(inference_times["env_step"]) * 1000
        )

    if "kitchen" in config.task.env_name:
        mean_kit_success = np.mean(np.array(episode_kit_success), axis=(0, 1))
        kit_metrics = {}
        for i in range(7):
            kit_metrics[f"p{i + 1}_NFE{num_steps}"] = mean_kit_success[i]
        metrics.update(kit_metrics)
        loguru.logger.info(f"Kit metrics: {kit_metrics}")

    # ── Generate and attach wandb visualizations ───────────────────────────
    loguru.logger.info(
        f"[viz] _viz_trajs={len(_viz_trajs)} _viz_multi_actions={len(_viz_multi_actions)} "
        f"obs_type={config.task.obs_type} intent_start={_viz_intent_start}"
    )
    if _viz_trajs and config.task.obs_type == "state":
        try:
            run_label = getattr(config.log, "exp_name", "run")
            traj_data   = {run_label: {"trajs": _viz_trajs,    "success": _viz_success}}
            intent_data = {run_label: _viz_intent_errors} if _viz_intent_errors else {}
            action_data = {}
            if _viz_all_actions:
                action_data[run_label] = np.concatenate(_viz_all_actions, axis=0)
            multi_action_data = {}
            if _viz_multi_actions:
                multi_action_data[run_label] = _viz_multi_actions
                # mean_action_spread: mean std across K samples per step, averaged over dims
                stacked = np.stack(_viz_multi_actions)  # (N_eps, K, act_steps, act_dim)
                metrics[f"mean_action_spread_{num_steps}"] = float(
                    stacked.std(axis=1).mean()
                )
            viz_figures = _make_eval_figures(traj_data, intent_data, action_data,
                                             multi_action_data)
            metrics.update(viz_figures)
        except Exception as e:  # never let viz crash training
            loguru.logger.warning(f"Eval visualization failed (non-fatal): {e}")

    return metrics


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Main pipeline function that calls the appropriate standalone function based on mode."""
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode("warn")
        # from torch/rl, compile use tensor cores for float32 matrix multiplication
        torch.set_float32_matmul_precision("high")

    # general config setup
    set_seed(config.optimization.seed)
    limit_threads(1)
    logger = Logger(config)
    loguru.logger.info("Finished setting up logger")

    # post process config
    if config.network.network_type == "chiunet":
        # make sure config.task.horizon is a power of 2
        old_horizon = config.task.horizon
        config.task.horizon = int(2 ** np.ceil(np.log2(old_horizon)))
        loguru.logger.warning(
            f"ChiUNet requires horizon to be a power of 2, old horizon: {old_horizon}, new horizon: {config.task.horizon}"
        )

    # env setup
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, info = envs.reset()
    # arch_variant is needed unconditionally (image and state, intent and baseline)
    _arch_variant_main = getattr(config.network, "arch_variant", "flow_action")
    if config.task.obs_type == "state":
        base_obs_dim = obs.shape[-1]
        config.task.obs_dim = base_obs_dim
        # Intent conditioning: for Config B (flow_action), augment obs_dim so the
        # encoder/network are built with the extra intent dimensions appended.
        # For Config A (flow_intent), the encoder always sees base obs only —
        # intent is handled by a separate flow model — so obs_dim stays unchanged.
        if getattr(config.task, "intent_conditioning", False):
            _intent_type_main = getattr(config.task, "intent_type", "mean")
            # For sequence intent, effective intent_dim = N steps × 7D eef per step
            if _intent_type_main == "sequence":
                config.task.intent_dim = config.task.intent_horizon * 7
            if not _arch_variant_main.startswith("flow_intent"):
                # Config B only: bump obs_dim to include intent conditioning dims.
                # For encoded_mean the conditioning vector has size intent_emb_dim (not raw intent_dim).
                _cond_dim = (
                    getattr(config.task, "intent_emb_dim", 64)
                    if _intent_type_main == "encoded_mean"
                    else config.task.intent_dim
                )
                config.task.obs_dim = base_obs_dim + _cond_dim
                loguru.logger.info(
                    f"Intent conditioning [Config B]: obs_dim {base_obs_dim} -> {config.task.obs_dim} "
                    f"(+{_cond_dim} intent dims, type={_intent_type_main})"
                )
            else:
                loguru.logger.info(
                    f"Intent conditioning [Config A]: obs_dim stays {base_obs_dim} "
                    f"(intent handled by flow model, not appended to obs)"
                )
    else:
        # For image observations, set obs_dim to embedding dimension
        # This is used by the network but not actually used when encoder_type is "image"
        config.task.obs_dim = config.network.emb_dim
    loguru.logger.info("Finished setting up env")

    # dataset setup
    dataset = make_dataset(config.task)
    loguru.logger.info("Finished setting up dataset")

    # For render-augmented networks, pass a single env so the renderer can be created.
    # The renderer's _unwrap_to_robosuite() handles the full wrapper chain internally.
    single_env = None
    if (
        getattr(config.network, 'use_render_augmentation', False)
        and getattr(config.task, 'use_image_renderer', False)
        and hasattr(envs, 'envs')
        and len(envs.envs) > 0
    ):
        single_env = envs.envs[0]  # MultiStepWrapper (renderer unwraps further)

    if _arch_variant_main.startswith("flow_intent"):
        # Config A: flow intent model + deterministic MLP action decoder.
        # No env needed (no renderer).
        agent = FlowIntentAgent(config)
        loguru.logger.info("[main] Using Config-A agent: FlowIntentAgent")
    else:
        agent = TrainingAgent(config, env=single_env)

    # Inject action normalizer into both train and EMA RenderAugmentedNetwork so they
    # can unnormalize draft actions to world coords for rendering. EMA deepcopy resets
    # action_normalizer to None (not parametric), so must be injected separately.
    # Config A (FlowIntentAgent) has no flow_map / get_net(), so skip this block.
    if not _arch_variant_main.startswith("flow_intent"):
        for net_or_ema in (get_net(agent), get_ema_net(agent)):
            if hasattr(net_or_ema, 'set_action_normalizer'):
                net_or_ema.set_action_normalizer(dataset.normalizer["action"])
        if hasattr(get_net(agent), 'set_action_normalizer'):
            loguru.logger.info("[Train] Action normalizer injected into RenderAugmentedNetwork (train + EMA)")

    # Instantiate learned intent predictor and (optionally) intent seq encoder.
    # Config A handles intent via the flow intent model inside FlowIntentAgent;
    # it never uses the standalone IntentPredictor or IntentEncoder here.
    intent_predictor = None
    intent_encoder = None  # only used for Config B + intent_type=="encoded_mean"
    if (getattr(config.task, "intent_conditioning", False)
            and getattr(config.task, "intent_predictor", False)
            and not _arch_variant_main.startswith("flow_intent")):
        if config.task.obs_type == "state":
            _base_obs_dim = base_obs_dim
        else:
            # For image obs, the predictor takes concatenated raw lowdim features
            # (eef_pos + eef_quat + gripper), not encoder embeddings.
            # Predictor input: obs_steps × lowdim_dim (e.g. 2×9=18 for lift/square).
            _base_obs_dim = sum(
                config.task.shape_meta["obs"][k]["shape"][0]
                for k in dataset.lowdim_keys
            )
        _intent_type_inst = getattr(config.task, "intent_type", "mean")
        # For encoded_mean, predictor outputs intent_emb_dim (not raw intent_dim).
        _pred_out_dim = (
            getattr(config.task, "intent_emb_dim", 64)
            if _intent_type_inst == "encoded_mean"
            else config.task.intent_dim
        )
        intent_predictor = IntentPredictor(
            obs_steps=config.task.obs_steps,
            base_obs_dim=_base_obs_dim,
            intent_dim=_pred_out_dim,
        ).to(config.optimization.device)
        n_params = sum(p.numel() for p in intent_predictor.parameters())
        loguru.logger.info(
            f"IntentPredictor: {n_params/1e3:.1f}K params "
            f"(in={config.task.obs_steps * _base_obs_dim}, out={_pred_out_dim})"
        )

        # For encoded_mean: also create the per-step sequence encoder.
        # Its parameters are included in the predictor optimizer so both are
        # co-trained by the intent MSE loss (co-training lets the encoder learn
        # embeddings that are predictable from obs).
        if _intent_type_inst == "encoded_mean":
            _intent_emb_dim = getattr(config.task, "intent_emb_dim", 64)
            intent_encoder = IntentEncoder(
                raw_intent_dim=config.task.intent_dim,  # 7 (pos3+quat4)
                intent_emb_dim=_intent_emb_dim,
            ).to(config.optimization.device)
            enc_params = sum(p.numel() for p in intent_encoder.parameters())
            loguru.logger.info(
                f"IntentEncoder (encoded_mean): {enc_params/1e3:.1f}K params "
                f"(in=7, out={_intent_emb_dim})"
            )

    resume_state = None

    if getattr(config.optimization, 'pretrained_ckpt', None) and config.optimization.pretrained_ckpt != "None":
        loguru.logger.info(f"Loading pretrained weights from {config.optimization.pretrained_ckpt}")
        agent.load_pretrained(config.optimization.pretrained_ckpt)

    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    elif config.optimization.auto_resume:
        # Automatically look for checkpoint to resume from
        checkpoint_base_name = (
            f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_h{config.task.horizon}_seed{config.optimization.seed}"
        )
        if getattr(config.network, 'use_render_augmentation', False):
            checkpoint_base_name += "_render"
        if getattr(config.task, 'intent_conditioning', False):
            checkpoint_base_name += "_intent"
        if _arch_variant_main.startswith("flow_intent"):
            checkpoint_base_name += "_flow_intent"  # Config A distinguisher
            if _arch_variant_main == "flow_intent_mip":
                checkpoint_base_name += "_mip"
        else:
            if getattr(config.task, 'intent_predictor', False):
                checkpoint_base_name += "_learned"
            if getattr(config.task, 'intent_training_mode', 'independent') == 'joint':
                checkpoint_base_name += "_joint"
        if getattr(config.task, 'intent_type', 'mean') == 'encoded_mean':
            checkpoint_base_name += "_emb"  # encoded_mean has different model shapes
        if getattr(config.task, 'intent_key_groups', None) is not None:
            checkpoint_base_name += "_dual"
        # Prefer model_latest.pt in the run's log dir (has full training state)
        model_latest_path = logger.model_dir / "model_latest.pt"
        if model_latest_path.exists():
            loguru.logger.info(f"Found model_latest.pt, resuming from {model_latest_path}")
            resume_state = agent.load(str(model_latest_path), load_optimizer=True)
        else:
            checkpoint_path = logger.find_latest_checkpoint(checkpoint_base_name)
            if checkpoint_path:
                loguru.logger.info(f"Found checkpoint to resume from: {checkpoint_path}")
                loguru.logger.info("Loading checkpoint with optimizer state...")
                resume_state = agent.load(str(checkpoint_path), load_optimizer=True)
            else:
                loguru.logger.info("No checkpoint found, starting training from scratch")
    elif config.mode == "train" and not config.optimization.auto_resume:
        loguru.logger.info("Auto-resume disabled, starting training from scratch")

    # Restore intent predictor and encoder weights from checkpoint if available
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
            metrics.update(eval(config, envs, dataset, agent, logger, num_steps,
                                intent_predictor=intent_predictor))
            logger.log(metrics, category="eval")

        # print result in easy to read format
        for key, val in metrics.items():
            if "mean_success" in key:
                loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    main()
