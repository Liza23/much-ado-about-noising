"""Training pipeline for LIBERO dataset.

Supports baseline, flow-intent (Config A), and hierarchical-emb (Config B).

Usage:
    # Baseline
    python examples/train_libero.py task=libero_spatial_state \
        task.dataset_path=... task.bddl_file=...
    # Flow intent (Config A)
    python examples/train_libero.py task=libero_spatial_state_flow_intent \
        network=mlp_flow_intent task.dataset_path=... task.bddl_file=...
    # Hierarchical emb (Config B)
    python examples/train_libero.py task=libero_spatial_state_hierarchical_emb \
        task.dataset_path=... task.bddl_file=...
"""

import os
import time

import hydra
import loguru
import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.agent import TrainingAgent
from mip.config import Config
from mip.dataset_utils import loop_dataloader
from mip.datasets.libero_dataset import make_dataset
from mip.envs.libero import make_vec_env
from mip.flow_intent_agent import FlowIntentAgent
from mip.intent_encoder import IntentEncoder
from mip.intent_predictor import IntentPredictor
from mip.logger import Logger, compute_average_metrics, update_best_metrics
from mip.samplers import get_default_step_list
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import set_seed

torch.set_float32_matmul_precision("high")


def train(config: Config, envs, dataset, agent, logger, resume_state=None,
          intent_predictor=None, intent_encoder=None):
    arch_variant = (
        getattr(config.task, "arch_variant", None)
        or getattr(config.network, "arch_variant", "flow_action")
    )
    intent_conditioning = getattr(config.task, "intent_conditioning", False)
    task_id_conditioning = getattr(config.task, "task_id_conditioning", False)
    obs_type = getattr(config.task, "obs_type", "state")

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

    _main_opt = agent.intent_optimizer if arch_variant == "flow_intent" else agent.optimizer
    lr_scheduler = CosineAnnealingLR(_main_opt, T_max=config.optimization.gradient_steps)
    action_lr_scheduler = (
        CosineAnnealingLR(agent.action_optimizer, T_max=config.optimization.gradient_steps)
        if arch_variant == "flow_intent" else None
    )

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
            if intent_pred_lr_scheduler is not None:
                intent_pred_lr_scheduler.step()
            if action_lr_scheduler is not None:
                action_lr_scheduler.step()

    info_list = []
    start_time = time.time()

    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        batch = next(loop_loader)

        if obs_type == "image":
            obs_dict = {
                k: batch["obs"][k][:, : config.task.obs_steps, ...].to(config.optimization.device)
                for k in batch["obs"]
            }
            batch_size = next(iter(obs_dict.values())).shape[0]
            if task_id_conditioning and "task_id" in batch:
                task_oh = F.one_hot(batch["task_id"].long(), num_classes=config.task.num_tasks).float()
                task_oh = task_oh.unsqueeze(1).expand(-1, config.task.obs_steps, -1).to(config.optimization.device)
                obs_dict["state"] = torch.cat([obs_dict["state"], task_oh], dim=-1)
            obs = TensorDict(obs_dict, batch_size=batch_size)
            base_obs = obs
        else:
            obs = batch["obs"]["state"].to(config.optimization.device)
            obs = obs[:, : config.task.obs_steps, :]
            if task_id_conditioning and "task_id" in batch:
                task_oh = F.one_hot(batch["task_id"].long(), num_classes=config.task.num_tasks).float()
                task_oh = task_oh.unsqueeze(1).expand(-1, config.task.obs_steps, -1).to(config.optimization.device)
                obs = torch.cat([obs, task_oh], dim=-1)
            base_obs = obs

        # Intent conditioning (Config B only — Config A passes intent via agent.update)
        _joint_hl_loss = None
        if (intent_conditioning
                and "intent" in batch
                and arch_variant != "flow_intent"
                and obs_type == "state"):
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

            intent_expanded = intent.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
            obs = torch.cat([obs, intent_expanded], dim=-1)

        elif (intent_conditioning
                and "intent" in batch
                and arch_variant != "flow_intent"
                and obs_type == "image"):
            # Config B image mode: encode intent, inject into obs["state"], rebuild TensorDict.
            training_mode = getattr(config.task, "intent_training_mode", "independent")
            _raw_intent = batch["intent"].to(config.optimization.device)
            if (getattr(config.task, "intent_type", "mean") == "encoded_mean"
                    and intent_encoder is not None):
                intent_gt_encoded = intent_encoder(_raw_intent).mean(dim=1)
            else:
                intent_gt_encoded = _raw_intent

            if training_mode == "joint" and intent_predictor is not None:
                # Predictor takes raw state (B, obs_steps, state_dim) — not augmented
                intent_pred_pre = intent_predictor(base_obs["state"])
                hl_loss_pre = F.mse_loss(intent_pred_pre, intent_gt_encoded)
                intent_pred_optimizer.zero_grad()
                hl_loss_pre.backward()
                intent_pred_optimizer.step()
                intent_pred_lr_scheduler.step()
                intent = intent_pred_pre.detach()
                _joint_hl_loss = hl_loss_pre.item()
            else:
                intent = intent_gt_encoded.detach()

            intent_expanded = intent.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
            augmented_state = torch.cat([obs["state"], intent_expanded.contiguous()], dim=-1)
            obs = TensorDict(
                {**{k: obs[k] for k in obs.keys()}, "state": augmented_state},
                batch_size=batch_size,
            )

        act = batch["action"].to(config.optimization.device)
        act = act[:, : config.task.horizon, :]

        delta_t_scalar = warmup_scheduler(n_gradient_step)
        delta_t = torch.full(
            (act.shape[0],), delta_t_scalar, device=config.optimization.device
        )

        if arch_variant == "flow_intent":
            _intent_gt = batch["intent"].to(config.optimization.device)
            info = agent.update(act, base_obs, delta_t, intent_gt=_intent_gt)
            lr_scheduler.step()
            action_lr_scheduler.step()
        else:
            info = agent.update(act, obs, delta_t)
            lr_scheduler.step()

        # Update HL predictor (independent mode)
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
                # For image mode, predictor takes raw state (no images, no intent appended)
                _pred_input = base_obs["state"] if obs_type == "image" else base_obs
                intent_pred = intent_predictor(_pred_input)
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
            logger.save_agent(
                agent=agent,
                identifier="latest",
                training_state={
                    "n_gradient_step": n_gradient_step,
                    "best_metrics": best_metrics,
                    "eval_history": eval_history,
                },
            )

        if (n_gradient_step + 1) % config.log.eval_freq == 0:
            loguru.logger.info("Evaluating...")
            agent.eval()
            if intent_predictor is not None:
                intent_predictor.eval()
            metrics = {"step": n_gradient_step}
            _eval_nsteps = getattr(config.log, "eval_nsteps", 0)
            num_steps_list = [_eval_nsteps] if _eval_nsteps else get_default_step_list(config.optimization.loss_type)
            for num_steps in num_steps_list:
                metrics.update(evaluate(
                    config, envs, dataset, agent, logger, num_steps,
                    intent_predictor=intent_predictor, arch_variant=arch_variant,
                ))

            old_best = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            eval_history.append(metrics.copy())
            avg_metrics = compute_average_metrics(eval_history)

            primary_key = f"mean_success_{num_steps_list[-1]}"
            if primary_key in metrics:
                is_new_best = (
                    primary_key not in old_best
                    or metrics[primary_key] > old_best[primary_key]
                )
                if is_new_best:
                    success_rate = metrics[primary_key]
                    loguru.logger.info(f"New best! {primary_key} = {success_rate:.4f}")
                    logger.save_agent(agent=agent, identifier="best")
                    env_name_for_ckpt = config.task.env_name
                    if len(list(getattr(config.task, "bddl_files", None) or [])) > 1:
                        env_name_for_ckpt += "_suite"
                    ckpt_name = (
                        f"{env_name_for_ckpt}_{config.task.env_type}_{config.task.obs_type}_"
                        f"{config.optimization.loss_type}_{config.network.network_type}_"
                        f"{config.network.emb_dim}_seed{config.optimization.seed}"
                    )
                    if intent_conditioning:
                        ckpt_name += "_intent"
                    if arch_variant == "flow_intent":
                        ckpt_name += "_flow_intent"
                    elif getattr(config.task, "intent_predictor", False):
                        ckpt_name += "_learned"
                    if getattr(config.task, "intent_type", "mean") == "encoded_mean":
                        ckpt_name += "_emb"
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
            if intent_predictor is not None:
                intent_predictor.train()
            if intent_encoder is not None:
                intent_encoder.train()

            early_stop_sr = getattr(config.optimization, "early_stop_sr", None)
            if early_stop_sr is not None and primary_key in metrics:
                if metrics[primary_key] >= early_stop_sr:
                    loguru.logger.info(
                        f"Early stop: {primary_key}={metrics[primary_key]:.3f} >= {early_stop_sr}. Done."
                    )
                    break


def _run_eval_episodes(config, eval_envs, dataset, agent, num_steps,
                       intent_predictor, arch_variant, intent_conditioning,
                       obs_type, image_obs_keys, task_idx: int = 0):
    """Run eval episodes on already-constructed eval_envs. Returns (rewards, steps, successes)."""
    episode_rewards = []
    episode_steps = []
    episode_success = []
    for eval_batch_idx in range(config.log.eval_episodes // config.task.num_envs):
        ep_reward = [0.0] * config.task.num_envs
        success = [0] * config.task.num_envs
        obs, _ = eval_envs.reset()
        t = 0

        while t < config.task.max_episode_steps:
            if obs_type == "image":
                obs_dict = {}
                obs_dict["state"] = torch.tensor(
                    dataset.normalizer["obs"]["state"].normalize(obs["state"].astype(np.float32)),
                    device=config.optimization.device, dtype=torch.float32,
                )
                for img_key in image_obs_keys:
                    obs_dict[img_key] = torch.tensor(
                        obs[img_key].astype(np.float32),
                        device=config.optimization.device, dtype=torch.float32,
                    )
                task_id_conditioning = getattr(config.task, "task_id_conditioning", False)
                if task_id_conditioning:
                    task_oh = torch.zeros(
                        config.task.num_envs, config.task.obs_steps, config.task.num_tasks,
                        device=config.optimization.device,
                    )
                    task_oh[:, :, task_idx] = 1.0
                    obs_dict["state"] = torch.cat([obs_dict["state"], task_oh], dim=-1)
                obs_tensor = TensorDict(obs_dict, batch_size=config.task.num_envs)
                if intent_conditioning and arch_variant != "flow_intent":
                    if intent_predictor is not None:
                        with torch.no_grad():
                            intent_proxy = intent_predictor(obs_tensor["state"])
                    else:
                        raise RuntimeError(
                            "LIBERO eval requires intent_predictor for Config B image mode."
                        )
                    intent_expanded = intent_proxy.unsqueeze(1).expand(
                        -1, config.task.obs_steps, -1
                    )
                    augmented_state = torch.cat(
                        [obs_tensor["state"], intent_expanded.contiguous()], dim=-1
                    )
                    obs_tensor = TensorDict(
                        {**{k: obs_tensor[k] for k in obs_tensor.keys()}, "state": augmented_state},
                        batch_size=config.task.num_envs,
                    )
            else:
                obs = obs.astype(np.float32)
                obs = dataset.normalizer["obs"]["state"].normalize(obs)
                obs_tensor = torch.tensor(
                    obs, device=config.optimization.device, dtype=torch.float32
                )
                task_id_conditioning = getattr(config.task, "task_id_conditioning", False)
                if task_id_conditioning:
                    task_oh = torch.zeros(
                        config.task.num_envs, config.task.obs_steps, config.task.num_tasks,
                        device=config.optimization.device,
                    )
                    task_oh[:, :, task_idx] = 1.0
                    obs_tensor = torch.cat([obs_tensor, task_oh], dim=-1)
                if intent_conditioning and arch_variant != "flow_intent":
                    if intent_predictor is not None:
                        with torch.no_grad():
                            intent_proxy = intent_predictor(obs_tensor)
                    else:
                        raise RuntimeError(
                            "LIBERO eval requires intent_predictor for Config B."
                        )
                    intent_expanded = intent_proxy.unsqueeze(1).expand(
                        -1, config.task.obs_steps, -1
                    )
                    obs_tensor = torch.cat([obs_tensor, intent_expanded], dim=-1)

            if arch_variant == "flow_intent":
                with torch.no_grad():
                    act_normed = agent.sample(
                        obs=obs_tensor, use_ema=True, num_steps=num_steps,
                    )
            else:
                act_0 = torch.randn(
                    (config.task.num_envs, config.task.horizon, config.task.act_dim),
                    device=config.optimization.device,
                )
                with torch.no_grad():
                    act_normed = agent.sample(
                        act_0=act_0,
                        obs=obs_tensor if obs_type == "image" else {"state": obs_tensor},
                        num_steps=num_steps,
                        use_ema=True,
                    )

            act_normed = act_normed.detach().cpu().numpy()
            act = dataset.normalizer["action"].unnormalize(act_normed)

            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act = act[:, start:end, :]

            obs, reward, terminated, truncated, info = eval_envs.step(act)
            ep_reward = [ep_reward[i] + reward[i] for i in range(config.task.num_envs)]
            t += config.task.act_steps

            # Check success at every step: successful envs auto-reset (Gymnasium
            # vector env behavior), so terminal success is in _final_info, not
            # in the current-step info.
            if "_final_info" in info:
                for i in range(config.task.num_envs):
                    if info["_final_info"][i]:
                        fi = info["final_info"][i]
                        if fi and "success" in fi:
                            s = fi["success"]
                            success[i] = max(success[i], int(bool(np.asarray(s).any())))
            if "success" in info:
                for i in range(config.task.num_envs):
                    s = info["success"][i] if hasattr(info["success"], "__len__") else info["success"]
                    success[i] = max(success[i], int(bool(np.asarray(s).any())))

        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

        if (eval_batch_idx + 1) % max(1, (config.log.eval_episodes // config.task.num_envs) // 5) == 0:
            loguru.logger.info(
                f"Eval progress Nstep={num_steps}: "
                f"{eval_batch_idx + 1}/{config.log.eval_episodes // config.task.num_envs} batches"
            )
    return episode_rewards, episode_steps, episode_success


def evaluate(config: Config, envs, dataset, agent, logger, num_steps: int = 1,
             intent_predictor=None, arch_variant="flow_action"):
    intent_conditioning = getattr(config.task, "intent_conditioning", False)
    obs_type = getattr(config.task, "obs_type", "state")
    image_obs_keys = list(getattr(config.task, "image_obs_keys", None) or [])

    bddl_files = list(getattr(config.task, "bddl_files", None) or [])

    if bddl_files:
        # Multi-task eval: run episodes on each task, average success across tasks.
        all_rewards, all_steps, all_success = [], [], []
        per_task_success = {}
        for task_idx, bddl_file in enumerate(bddl_files):
            task_name = os.path.splitext(os.path.basename(bddl_file))[0]
            config.task.bddl_file = bddl_file
            eval_envs = make_vec_env(
                config.task, seed=config.optimization.seed + 1000 + num_steps
            )
            try:
                r, s, succ = _run_eval_episodes(
                    config, eval_envs, dataset, agent, num_steps,
                    intent_predictor, arch_variant, intent_conditioning,
                    obs_type, image_obs_keys, task_idx=task_idx,
                )
            finally:
                eval_envs.close()
            task_success = float(np.nanmean(succ))
            per_task_success[task_name] = task_success
            loguru.logger.info(f"  [{task_name}] success={task_success:.3f}")
            all_rewards.extend(r)
            all_steps.extend(s)
            all_success.extend(succ)

        mean_success = float(np.nanmean(all_success))
        loguru.logger.info(
            f"Nstep={num_steps} | suite mean_success={mean_success:.3f} "
            f"(over {len(bddl_files)} tasks)"
        )
        metrics = {
            f"mean_step_{num_steps}": float(np.nanmean(all_steps)),
            f"mean_reward_{num_steps}": float(np.nanmean(all_rewards)),
            f"mean_success_{num_steps}": mean_success,
        }
        for task_name, s in per_task_success.items():
            metrics[f"task_success_{num_steps}/{task_name}"] = s
        return metrics
    else:
        # Single-task eval (original behavior).
        # Recreate eval envs each call. The long-lived LIBERO image envs have been
        # observed to stall after several eval cycles, leaving jobs "running" but
        # stuck inside evaluation for hours.
        eval_envs = make_vec_env(
            config.task, seed=config.optimization.seed + 1000 + num_steps
        )
        try:
            episode_rewards, episode_steps, episode_success = _run_eval_episodes(
                config, eval_envs, dataset, agent, num_steps,
                intent_predictor, arch_variant, intent_conditioning,
                obs_type, image_obs_keys,
            )
        finally:
            eval_envs.close()

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

    arch_variant = (
        getattr(config.task, "arch_variant", None)
        or getattr(config.network, "arch_variant", "flow_action")
    )
    intent_conditioning = getattr(config.task, "intent_conditioning", False)
    task_id_conditioning = getattr(config.task, "task_id_conditioning", False)

    obs_type = getattr(config.task, "obs_type", "state")
    config.task.save_video = config.log.save_video
    # For suite configs, bddl_file is null but bddl_files lists all tasks.
    # Use the first task's bddl_file to initialize the env (just for obs shape).
    bddl_files = list(getattr(config.task, "bddl_files", None) or [])
    if not config.task.bddl_file and bddl_files:
        config.task.bddl_file = bddl_files[0]
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()

    if obs_type == "image":
        # For image obs, obs_dim is the encoder embedding dimension.
        # The actual obs is a dict; state_dim is read from its "state" key.
        config.task.obs_dim = config.network.emb_dim
        base_obs_dim = obs["state"].shape[-1]
        loguru.logger.info(
            f"Image mode: state_dim={base_obs_dim}, obs_dim (emb)={config.task.obs_dim}"
        )
        # Intent conditioning [Config B image]: update shape_meta so the encoder
        # is built with the intent-augmented state dim (state_dim + intent_emb_dim).
        if intent_conditioning and arch_variant != "flow_intent":
            _intent_type = getattr(config.task, "intent_type", "mean")
            _cond_dim = (
                getattr(config.task, "intent_emb_dim", 64)
                if _intent_type == "encoded_mean"
                else config.task.intent_dim
            )
            _new_state_dim = base_obs_dim + _cond_dim
            config.task.shape_meta["obs"]["state"]["shape"] = [_new_state_dim]
            loguru.logger.info(
                f"Intent conditioning [Config B image]: shape_meta state_dim "
                f"{base_obs_dim} -> {_new_state_dim}"
            )
        if task_id_conditioning:
            dataset_paths = list(getattr(config.task, "dataset_paths", None) or [config.task.dataset_path])
            config.task.num_tasks = len(dataset_paths)
            _cur_state_dim = config.task.shape_meta["obs"]["state"]["shape"][0]
            config.task.shape_meta["obs"]["state"]["shape"] = [_cur_state_dim + config.task.num_tasks]
            loguru.logger.info(
                f"Task ID conditioning [image]: shape_meta state_dim "
                f"{_cur_state_dim} -> {_cur_state_dim + config.task.num_tasks} (num_tasks={config.task.num_tasks})"
            )
    else:
        config.task.obs_dim = obs.shape[-1]
        base_obs_dim = config.task.obs_dim
        loguru.logger.info(f"obs_dim resolved to {config.task.obs_dim}")

    # Intent conditioning: bump obs_dim for Config B (state mode only)
    if intent_conditioning and arch_variant != "flow_intent" and obs_type == "state":
        _intent_type = getattr(config.task, "intent_type", "mean")
        _cond_dim = (
            getattr(config.task, "intent_emb_dim", 64)
            if _intent_type == "encoded_mean"
            else config.task.intent_dim
        )
        config.task.obs_dim = base_obs_dim + _cond_dim

    # Task ID conditioning: bump obs_dim (state mode only; image mode uses shape_meta)
    if task_id_conditioning and obs_type == "state":
        dataset_paths = list(getattr(config.task, "dataset_paths", None) or [config.task.dataset_path])
        config.task.num_tasks = len(dataset_paths)
        config.task.obs_dim = config.task.obs_dim + config.task.num_tasks
        loguru.logger.info(
            f"Task ID conditioning [state]: obs_dim -> {config.task.obs_dim} (num_tasks={config.task.num_tasks})"
        )
        loguru.logger.info(
            f"Intent conditioning [Config B]: obs_dim {base_obs_dim} -> {config.task.obs_dim}"
        )

    dataset = make_dataset(config.task)
    loguru.logger.info(f"Dataset: {dataset}")

    # Use fresh envs for evaluation to avoid long-lived LIBERO image env stalls.
    envs.close()
    envs = None

    if arch_variant == "flow_intent":
        agent = FlowIntentAgent(config)
        loguru.logger.info("Using Config-A agent: FlowIntentAgent")
    else:
        agent = TrainingAgent(config)

    # Intent predictor + encoder (Config B only)
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
            intent_encoder = IntentEncoder(
                raw_intent_dim=config.task.intent_dim,
                intent_emb_dim=getattr(config.task, "intent_emb_dim", 64),
            ).to(config.optimization.device)

    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        loguru.logger.info(f"Loading model from {config.optimization.model_path}")
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)
    elif config.mode == "train" and config.optimization.auto_resume:
        env_name_for_ckpt = config.task.env_name
        if len(list(getattr(config.task, "bddl_files", None) or [])) > 1:
            env_name_for_ckpt += "_suite"
        ckpt_name = (
            f"{env_name_for_ckpt}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_seed{config.optimization.seed}"
        )
        if intent_conditioning:
            ckpt_name += "_intent"
        if arch_variant == "flow_intent":
            ckpt_name += "_flow_intent"
        elif getattr(config.task, "intent_predictor", False):
            ckpt_name += "_learned"
        if getattr(config.task, "intent_type", "mean") == "encoded_mean":
            ckpt_name += "_emb"
        model_latest_path = logger.model_dir / "model_latest.pt"
        if model_latest_path.exists():
            loguru.logger.info(f"Auto-resuming from model_latest.pt at {model_latest_path}")
            resume_state = agent.load(str(model_latest_path), load_optimizer=True)
        else:
            ckpt_path = logger.find_latest_checkpoint(ckpt_name)
            if ckpt_path:
                loguru.logger.info(f"Auto-resuming from {ckpt_path}")
                resume_state = agent.load(str(ckpt_path), load_optimizer=True)

    # Restore intent predictor/encoder from checkpoint
    if intent_predictor is not None and resume_state is not None:
        pred_state = resume_state.get("intent_predictor_state")
        if pred_state is not None:
            intent_predictor.load_state_dict(pred_state)
    if intent_encoder is not None and resume_state is not None:
        enc_state = resume_state.get("intent_encoder_state")
        if enc_state is not None:
            intent_encoder.load_state_dict(enc_state)

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state,
              intent_predictor=intent_predictor, intent_encoder=intent_encoder)
    elif config.mode == "eval":
        agent.eval()
        if intent_predictor is not None:
            intent_predictor.eval()
        _eval_nsteps = getattr(config.log, "eval_nsteps", 0)
        num_steps_list = [_eval_nsteps] if _eval_nsteps else get_default_step_list(config.optimization.loss_type)
        for num_steps in num_steps_list:
            metrics = evaluate(
                config, envs, dataset, agent, logger, num_steps,
                intent_predictor=intent_predictor, arch_variant=arch_variant,
            )
            for key, val in metrics.items():
                if "success" in key:
                    loguru.logger.info(f"{key}: {val}")
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    main()
