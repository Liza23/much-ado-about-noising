"""Oracle intent eval: quantify the train/eval mismatch for flow_intent models.

For each batch in the dataset:
  1. GT oracle: decode action using ground-truth future eef intent
  2. ODE-sampled: decode action using ODE-sampled intent from obs
  3. Compare action MSE (in normalized space) against the GT action labels

This definitively proves (or disproves) that the intent predictor quality is the
bottleneck causing flow_intent to underperform the baseline at eval time.

Usage:
    python examples/eval_oracle_intent.py \\
        task=libero_spatial_suite_image_flow_intent \\
        task.dataset_paths=["/path/to/dataset.hdf5"] \\
        task.bddl_files=[] \\
        optimization.model_path=/path/to/model_best.pt \\
        ++eval_oracle.num_batches=200 \\
        ++eval_oracle.batch_size=64 \\
        ++eval_oracle.num_steps=10
"""

import os

import hydra
import loguru
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.config import Config
from mip.datasets.libero_dataset import make_dataset
from mip.flow_intent_agent import FlowIntentAgent
from mip.torch_utils import set_seed


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    set_seed(config.optimization.seed)
    device = config.optimization.device

    arch_variant = (
        getattr(config.task, "arch_variant", None)
        or getattr(config.network, "arch_variant", "flow_action")
    )
    assert arch_variant == "flow_intent", (
        f"This script only works for flow_intent models, got arch_variant={arch_variant}"
    )

    # Oracle-specific hyperparams — passed via ++eval_oracle.num_batches=N on CLI
    _oracle_cfg = OmegaConf.select(config, "eval_oracle", default=None)
    num_batches = int(getattr(_oracle_cfg, "num_batches", 200) if _oracle_cfg is not None else 200)
    batch_size = int(getattr(_oracle_cfg, "batch_size", 64) if _oracle_cfg is not None else 64)
    num_steps = int(getattr(_oracle_cfg, "num_steps", 10) if _oracle_cfg is not None else 10)

    loguru.logger.info(
        f"Oracle eval: num_batches={num_batches}, batch_size={batch_size}, num_steps={num_steps}"
    )

    # Build a temporary env just to get obs_dim — close immediately after.
    from mip.envs.libero import make_vec_env

    bddl_files = list(getattr(config.task, "bddl_files", None) or [])
    obs_type = getattr(config.task, "obs_type", "state")

    if not config.task.bddl_file and bddl_files:
        config.task.bddl_file = bddl_files[0]
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    envs.close()

    if obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        config.task.obs_dim = obs.shape[-1]
    loguru.logger.info(f"obs_dim={config.task.obs_dim}")

    # Dataset (intent_conditioning must be True in the task config)
    dataset = make_dataset(config.task)
    loguru.logger.info(f"Dataset size: {len(dataset)}")
    assert dataset.intent_conditioning, "Dataset must have intent_conditioning=True"

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Build agent and load checkpoint
    agent = FlowIntentAgent(config)

    model_path = getattr(config.optimization, "model_path", None)
    assert model_path and model_path != "None", (
        "Provide optimization.model_path=/path/to/model_best.pt"
    )
    loguru.logger.info(f"Loading checkpoint: {model_path}")
    agent.load(model_path, load_optimizer=False)

    # ---- Evaluation loop -------------------------------------------------------
    gt_mses = []
    ode_mses = []
    intent_mses = []  # MSE between GT intent and ODE-sampled intent

    obs_steps = config.task.obs_steps
    act_steps = config.task.act_steps

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        # Move to device; dataset returns full horizon, agent expects only obs_steps.
        if obs_type == "image":
            obs_batch = {}
            for k, v in batch["obs"].items():
                t = v.to(device, dtype=torch.float32)
                obs_batch[k] = t[:, :obs_steps, ...].contiguous()
            from tensordict import TensorDict
            obs_tensor = TensorDict(obs_batch, batch_size=batch_size)
        else:
            obs_tensor = batch["obs"]["state"].to(device, dtype=torch.float32)
            obs_tensor = obs_tensor[:, :obs_steps, :].contiguous()

        gt_intent = batch["intent"].to(device, dtype=torch.float32)  # (B, intent_dim)
        gt_action = batch["action"].to(device, dtype=torch.float32)  # (B, horizon, act_dim)

        # Ground-truth action slice we care about: [obs_steps-1 : obs_steps-1+act_steps]
        start = obs_steps - 1
        end = start + act_steps
        gt_action_slice = gt_action[:, start:end, :]

        # 1. GT oracle: decode with ground-truth intent
        pred_action_gt = agent.sample_given_intent(
            obs=obs_tensor, intent_vec=gt_intent, use_ema=True, num_steps=num_steps
        )
        pred_action_gt_slice = pred_action_gt[:, start:end, :]
        gt_mse = F.mse_loss(pred_action_gt_slice, gt_action_slice).item()
        gt_mses.append(gt_mse)

        # 2. ODE-sampled intent: sample intent from noise then decode
        ode_intent = agent.sample_intent(obs=obs_tensor, use_ema=True, num_steps=num_steps)
        pred_action_ode = agent.sample_given_intent(
            obs=obs_tensor, intent_vec=ode_intent, use_ema=True, num_steps=num_steps
        )
        pred_action_ode_slice = pred_action_ode[:, start:end, :]
        ode_mse = F.mse_loss(pred_action_ode_slice, gt_action_slice).item()
        ode_mses.append(ode_mse)

        # 3. Intent quality: MSE between GT intent and ODE-sampled intent
        intent_mse = F.mse_loss(ode_intent, gt_intent).item()
        intent_mses.append(intent_mse)

        if (batch_idx + 1) % 20 == 0:
            loguru.logger.info(
                f"[{batch_idx+1}/{num_batches}] "
                f"action_mse_gt={np.mean(gt_mses):.4f}  "
                f"action_mse_ode={np.mean(ode_mses):.4f}  "
                f"intent_mse={np.mean(intent_mses):.4f}"
            )

    # ---- Summary ---------------------------------------------------------------
    mean_gt = np.mean(gt_mses)
    mean_ode = np.mean(ode_mses)
    mean_intent = np.mean(intent_mses)
    ratio = mean_ode / (mean_gt + 1e-9)

    loguru.logger.info("=" * 60)
    loguru.logger.info("ORACLE EVAL RESULTS")
    loguru.logger.info(f"  Batches evaluated : {len(gt_mses)}")
    loguru.logger.info(f"  action_mse (GT intent)  : {mean_gt:.4f}  <- upper bound decoder can reach")
    loguru.logger.info(f"  action_mse (ODE intent) : {mean_ode:.4f}  <- what eval actually gets")
    loguru.logger.info(f"  intent_mse (ODE vs GT)  : {mean_intent:.4f}  <- intent predictor error")
    loguru.logger.info(f"  ODE / GT MSE ratio      : {ratio:.2f}x")
    loguru.logger.info("=" * 60)
    if ratio > 2.0:
        loguru.logger.info(
            "CONCLUSION: Large ratio -> intent predictor is the bottleneck. "
            "Train/eval mismatch is the root cause of performance gap."
        )
    elif ratio > 1.3:
        loguru.logger.info(
            "CONCLUSION: Moderate ratio -> intent predictor contributes to gap, "
            "but decoder may also be partly at fault."
        )
    else:
        loguru.logger.info(
            "CONCLUSION: Small ratio -> intent predictor quality is NOT the bottleneck. "
            "Look elsewhere for the performance gap."
        )


if __name__ == "__main__":
    main()
