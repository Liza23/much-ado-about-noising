"""Offline eval of a saved LIBERO-Goal checkpoint with per-task SR breakdown.

Usage:
    python examples/eval_libero_goal_offline.py \
        task=libero_goal_suite_image \
        network.rgb_model_name=resnet50 \
        optimization.loss_type=flow \
        optimization.model_path=/path/to/model_best.pt \
        log.eval_episodes=50 \
        task.num_envs=5 \
        log.eval_nsteps=9
"""

import os
import sys
from pathlib import Path

import hydra
import loguru
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mip.agent import TrainingAgent
from mip.config import Config
from mip.datasets.libero_dataset import make_dataset
from mip.envs.libero import make_vec_env
from mip.flow_intent_agent import FlowIntentAgent
from mip.samplers import get_default_step_list
from mip.torch_utils import set_seed

# Import evaluate() from train_libero
sys.path.insert(0, str(ROOT / "examples"))
from train_libero import evaluate


class _NullLogger:
    def log(self, *a, **kw): pass
    def info(self, *a, **kw): pass


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config: Config):
    set_seed(config.optimization.seed)

    model_path = getattr(config.optimization, "model_path", None)
    assert model_path and model_path != "None", \
        "Provide optimization.model_path=/path/to/model_best.pt"

    arch_variant = (
        getattr(config.task, "arch_variant", None)
        or getattr(config.network, "arch_variant", "flow_action")
    )
    loguru.logger.info(f"arch_variant={arch_variant}  model={model_path}")

    # Probe obs_dim from env
    bddl_files = list(getattr(config.task, "bddl_files", None) or [])
    assert bddl_files, "task must have bddl_files (use a suite config)"

    config.task.bddl_file = bddl_files[0]
    probe_envs = make_vec_env(config.task, seed=config.optimization.seed)
    probe_obs, _ = probe_envs.reset()
    if getattr(config.task, "obs_type", "state") == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        config.task.obs_dim = probe_obs["state"].shape[-1]
    probe_envs.close()
    loguru.logger.info(f"obs_dim={config.task.obs_dim}")

    dataset = make_dataset(config.task)

    if arch_variant == "flow_intent":
        agent = FlowIntentAgent(config)
    else:
        agent = TrainingAgent(config)

    loguru.logger.info(f"Loading checkpoint: {model_path}")
    agent.load(model_path, load_optimizer=False)
    agent.eval()

    _eval_nsteps = getattr(config.log, "eval_nsteps", 0)
    num_steps_list = (
        [_eval_nsteps] if _eval_nsteps
        else get_default_step_list(config.optimization.loss_type)
    )

    loguru.logger.info(
        f"Evaluating {len(bddl_files)} tasks × {config.log.eval_episodes} episodes, "
        f"num_envs={config.task.num_envs}, ODE steps={num_steps_list}"
    )

    for num_steps in num_steps_list:
        loguru.logger.info(f"\n{'='*60}")
        loguru.logger.info(f"ODE steps = {num_steps}")
        loguru.logger.info(f"{'='*60}")
        metrics = evaluate(
            config, envs=None, dataset=dataset, agent=agent,
            logger=_NullLogger(), num_steps=num_steps,
            intent_predictor=None, arch_variant=arch_variant,
        )
        loguru.logger.info(f"\nFINAL METRICS (nsteps={num_steps}):")
        for k, v in sorted(metrics.items()):
            if "success" in k:
                loguru.logger.info(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
