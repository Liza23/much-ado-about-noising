"""Intent ablation eval for FlowIntentAgent on Robomimic image tasks.

Runs three conditions on the same checkpoint:
  1. STANDARD  — agent.sample()  (ODE-sampled intent)
  2. ZERO      — agent.sample_given_intent(zeros)
  3. RANDOM    — agent.sample_given_intent(randn, no ODE)

If STANDARD ≈ ZERO ≈ RANDOM → policy ignores intent entirely.
If STANDARD >> ZERO/RANDOM   → intent is load-bearing.

Usage:
    python examples/eval_intent_ablation.py \
        task=lift_mh_image_slot_intent \
        network=mlp_flow_intent \
        optimization.loss_type=flow \
        optimization.model_path=/data/user_data/ldahiya/mip_render/lift-mh-image/slot_intent_seed0_v4/models/model_best.pt \
        log.eval_episodes=50
"""

import os

import hydra
import loguru
import numpy as np
import torch
from omegaconf import DictConfig

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from mip.datasets.robomimic_dataset import make_dataset
from mip.envs.robomimic.robomimic_env import make_vec_env
from mip.flow_intent_agent import FlowIntentAgent
from mip.torch_utils import set_seed


def _run_condition(label, agent, envs, dataset, config, intent_mode: str):
    """Run eval_episodes episodes under a given intent_mode.

    intent_mode: "standard" | "zero" | "random"
    Returns list of per-episode success flags.
    """
    device = config.optimization.device
    obs_steps = config.task.obs_steps
    act_steps = config.task.act_steps
    num_envs = config.task.num_envs
    intent_dim = config.task.intent_dim
    num_episodes = config.log.eval_episodes
    num_batches = max(1, num_episodes // num_envs)

    all_success = []
    for batch_idx in range(num_batches):
        obs, _ = envs.reset()
        ep_reward = np.zeros(num_envs)
        t = 0

        while t < config.task.max_episode_steps:
            # Normalize image obs
            obs_dict = {}
            for k in obs:
                obs_dict[k] = torch.tensor(
                    dataset.normalizer["obs"][k].normalize(obs[k].astype(np.float32)),
                    device=device, dtype=torch.float32,
                )

            with torch.no_grad():
                if intent_mode == "standard":
                    act_normed = agent.sample(obs=obs_dict, use_ema=True, num_steps=9)
                else:
                    if intent_mode == "zero":
                        intent_vec = torch.zeros(num_envs, intent_dim, device=device)
                    else:  # "random"
                        intent_vec = torch.randn(num_envs, intent_dim, device=device)
                    act_normed = agent.sample_given_intent(
                        obs=obs_dict, intent_vec=intent_vec, use_ema=True, num_steps=9
                    )

            act_normed = act_normed.detach().cpu().numpy()
            act = dataset.normalizer["action"].unnormalize(act_normed)
            start = obs_steps - 1
            act = act[:, start:start + act_steps, :]
            if config.task.abs_action and config.task.env_name in [
                "can", "lift", "square", "tool_hang", "transport"
            ]:
                act = dataset.undo_transform_action(act)

            obs, reward, terminated, truncated, info = envs.step(act)
            ep_reward += reward
            t += act_steps

        success = [1.0 if s > 0 else 0.0 for s in ep_reward]
        all_success.extend(success)
        loguru.logger.info(
            f"[{label}] batch {batch_idx+1}/{num_batches} | "
            f"SR={np.mean(all_success):.3f} ({sum(all_success)}/{len(all_success)})"
        )

    return all_success


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config: DictConfig):
    set_seed(config.optimization.seed)

    model_path = getattr(config.optimization, "model_path", None)
    assert model_path and model_path != "None", "Provide optimization.model_path=..."

    config.task.num_envs = 4

    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    # For image obs, obs_dim = embedding dim (same as train_robomimic)
    config.task.obs_dim = config.network.emb_dim

    dataset = make_dataset(config.task)

    agent = FlowIntentAgent(config)
    loguru.logger.info(f"Loading checkpoint: {model_path}")
    agent.load(model_path, load_optimizer=False)
    agent.eval()

    results = {}
    for label, mode in [("STANDARD", "standard"), ("ZERO", "zero"), ("RANDOM", "random")]:
        loguru.logger.info("=" * 50)
        loguru.logger.info(f"Running {label} ...")
        sr = np.mean(_run_condition(label, agent, envs, dataset, config, mode))
        results[label] = sr
        loguru.logger.info(f"{label} SR = {sr:.3f}")

    envs.close()

    loguru.logger.info("=" * 50)
    loguru.logger.info("ABLATION RESULTS")
    for label, sr in results.items():
        loguru.logger.info(f"  {label:10s}: {sr:.3f}")
    loguru.logger.info("=" * 50)
    if results["STANDARD"] - max(results["ZERO"], results["RANDOM"]) > 0.1:
        loguru.logger.info("CONCLUSION: Intent is load-bearing — policy uses it.")
    else:
        loguru.logger.info("CONCLUSION: Policy ignores intent — zero/random performs equally.")


if __name__ == "__main__":
    main()
