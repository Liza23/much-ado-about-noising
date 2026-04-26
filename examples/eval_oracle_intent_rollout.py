"""Oracle intent rollout eval: does the decoder fail at rollout time?

Runs two sets of live LIBERO episodes:
  1. STANDARD: agent.sample(obs) — ODE-sampled intent → action
  2. ORACLE:   peek ahead in sim with zero actions, compute GT future eef,
               inject as intent → agent.sample_given_intent(obs, gt_intent)

If oracle SR >> standard SR  → intent ODE fails on off-distribution rollout obs
If oracle SR ≈  standard SR  → decoder itself is the rollout bottleneck

Usage:
    python examples/eval_oracle_intent_rollout.py \\
        task=libero_10_suite_image_flow_intent \\
        network=mlp_flow_intent \\
        optimization.loss_type=flow \\
        optimization.model_path=/path/to/model_best.pt \\
        ++eval_oracle.num_episodes=20 \\
        ++eval_oracle.num_steps=9
"""

import os

import hydra
import loguru
import numpy as np
import torch
from omegaconf import OmegaConf

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.config import Config
from mip.datasets.libero_dataset import make_dataset
from mip.envs.libero import make_vec_env
from mip.flow_intent_agent import FlowIntentAgent
from mip.torch_utils import set_seed


def _get_gt_intent(eval_envs, act_steps: int, eef_normalizer, device: str) -> torch.Tensor:
    """Peek ahead act_steps with zero actions to get GT future eef intent.

    Saves and restores each env's sim state so the rollout is unaffected.
    Returns (num_envs, intent_dim) normalized intent tensor.
    """
    num_envs = len(eval_envs.envs)
    # eval_envs.envs[i] is MultiStepWrapper; .env is LiberoGymWrapper; ._env is ControlEnv
    act_dim = eval_envs.envs[0].env._env.env.action_dim
    zero_action = np.zeros(act_dim, dtype=np.float32)

    all_intents = []
    for multi_step_wrapper in eval_envs.envs:
        libero_wrapper = multi_step_wrapper.env   # LiberoGymWrapper
        inner = libero_wrapper._env               # ControlEnv
        sim_state = inner.get_sim_state()         # save

        future_eefs = []
        for _ in range(act_steps):
            raw_obs, _, done, _ = inner.step(zero_action)
            ee = libero_wrapper._extract_state_key(raw_obs, "ee_states")  # (6,)
            future_eefs.append(ee)
            if done:
                break

        inner.set_state(sim_state)        # restore
        inner.env.sim.forward()

        future_eef_arr = np.stack(future_eefs, axis=0)  # (T, 6)
        all_intents.append(future_eef_arr.mean(axis=0))  # (6,)

    raw_intent = np.stack(all_intents, axis=0)           # (num_envs, 6)
    normed = eef_normalizer.normalize(raw_intent)        # (num_envs, 6)
    return torch.tensor(normed, device=device, dtype=torch.float32)


def _run_episodes(config, eval_envs, dataset, agent, num_steps, use_oracle, image_obs_keys, num_episodes=None):
    """Run eval episodes. Returns list of per-episode success flags."""
    from tensordict import TensorDict

    device = config.optimization.device
    obs_steps = config.task.obs_steps
    act_steps = config.task.act_steps
    num_envs = config.task.num_envs

    eef_normalizer = dataset.normalizer["eef"] if use_oracle else None

    if num_episodes is None:
        num_episodes = getattr(
            OmegaConf.select(config, "eval_oracle", default=None), "num_episodes", 20
        )
    num_batches = max(1, num_episodes // num_envs)

    all_success = []
    for batch_idx in range(num_batches):
        obs, _ = eval_envs.reset()
        success = [0] * num_envs
        t = 0

        while t < config.task.max_episode_steps:
            # Build obs tensor
            obs_dict = {}
            obs_dict["state"] = torch.tensor(
                dataset.normalizer["obs"]["state"].normalize(
                    obs["state"].astype(np.float32)
                ),
                device=device, dtype=torch.float32,
            )
            for img_key in image_obs_keys:
                obs_dict[img_key] = torch.tensor(
                    obs[img_key].astype(np.float32),
                    device=device, dtype=torch.float32,
                )
            obs_tensor = TensorDict(obs_dict, batch_size=num_envs)

            with torch.no_grad():
                if use_oracle:
                    gt_intent = _get_gt_intent(
                        eval_envs, act_steps, eef_normalizer, device
                    )
                    act_normed = agent.sample_given_intent(
                        obs=obs_tensor, intent_vec=gt_intent,
                        use_ema=True, num_steps=num_steps,
                    )
                else:
                    act_normed = agent.sample(
                        obs=obs_tensor, use_ema=True, num_steps=num_steps,
                    )

            act_normed = act_normed.detach().cpu().numpy()
            act = dataset.normalizer["action"].unnormalize(act_normed)
            start = obs_steps - 1
            act = act[:, start:start + act_steps, :]

            obs, reward, terminated, truncated, info = eval_envs.step(act)
            t += act_steps

            if "_final_info" in info:
                for i in range(num_envs):
                    if info["_final_info"][i]:
                        fi = info["final_info"][i]
                        if fi and "success" in fi:
                            success[i] = max(success[i], int(bool(np.asarray(fi["success"]).any())))
            if "success" in info:
                for i in range(num_envs):
                    s = info["success"][i] if hasattr(info["success"], "__len__") else info["success"]
                    success[i] = max(success[i], int(bool(np.asarray(s).any())))

        all_success.extend(success)
        label = "ORACLE" if use_oracle else "STANDARD"
        loguru.logger.info(
            f"[{label}] batch {batch_idx+1}/{num_batches} done | "
            f"running SR={np.mean(all_success):.3f} ({sum(all_success)}/{len(all_success)})"
        )

    return all_success


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    set_seed(config.optimization.seed)

    arch_variant = (
        getattr(config.task, "arch_variant", None)
        or getattr(config.network, "arch_variant", "flow_action")
    )
    assert arch_variant == "flow_intent", (
        f"This script only works for flow_intent models, got arch_variant={arch_variant}"
    )

    _oracle_cfg = OmegaConf.select(config, "eval_oracle", default=None)
    num_steps = int(getattr(_oracle_cfg, "num_steps", 9) if _oracle_cfg else 9)
    num_episodes = int(getattr(_oracle_cfg, "num_episodes", 20) if _oracle_cfg else 20)

    loguru.logger.info(f"Oracle rollout eval: num_episodes={num_episodes}, num_steps={num_steps}")

    obs_type = getattr(config.task, "obs_type", "state")
    assert obs_type == "image", "This script targets image-obs flow_intent models."

    bddl_files = list(getattr(config.task, "bddl_files", None) or [])
    if not bddl_files and config.task.bddl_file:
        bddl_files = [config.task.bddl_file]

    # Use num_envs=1 so SyncVectorEnv gives direct env access for sim state save/restore
    config.task.num_envs = 1

    # Probe obs_dim using the first task
    config.task.bddl_file = bddl_files[0]
    eval_envs = make_vec_env(config.task, seed=config.optimization.seed)
    eval_envs.reset()
    config.task.obs_dim = config.network.emb_dim
    eval_envs.close()

    image_obs_keys = list(getattr(config.task, "image_obs_keys", ["agentview_rgb", "eye_in_hand_rgb"]))

    dataset = make_dataset(config.task)
    assert dataset.intent_conditioning, "Dataset must have intent_conditioning=True"
    loguru.logger.info(f"Dataset size: {len(dataset)}")

    agent = FlowIntentAgent(config)
    model_path = getattr(config.optimization, "model_path", None)
    assert model_path and model_path != "None", "Provide optimization.model_path=..."
    loguru.logger.info(f"Loading checkpoint: {model_path}")
    agent.load(model_path, load_optimizer=False)
    agent.eval()

    episodes_per_task = max(1, num_episodes // len(bddl_files))
    loguru.logger.info(
        f"Evaluating {len(bddl_files)} tasks × {episodes_per_task} episodes each "
        f"= {len(bddl_files) * episodes_per_task} total episodes"
    )

    results = {}
    for use_oracle in [False, True]:
        label = "oracle" if use_oracle else "standard"
        loguru.logger.info("=" * 60)
        loguru.logger.info(f"Running {label.upper()} episodes...")
        all_success = []
        for task_idx, bddl_file in enumerate(bddl_files):
            task_name = os.path.splitext(os.path.basename(bddl_file))[0]
            config.task.bddl_file = bddl_file
            eval_envs = make_vec_env(config.task, seed=config.optimization.seed)
            task_success = _run_episodes(
                config, eval_envs, dataset, agent, num_steps, use_oracle, image_obs_keys,
                num_episodes=episodes_per_task,
            )
            eval_envs.close()
            task_sr = np.mean(task_success)
            loguru.logger.info(
                f"  [{label.upper()}] task {task_idx+1}/{len(bddl_files)} "
                f"{task_name}: SR={task_sr:.3f}"
            )
            all_success.extend(task_success)
        results[label] = np.mean(all_success)
        loguru.logger.info(f"{label.upper()} SR = {results[label]:.3f} (avg over {len(bddl_files)} tasks)")

    loguru.logger.info("=" * 60)
    loguru.logger.info("ORACLE ROLLOUT RESULTS")
    loguru.logger.info(f"  Standard SR (ODE intent)  : {results['standard']:.3f}")
    loguru.logger.info(f"  Oracle SR   (GT intent)   : {results['oracle']:.3f}")
    delta = results["oracle"] - results["standard"]
    loguru.logger.info(f"  Delta                     : {delta:+.3f}")
    loguru.logger.info("=" * 60)
    if delta > 0.15:
        loguru.logger.info(
            "CONCLUSION: Oracle SR >> Standard SR -> intent ODE fails on rollout obs "
            "(off-distribution). Intent predictor is the rollout bottleneck."
        )
    elif delta < -0.05:
        loguru.logger.info(
            "CONCLUSION: Oracle SR <= Standard SR -> GT intent doesn't help at rollout. "
            "Decoder itself is the rollout bottleneck (mode averaging / compounding errors)."
        )
    else:
        loguru.logger.info(
            "CONCLUSION: Similar SRs -> neither intent ODE nor decoder clearly explains the gap. "
            "Look at action compounding, task diversity, or training data coverage."
        )


if __name__ == "__main__":
    main()
