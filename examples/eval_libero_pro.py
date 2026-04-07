"""Evaluate a trained MIP policy on LIBERO-PRO spatial benchmark.

Runs the policy across all 10 tasks of one or more LIBERO-PRO suites using
fixed initial states (*.pruned_init files) for reproducible evaluation.

Usage:
    # Evaluate on original spatial suite (P0 — baseline)
    python examples/eval_libero_pro.py \\
        task=libero_spatial_state \\
        task.dataset_path=~/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \\
        eval.checkpoint=checkpoints/libero_spatial_state_state_flow_mlp_512_seed0_success42.pt \\
        eval.suite=libero_spatial

    # Evaluate on task perturbation (P1)
    python examples/eval_libero_pro.py \\
        task=libero_spatial_state \\
        task.dataset_path=... \\
        eval.checkpoint=checkpoints/... \\
        eval.suite=libero_spatial_with_mug

    # Evaluate on all spatial suites
    python examples/eval_libero_pro.py \\
        task=libero_spatial_state \\
        task.dataset_path=... \\
        eval.checkpoint=checkpoints/... \\
        eval.suites=[libero_spatial,libero_spatial_with_mug,libero_spatial_with_diffpos_stick]

Available spatial suites (P0=original, P1=task perturbation, P2=position perturbation):
    P0: libero_spatial
    P1: libero_spatial_with_mug, libero_spatial_with_red_stick,
        libero_spatial_with_yellow_book, libero_spatial_with_blue_stick,
        libero_spatial_with_green_mug, libero_spatial_with_milk,
        libero_spatial_with_red_box
    P2: libero_spatial_with_diffpos_stick
"""

import csv
import os
import sys

import hydra
import loguru
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from mip.agent import TrainingAgent
from mip.config import Config
from mip.datasets.libero_dataset import make_dataset
from mip.envs.libero.libero_env_wrapper import LiberoGymWrapper
from mip.env_utils import MultiStepWrapper
from mip.flow_intent_agent import FlowIntentAgent
from mip.samplers import get_default_step_list
from mip.torch_utils import set_seed

torch.set_float32_matmul_precision("high")

# All spatial LIBERO-PRO suites (P0 + P1 + P2)
SPATIAL_SUITES = [
    "libero_spatial",                     # P0 — original
    "libero_spatial_with_mug",            # P1 — task perturbation
    "libero_spatial_with_red_stick",      # P1
    "libero_spatial_with_yellow_book",    # P1
    "libero_spatial_with_blue_stick",     # P1
    "libero_spatial_with_green_mug",      # P1
    "libero_spatial_with_milk",           # P1
    "libero_spatial_with_red_box",        # P1
    "libero_spatial_with_diffpos_stick",  # P2 — position perturbation
]


def _get_suite_tasks(suite: str, libero_pro_root: str):
    """Return list of (bddl_path, init_path) for all tasks in a suite."""
    bddl_dir = os.path.join(libero_pro_root, "libero", "libero", "bddl_files", suite)
    init_dir = os.path.join(libero_pro_root, "libero", "libero", "init_files", suite)

    if not os.path.isdir(bddl_dir):
        raise FileNotFoundError(f"BDDL directory not found: {bddl_dir}")
    if not os.path.isdir(init_dir):
        raise FileNotFoundError(f"Init files directory not found: {init_dir}")

    bddl_files = sorted(f for f in os.listdir(bddl_dir) if f.endswith(".bddl"))
    tasks = []
    for bddl_fname in bddl_files:
        stem = bddl_fname[: -len(".bddl")]
        init_fname = stem + ".pruned_init"
        init_path = os.path.join(init_dir, init_fname)
        if not os.path.exists(init_path):
            loguru.logger.warning(f"No init file for {stem}, skipping.")
            continue
        tasks.append((stem, os.path.join(bddl_dir, bddl_fname), init_path))

    return tasks


def evaluate_suite(config, dataset, agent, suite: str, libero_pro_root: str,
                   n_init_states: int, num_steps: int,
                   task_name_filter: str | None = None,
                   arch_variant: str = "flow_action"):
    """Evaluate agent on all tasks in a LIBERO-PRO suite.

    Returns dict of task_name -> success_rate.
    """
    obs_type = getattr(config.task, "obs_type", "state")
    image_obs_keys = list(getattr(config.task, "image_obs_keys", None) or [])

    tasks = _get_suite_tasks(suite, libero_pro_root)
    if task_name_filter:
        tasks = [t for t in tasks if task_name_filter in t[0]]
    if not tasks:
        loguru.logger.error(f"No tasks found for suite: {suite}")
        return {}

    loguru.logger.info(f"\n{'='*60}")
    loguru.logger.info(f"Suite: {suite}  ({len(tasks)} tasks, {n_init_states} init states each)")
    loguru.logger.info(f"{'='*60}")

    suite_results = {}
    for task_name, bddl_path, init_path in tasks:
        # Load fixed init states
        init_states = torch.load(init_path, map_location="cpu", weights_only=False)
        init_states = init_states[:n_init_states]

        # Build single-env wrapper
        env = LiberoGymWrapper(
            bddl_file=bddl_path,
            obs_keys=config.task.obs_keys,
            obs_type=obs_type,
            image_obs_keys=image_obs_keys,
        )
        env = MultiStepWrapper(
            env,
            n_obs_steps=config.task.obs_steps,
            n_action_steps=config.task.act_steps,
            max_episode_steps=config.task.max_episode_steps,
        )

        successes = []
        for init_state in init_states:
            # Reset to fixed initial state
            obs, _ = env.reset()
            # Override with fixed init state (access underlying LiberoGymWrapper)
            underlying = env.env  # MultiStepWrapper.env
            obs_init, _ = underlying.set_init_state(init_state)
            # Re-stack obs_steps copies of the initial obs for history
            if obs_type == "image":
                obs = {k: np.stack([obs_init[k]] * config.task.obs_steps, axis=0)
                       for k in obs_init}
            else:
                obs = np.stack([obs_init] * config.task.obs_steps, axis=0)

            success = False
            t = 0
            while t < config.task.max_episode_steps:
                if obs_type == "image":
                    state_normed = dataset.normalizer["obs"]["state"].normalize(
                        obs["state"].astype(np.float32)
                    )
                    obs_tensor = {"state": torch.tensor(
                        state_normed, device=config.optimization.device, dtype=torch.float32
                    ).unsqueeze(0)}
                    for img_key in image_obs_keys:
                        obs_tensor[img_key] = torch.tensor(
                            obs[img_key].astype(np.float32),
                            device=config.optimization.device, dtype=torch.float32
                        ).unsqueeze(0)
                    from tensordict import TensorDict
                    obs_in = TensorDict(obs_tensor, batch_size=1)
                else:
                    obs_normed = dataset.normalizer["obs"]["state"].normalize(
                        obs.astype(np.float32)
                    )
                    obs_state = torch.tensor(
                        obs_normed, device=config.optimization.device, dtype=torch.float32
                    ).unsqueeze(0)

                with torch.no_grad():
                    if arch_variant == "flow_intent":
                        act_normed = agent.sample(
                            obs=obs_in if obs_type == "image" else obs_state,
                            num_steps=num_steps,
                            use_ema=True,
                        )
                    else:
                        act_0 = torch.randn(
                            (1, config.task.horizon, config.task.act_dim),
                            device=config.optimization.device,
                        )
                        act_normed = agent.sample(
                            act_0=act_0,
                            obs=obs_in if obs_type == "image" else {"state": obs_state},
                            num_steps=num_steps,
                            use_ema=True,
                        )

                act_normed = act_normed.detach().cpu().numpy()
                act = dataset.normalizer["action"].unnormalize(act_normed)

                start = config.task.obs_steps - 1
                end = start + config.task.act_steps
                act_exec = act[0, start:end, :]  # (act_steps, act_dim)

                # Step through each action
                for a in act_exec:
                    obs, reward, terminated, truncated, info = env.step(a[None])
                    t += 1
                    s = info.get("success", False)
                    if isinstance(s, (list, np.ndarray)):
                        s = bool(np.asarray(s).any())
                    if s:
                        success = True
                    if terminated or truncated or success:
                        break
                if success or terminated or truncated:
                    break

            successes.append(float(success))

        sr = float(np.mean(successes))
        short_name = task_name.split("_pick_")[1] if "_pick_" in task_name else task_name
        loguru.logger.info(f"  {short_name[:60]:60s}  SR={sr:.2f}")
        suite_results[task_name] = sr
        env.close()

    mean_sr = float(np.mean(list(suite_results.values())))
    loguru.logger.info(f"  {'SUITE MEAN':60s}  SR={mean_sr:.2f}")
    return suite_results


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config: Config):
    set_seed(config.optimization.seed)

    eval_cfg = config.get("eval", {})
    checkpoint = eval_cfg.get("checkpoint", None)
    suites = eval_cfg.get("suites", None) or [eval_cfg.get("suite", "libero_spatial")]
    libero_pro_root = os.path.expanduser(
        eval_cfg.get("libero_pro_root", "~/LIBERO-PRO")
    )
    n_init_states = int(eval_cfg.get("n_init_states", 20))
    num_steps_arg = eval_cfg.get("num_steps", None)
    task_name_filter = eval_cfg.get("task_name_filter", None)
    out_csv = eval_cfg.get("out_csv", None)

    if not checkpoint:
        raise ValueError("eval.checkpoint must be provided.")

    # Build a single env to resolve obs_dim at runtime
    obs_type = getattr(config.task, "obs_type", "state")
    image_obs_keys = list(getattr(config.task, "image_obs_keys", None) or [])

    # Use the first bddl file of the first suite to init obs shape
    first_suite = suites[0]
    tasks = _get_suite_tasks(first_suite, libero_pro_root)
    if not tasks:
        raise RuntimeError(f"No tasks found in suite {first_suite}")

    _, first_bddl, _ = tasks[0]
    probe_env = LiberoGymWrapper(
        bddl_file=first_bddl,
        obs_keys=config.task.obs_keys,
        obs_type=obs_type,
        image_obs_keys=image_obs_keys,
    )
    probe_obs, _ = probe_env.reset()
    if obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    else:
        config.task.obs_dim = probe_obs.shape[-1]
    probe_env.close()
    loguru.logger.info(f"obs_dim={config.task.obs_dim}")

    # Build agent and load checkpoint
    arch_variant = (
        getattr(config.task, "arch_variant", None)
        or getattr(config.network, "arch_variant", "flow_action")
    )
    if arch_variant == "flow_intent":
        agent = FlowIntentAgent(config)
    else:
        agent = TrainingAgent(config)

    loguru.logger.info(f"Loading checkpoint: {checkpoint}")
    agent.load(checkpoint)
    agent.eval()

    # Build dataset for normalizer (uses task.dataset_path / dataset_paths)
    dataset = make_dataset(config.task, mode="train")

    num_steps_list = (
        [int(num_steps_arg)] if num_steps_arg
        else get_default_step_list(config.optimization.loss_type)
    )

    all_results = {}
    for num_steps in num_steps_list:
        loguru.logger.info(f"\n{'#'*60}")
        loguru.logger.info(f"ODE steps = {num_steps}")
        loguru.logger.info(f"{'#'*60}")
        step_results = {}
        for suite in suites:
            sr_dict = evaluate_suite(
                config, dataset, agent, suite, libero_pro_root,
                n_init_states, num_steps,
                task_name_filter=task_name_filter,
                arch_variant=arch_variant,
            )
            step_results[suite] = sr_dict
        all_results[num_steps] = step_results

    # Final summary
    loguru.logger.info("\n" + "="*60)
    loguru.logger.info("FINAL SUMMARY")
    loguru.logger.info("="*60)
    for num_steps, step_results in all_results.items():
        loguru.logger.info(f"\nODE steps={num_steps}:")
        for suite, sr_dict in step_results.items():
            mean_sr = float(np.mean(list(sr_dict.values()))) if sr_dict else 0.0
            loguru.logger.info(f"  {suite}: mean_SR={mean_sr:.3f}")

    if out_csv:
        out_csv = os.path.expanduser(out_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        rows = []
        for num_steps, step_results in all_results.items():
            for suite, sr_dict in step_results.items():
                for task_name, success_rate in sr_dict.items():
                    rows.append({
                        "num_steps": num_steps,
                        "suite": suite,
                        "task_name": task_name,
                        "success_rate": f"{float(success_rate):.4f}",
                        "n_init_states": n_init_states,
                        "checkpoint": checkpoint,
                    })
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "num_steps",
                    "suite",
                    "task_name",
                    "success_rate",
                    "n_init_states",
                    "checkpoint",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        loguru.logger.info(f"Saved CSV summary to {out_csv}")


if __name__ == "__main__":
    main()
