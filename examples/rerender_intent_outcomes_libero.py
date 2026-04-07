"""Re-render saved LIBERO intent-outcome bundles with a different camera.

This keeps the same saved critical states and cluster centroids, and only
replays the fixed-intent rollouts to regenerate trajectories / rendered frames.
Useful when the original figures were produced with an undesirable render
camera (for example, LIBERO's close-up agentview).
"""

import argparse
import copy
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.chdir(ROOT)

from collect_diversity_rollouts import load_config, parse_run_spec, setup_config_for_env, load_model
from collect_intent_outcomes_image import (
    fig_eef_trajectories,
    fig_rendered_frames_filtered,
    fig_trajectories_with_frames,
    make_dataset,
    make_vec_env,
    maybe_register_libero_pro_objects,
    restore_env_obs,
    run_fixed_intent_rollout,
)


def main():
    parser = argparse.ArgumentParser(description="Re-render a saved LIBERO intent-outcomes bundle.")
    parser.add_argument("--run", required=True, help="Original run spec used to build the bundle.")
    parser.add_argument("--in-pkl", required=True, help="Input intent_outcomes.pkl path.")
    parser.add_argument("--out-pkl", required=True, help="Output pickle path.")
    parser.add_argument("--out-dir", required=True, help="Output figure directory.")
    parser.add_argument("--render-camera", default="birdview", help="LIBERO render camera name.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label, ckpt_path, overrides = parse_run_spec(args.run)
    overrides = list(overrides) + [f"task.render_camera_name={args.render_camera}"]

    config = load_config(overrides)
    config.optimization.device = args.device
    config.task.num_envs = 1
    config.task.save_video = True
    maybe_register_libero_pro_objects(config)

    with open(args.in_pkl, "rb") as f:
        result = pickle.load(f)

    envs = make_vec_env(config, seed=args.seed)
    setup_config_for_env(config, envs)
    dataset = make_dataset(config)
    agent, _ = load_model(ckpt_path, config, dataset, args.device)
    agent.eval()

    n_future_steps = int(result["n_future_steps"])
    new_outcomes = []
    for oi, outcome in enumerate(result["outcomes"]):
        print(f"Re-rendering outcome {oi + 1}/{len(result['outcomes'])} at t={outcome['timestep']}")
        new_cluster_outcomes = []
        for ci, centroid in enumerate(outcome["centroids"]):
            print(f"  cluster {ci + 1}/{len(outcome['centroids'])}  camera={args.render_camera}")
            obs_buf = restore_env_obs(envs, outcome["sim_state"], config, ensure_fresh_reset=True)
            cluster = run_fixed_intent_rollout(
                envs,
                obs_buf,
                centroid,
                agent,
                config,
                dataset,
                args.device,
                n_future_steps=n_future_steps,
                capture_render=True,
            )
            cluster["intent"] = centroid.copy()
            new_cluster_outcomes.append(cluster)

        updated = copy.deepcopy(outcome)
        updated["cluster_outcomes"] = new_cluster_outcomes
        new_outcomes.append(updated)

    out_result = dict(result)
    out_result["outcomes"] = new_outcomes
    out_result["render_camera_name"] = args.render_camera

    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(out_result, f)
    print(f"Saved pkl: {out_pkl}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"{config.task.env_name} [{label}] [{args.render_camera}]"
    fig_eef_trajectories(
        new_outcomes,
        out_dir,
        task_name,
        config.task.obs_type,
        show_intent_position=False,
    )
    fig_rendered_frames_filtered(
        new_outcomes,
        out_dir,
        task_name,
        success_only=False,
        success_threshold=0.5,
    )
    fig_trajectories_with_frames(
        new_outcomes,
        out_dir,
        task_name,
        success_only=False,
        success_threshold=0.5,
        show_intent_position=False,
    )
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
