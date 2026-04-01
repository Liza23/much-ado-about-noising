"""Evaluate success rate for one or more checkpoints and save to CSV.

Usage:
    python examples/eval_success_rate.py \
        --run "baseline:checkpoints/foo.pt:task=lift_mh_state" \
        --run "flow_intent:checkpoints/bar.pt:task=lift_mh_state_flow_intent:+network.arch_variant=flow_intent" \
        --n-rollouts 100 \
        --out results/lift_mh_state_sr.csv
"""

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

from mip.torch_utils import set_seed
from mip.datasets.robomimic_dataset import make_dataset as make_dataset_robomimic
from mip.datasets.pusht_dataset import make_dataset as make_dataset_pusht
from mip.envs.robomimic.robomimic_env import make_vec_env as make_vec_env_robomimic
from mip.envs.pusht import make_vec_env as make_vec_env_pusht

def make_vec_env(task_config, seed):
    if task_config.env_name == "pusht":
        return make_vec_env_pusht(task_config)
    return make_vec_env_robomimic(task_config, seed=seed)

def make_dataset(task_config):
    if task_config.env_name == "pusht":
        return make_dataset_pusht(task_config)
    return make_dataset_robomimic(task_config)

from collect_diversity_rollouts import (
    load_config, parse_run_spec, setup_config_for_env,
    load_model, collect_rollouts,
)
from mip.samplers import get_default_step_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate success rate for multiple checkpoints")
    parser.add_argument("--run", action="append", required=True, dest="runs",
                        metavar="label:ckpt:override...",
                        help="Run spec (repeatable): label:ckpt_path:hydra_override...")
    parser.add_argument("--n-rollouts", type=int, default=100,
                        help="Number of rollout episodes per variant")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for run_spec in args.runs:
        label, ckpt_path, user_overrides = parse_run_spec(run_spec)

        if not Path(ckpt_path).exists():
            print(f"[SKIP] {label}: checkpoint not found at {ckpt_path}")
            rows.append({"variant": label, "success_rate": "N/A",
                         "n_success": "N/A", "n_total": args.n_rollouts,
                         "checkpoint": ckpt_path})
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {label}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'=' * 60}")

        overrides = user_overrides + [
            f"optimization.device={args.device}",
            "task.num_envs=1",
            f"optimization.seed={args.seed}",
        ]
        config = load_config(overrides)
        envs = make_vec_env(config.task, seed=args.seed)
        setup_config_for_env(config, envs)
        dataset = make_dataset(config.task)

        try:
            agent, intent_predictor = load_model(ckpt_path, config, dataset, args.device)
        except Exception as e:
            print(f"[ERROR] Failed to load {label}: {e}")
            envs.close()
            rows.append({"variant": label, "success_rate": "ERROR",
                         "n_success": "ERROR", "n_total": args.n_rollouts,
                         "checkpoint": ckpt_path})
            continue

        arch_variant = getattr(config.network, "arch_variant", "flow_action")
        is_fi = (arch_variant == "flow_intent")
        num_steps = int(get_default_step_list(config.optimization.loss_type)[0])

        # For non-flow_intent agents, use stochastic sampling so the comparison
        # is fair: flow_intent always draws fresh randn for intent, so baseline/
        # hier_emb should also draw fresh randn for action noise.
        if not is_fi:
            agent.config.optimization.sample_mode = "stochastic"

        print(f"  arch={arch_variant}  obs_type={config.task.obs_type}  "
              f"sample_mode={agent.config.optimization.sample_mode if not is_fi else 'randn(intent)'}  "
              f"intent_predictor={'yes' if intent_predictor else 'no'}  "
              f"num_steps={num_steps}")
        print(f"  Running {args.n_rollouts} episodes...")

        _, successes, _, _ = collect_rollouts(
            config, agent, dataset, envs,
            n_rollouts=args.n_rollouts,
            device=args.device,
            is_flow_intent=is_fi,
            intent_predictor=intent_predictor,
            num_steps=num_steps,
        )
        envs.close()

        sr = float(np.mean(successes))
        n_success = int(sum(successes))
        print(f"  → SR={sr:.1%}  ({n_success}/{len(successes)})")

        rows.append({
            "variant": label,
            "success_rate": f"{sr:.4f}",
            "n_success": n_success,
            "n_total": len(successes),
            "checkpoint": ckpt_path,
        })

    # Save CSV
    fieldnames = ["variant", "success_rate", "n_success", "n_total", "checkpoint"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {args.out}")
    print(f"{'=' * 60}")
    print(f"{'Variant':<25}  {'SR':>6}  {'N':>8}")
    print("-" * 45)
    for r in rows:
        sr_str = f"{float(r['success_rate']):.1%}" if r['success_rate'] not in ('N/A', 'ERROR') else r['success_rate']
        print(f"{r['variant']:<25}  {sr_str:>6}  {r['n_success']!s:>3}/{r['n_total']!s:<4}")


if __name__ == "__main__":
    main()
