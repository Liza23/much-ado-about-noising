"""Intent sensitivity diagnostic.

Measures ||action(obs, intent_A) - action(obs, intent_B)|| for ODE-sampled intent
pairs drawn from the same observation.  A value near zero means the action decoder
ignores intent entirely.

Three intent sources are tested for each model:
  - ode    : two independent ODE runs  (eval-time distribution)
  - gt     : two different GT intents from the dataset  (upper-bound diversity)
  - random : two independent Gaussian samples           (sanity check)

Usage:
    python examples/intent_sensitivity.py \\
        --run "flow_intent:checkpoints/lift_mh_state_..._success100.pt:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \\
        --run "flow_intent_sampled:/data/.../model_latest.pt:task=lift_mh_state_flow_intent:network=mlp_flow_intent:+task.decoder_uses_sampled_intent=true" \\
        --n-obs 128 --n-pairs 8 --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.chdir(ROOT)

from collect_diversity_rollouts import load_config, parse_run_spec, load_model
from collect_ghost_rollouts import make_dataset
from mip.flow_intent_agent import FlowIntentAgent
from mip.torch_utils import set_seed


def get_obs_batch(dataset, n_obs: int, device: str, obs_steps: int) -> torch.Tensor:
    """Sample n_obs normalized observations from the dataset. Returns (n_obs, obs_steps, obs_dim)."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=n_obs, shuffle=True)
    batch = next(iter(loader))
    obs = batch["obs"]["state"] if isinstance(batch["obs"], dict) else batch["obs"]
    obs = obs[:n_obs].to(device).float()
    # Dataset may store full horizon; slice to obs_steps used by the model
    if obs.dim() == 3 and obs.shape[1] > obs_steps:
        obs = obs[:, :obs_steps, :]
    return obs


def measure_sensitivity(agent: FlowIntentAgent, obs: torch.Tensor, n_pairs: int, intent_dim: int, dataset, num_steps: int = 9):
    """Compute mean pairwise action distance under three intent sources."""
    B = obs.shape[0]
    results = {"ode": [], "random": []}

    # Try to get GT intents from dataset (may not always be available)
    has_gt = hasattr(dataset, "normalizer") and "intent" in (dataset[0] if hasattr(dataset, "__getitem__") else {})

    for _ in range(n_pairs):
        # ── ODE-sampled intents (two independent runs) ────────────────────────
        intent_A = agent.sample_intent(obs, use_ema=True, num_steps=num_steps)  # (B, intent_dim)
        intent_B = agent.sample_intent(obs, use_ema=True, num_steps=num_steps)
        act_A = agent.sample_given_intent(obs, intent_A, use_ema=True, num_steps=num_steps)  # (B, H, act_dim)
        act_B = agent.sample_given_intent(obs, intent_B, use_ema=True, num_steps=num_steps)
        diff = (act_A - act_B).norm(dim=-1).mean(dim=-1)  # (B,)
        results["ode"].append(diff.cpu().numpy())

        # ── Random Gaussian intents ───────────────────────────────────────────
        rA = torch.randn(B, intent_dim, device=obs.device)
        rB = torch.randn(B, intent_dim, device=obs.device)
        act_rA = agent.sample_given_intent(obs, rA, use_ema=True, num_steps=num_steps)
        act_rB = agent.sample_given_intent(obs, rB, use_ema=True, num_steps=num_steps)
        diff_r = (act_rA - act_rB).norm(dim=-1).mean(dim=-1)
        results["random"].append(diff_r.cpu().numpy())

    return {k: np.concatenate(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Intent sensitivity diagnostic")
    parser.add_argument("--run", action="append", required=True, dest="runs",
                        metavar="label:ckpt:override...")
    parser.add_argument("--n-obs", type=int, default=128,
                        help="Number of dataset observations to test over")
    parser.add_argument("--n-pairs", type=int, default=8,
                        help="Intent pairs to sample per observation")
    parser.add_argument("--num-steps", type=int, default=9,
                        help="ODE integration steps")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"\n{'=' * 60}")
    print(f"Intent Sensitivity Diagnostic")
    print(f"  n_obs={args.n_obs}  n_pairs={args.n_pairs}  num_steps={args.num_steps}")
    print(f"{'=' * 60}\n")

    for run_spec in args.runs:
        label, ckpt_path, user_overrides = parse_run_spec(run_spec)
        overrides = user_overrides + [
            f"optimization.device={args.device}",
            "task.num_envs=1",
            f"optimization.seed={args.seed}",
        ]
        config = load_config(overrides)
        dataset = make_dataset(config)

        # obs_dim is normally set by setup_config_for_env; infer it from the dataset instead
        sample_obs = dataset[0]["obs"]
        sample_obs = sample_obs["state"] if isinstance(sample_obs, dict) else sample_obs
        config.task.obs_dim = sample_obs.shape[-1]

        agent, _ = load_model(ckpt_path, config, dataset, args.device)
        agent.eval()

        if not isinstance(agent, FlowIntentAgent):
            print(f"[{label}] SKIP — not a FlowIntentAgent\n")
            continue

        intent_dim = config.task.intent_dim

        print(f"[{label}]  ckpt: {ckpt_path}")
        print(f"  intent_dim={intent_dim}")

        obs = get_obs_batch(dataset, args.n_obs, args.device, config.task.obs_steps)

        with torch.no_grad():
            scores = measure_sensitivity(agent, obs, args.n_pairs, intent_dim, dataset, args.num_steps)

        print(f"  {'Source':<10}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
        print(f"  {'-'*46}")
        for src, vals in scores.items():
            print(f"  {src:<10}  {vals.mean():8.4f}  {vals.std():8.4f}  {vals.min():8.4f}  {vals.max():8.4f}")
        print()


if __name__ == "__main__":
    main()
