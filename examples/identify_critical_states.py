"""Identify critical vs redundant states via observation noise injection.

For each timestep in reference rollouts, adds Gaussian noise to the policy's
(normalized) observation and re-runs the episode forward. States where noising
causes irreversible failure are "critical"; states where the policy self-corrects
are "redundant".

Algorithm
---------
Phase 1 — Reference rollouts
  Run N episodes with the policy. At every chunk boundary save:
    sim_state  : full MuJoCo qpos/qvel (for env restore)
    obs_buf    : raw (unnormalized) obs buffer (1, obs_steps, obs_dim)
    timestep   : episode step counter
    eef_pos    : EEF xyz for visualization

Phase 2 — Noisy re-rollouts
  For each saved (sim_state, obs_buf, t), repeat n_noisy times:
    1. Restore env to sim_state
    2. Preprocess obs_buf → normalized obs tensor via to_fi_obs
    3. Inject Gaussian noise:  fi_obs_noisy = fi_obs + randn(same shape) * noise_std
    4. Sample ONE action chunk from the policy using the noisy obs
    5. Execute that action chunk in the real env
    6. Continue rollout for the rest of n_future_steps with CLEAN observations
    7. Record success / failure

Noise magnitude (noise_std) is in the normalized observation space (approx [-1,1]).
Typical values: 0.1 (subtle), 0.3 (moderate), 0.5 (strong).

Output pkl
----------
{
  "noise_std": float,
  "n_noisy": int,
  "n_future_steps": int,
  "label": str,
  "states": [
    {
      "episode": int,
      "timestep": int,
      "eef_pos": np.ndarray (3,),
      "sim_state": np.ndarray,        # full MuJoCo state
      "n_noisy": int,
      "n_success": int,
      "recovery_rate": float,         # n_success / n_noisy
      "outcomes": list[bool],         # per noisy rollout
    },
    ...
  ]
}

Usage
-----
    # Baseline BC model
    python examples/identify_critical_states.py \\
        --run "baseline:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0.pt:task=lift_mh_state" \\
        --n-rollouts 20 --n-noisy 10 --noise-std 0.3 --n-future-steps 100 \\
        --device cuda --out rollouts/critical_states_lift_mh_baseline.pkl

    # Flow-intent model
    python examples/identify_critical_states.py \\
        --run "flow_intent:checkpoints/lift_mh_fi.pt:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \\
        --n-rollouts 20 --n-noisy 10 --noise-std 0.3 --n-future-steps 100 \\
        --device cuda --out rollouts/critical_states_lift_mh_fi.pkl

    # Analyze saved results
    python examples/identify_critical_states.py --analyze-only \\
        --out rollouts/critical_states_lift_mh_baseline.pkl \\
        --out-dir rollouts/critical_states_figs
"""

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

from collect_intent_outcomes_image import (
    BaselineAgentAdapter,
    make_vec_env,
    make_dataset,
    to_fi_obs,
    get_sim_state,
    restore_env_obs,
    update_obs_buf,
    get_eef_from_envs,
    undo_action,
    _get_inner_step_env,
    _step_inner,
    maybe_register_libero_pro_objects,
)
from collect_diversity_rollouts import (
    load_config,
    parse_run_spec,
    setup_config_for_env,
    load_model,
    ResidualPARLWrapper,
    DSRLWrapper,
    PlainDSRLWrapper,
    ResidualSACWrapper,
)
from mip.agent import TrainingAgent
from mip.flow_intent_agent import FlowIntentAgent
from mip.torch_utils import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Reference rollout: collect (sim_state, obs_buf, t) at each chunk boundary
# ─────────────────────────────────────────────────────────────────────────────

def collect_reference_states(config, agent, dataset, envs, args, device):
    """Run n_rollouts episodes and save env state at every chunk boundary.

    Saves all timesteps that have at least one act_steps chunk remaining.
    The noisy re-rollout uses the full remaining horizon (max_episode_steps - t),
    so there is no need to filter by n_future_steps here.
    Returns a list of dicts: {obs_buf, sim_state, timestep, episode, eef_pos}.
    """
    num_steps = 9
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    n_episodes = 0
    saved = []

    print(f"\nPhase 1: collecting reference states from {args.n_rollouts} episodes ...")

    while n_episodes < args.n_rollouts:
        obs, _ = envs.reset()
        t = 0
        ep_idx = n_episodes

        while t < config.task.max_episode_steps:
            # Save every chunk boundary that has at least one chunk remaining
            if t + act_steps <= config.task.max_episode_steps:
                sim_state = get_sim_state(envs, config)
                obs_snap = obs.copy() if config.task.obs_type == "state" else {
                    k: v.copy() for k, v in obs.items()
                }
                eef_pos = get_eef_from_envs(envs, config)
                saved.append({
                    "obs_buf": obs_snap,
                    "sim_state": sim_state,
                    "timestep": t,
                    "episode": ep_idx,
                    "eef_pos": eef_pos,
                })

            # Advance reference rollout one chunk
            fi_obs, _ = to_fi_obs(obs, config, dataset, device)
            with torch.no_grad():
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
            act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
            inner = _get_inner_step_env(envs, config)
            done = False
            for a_i in range(act_steps):
                act = undo_action(act_un[0, s + a_i], config, dataset)
                _, done, _ = _step_inner(inner, act)
                if done:
                    break
            obs = update_obs_buf(obs, envs, config)
            t += act_steps
            if done:
                break

        n_episodes += config.task.num_envs
        print(f"  Episode {n_episodes}/{args.n_rollouts}  |  saved so far: {len(saved)}")

    print(f"Phase 1 done. Total candidate states: {len(saved)}\n")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Noisy re-rollout: inject noise at step 0, run clean afterwards
# ─────────────────────────────────────────────────────────────────────────────

def run_noisy_rollout(
    envs, obs_buf, agent, config, dataset, device, noise_std, n_future_steps, num_steps=9,
):
    """Restore env, inject obs noise for the FIRST chunk only, then roll out cleanly.

    Returns dict with keys: success (bool), n_steps (int).
    """
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    done = False
    info = {}
    steps_run = 0

    # ── Chunk 0: noisy observation ────────────────────────────────────────────
    fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
    # noise injected in normalized obs space
    noise = torch.randn_like(fi_obs) * noise_std
    fi_obs_noisy = fi_obs + noise

    with torch.no_grad():
        if isinstance(agent, (FlowIntentAgent, ResidualPARLWrapper, DSRLWrapper, ResidualSACWrapper)):
            act_norm = agent.sample(obs=fi_obs_noisy, use_ema=True, num_steps=num_steps)
        else:
            act_norm = agent.sample(obs=fi_obs_noisy, use_ema=True, num_steps=num_steps)

    act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
    remaining = n_future_steps
    for a_i in range(min(act_steps, remaining)):
        act = undo_action(act_un[0, s + a_i], config, dataset)
        _, done, info = _step_inner(inner, act)
        steps_run += 1
        if done:
            break
    obs_buf = update_obs_buf(obs_buf, envs, config)
    remaining -= act_steps

    # ── Subsequent chunks: clean observations ─────────────────────────────────
    while not done and remaining > 0:
        fi_obs_clean, _ = to_fi_obs(obs_buf, config, dataset, device)
        with torch.no_grad():
            act_norm = agent.sample(obs=fi_obs_clean, use_ema=True, num_steps=num_steps)
        act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
        for a_i in range(min(act_steps, remaining)):
            act = undo_action(act_un[0, s + a_i], config, dataset)
            _, done, info = _step_inner(inner, act)
            steps_run += 1
            if done:
                break
        obs_buf = update_obs_buf(obs_buf, envs, config)
        remaining -= act_steps

    return {
        "success": bool(info.get("success", False)),
        "n_steps": steps_run,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: noisy re-rollout per saved state
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_critical_states(ref_states, envs, agent, config, dataset, device, args):
    """For each saved state, run n_noisy noisy re-rollouts and record outcomes."""
    results = []
    n_states = len(ref_states)
    print(f"Phase 2: {n_states} states × {args.n_noisy} noisy rollouts each ...\n")

    for i, state in enumerate(ref_states):
        outcomes = []
        # Give the policy the full remaining horizon to recover
        remaining_steps = config.task.max_episode_steps - state["timestep"]
        for k in range(args.n_noisy):
            obs_buf = restore_env_obs(envs, state["sim_state"], config)
            result = run_noisy_rollout(
                envs, obs_buf, agent, config, dataset, device,
                noise_std=args.noise_std,
                n_future_steps=remaining_steps,
            )
            outcomes.append(result["success"])

        n_success = sum(outcomes)
        recovery_rate = n_success / args.n_noisy
        results.append({
            "episode": state["episode"],
            "timestep": state["timestep"],
            "eef_pos": state["eef_pos"],
            "sim_state": state["sim_state"],
            "n_noisy": args.n_noisy,
            "n_success": n_success,
            "recovery_rate": recovery_rate,
            "outcomes": outcomes,
        })

        if (i + 1) % max(1, n_states // 20) == 0 or i == n_states - 1:
            print(
                f"  [{i + 1:4d}/{n_states}]  "
                f"ep={state['episode']:3d}  t={state['timestep']:4d}  "
                f"recovery={recovery_rate:.2f}  "
                f"({'critical' if recovery_rate < 0.5 else 'redundant'})"
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis & plotting
# ─────────────────────────────────────────────────────────────────────────────

def analyze_and_plot(data, out_dir, label="policy"):
    """Generate summary figures from a saved pkl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states = data["states"]
    noise_std = data["noise_std"]
    n_noisy = data["n_noisy"]
    label = data.get("label", label)

    timesteps = np.array([s["timestep"] for s in states])
    recovery = np.array([s["recovery_rate"] for s in states])
    is_critical = recovery < 0.5

    # ── 1. Recovery rate vs timestep ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(
        timesteps, recovery,
        c=recovery, cmap="RdYlGn", vmin=0, vmax=1,
        s=20, alpha=0.6, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Recovery rate")
    # rolling mean
    if len(timesteps) > 10:
        sort_idx = np.argsort(timesteps)
        ts_s = timesteps[sort_idx]
        rv_s = recovery[sort_idx]
        window = max(5, len(ts_s) // 20)
        rm = np.convolve(rv_s, np.ones(window) / window, mode="valid")
        ax.plot(ts_s[window // 2: window // 2 + len(rm)], rm, "k-", lw=2, label=f"Rolling mean (w={window})")
        ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="50% threshold")
    ax.set_xlabel("Episode timestep")
    ax.set_ylabel("Recovery rate after noise injection")
    ax.set_title(f"{label}  |  noise_std={noise_std}  n_noisy={n_noisy}")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_vs_timestep.png", dpi=150)
    plt.close(fig)

    # ── 2. EEF position scatter colored by criticality ───────────────────────
    eef = np.array([s["eef_pos"] for s in states])
    if eef.shape[1] >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (xi, yi, xl, yl) in zip(
            axes,
            [(0, 1, "EEF x", "EEF y"), (0, 2, "EEF x", "EEF z")],
        ):
            sc = ax.scatter(
                eef[:, xi], eef[:, yi],
                c=recovery, cmap="RdYlGn", vmin=0, vmax=1,
                s=20, alpha=0.7, edgecolors="none",
            )
            plt.colorbar(sc, ax=ax, label="Recovery rate")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
        fig.suptitle(f"{label}  —  EEF position vs recovery rate  (noise_std={noise_std})", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "eef_recovery_scatter.png", dpi=150)
        plt.close(fig)

    # ── 3. Recovery rate distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(recovery, bins=20, range=(0, 1), color="#4C72B0", edgecolor="white", lw=0.5)
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="50% threshold")
    ax.set_xlabel("Recovery rate")
    ax.set_ylabel("Count")
    n_crit = is_critical.sum()
    ax.set_title(
        f"{label}  |  noise_std={noise_std}\n"
        f"critical: {n_crit}/{len(states)} ({100*n_crit/len(states):.0f}%)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_distribution.png", dpi=150)
    plt.close(fig)

    # ── 4. Mean recovery by timestep bin ─────────────────────────────────────
    if len(timesteps) > 10:
        bins = np.arange(0, timesteps.max() + 16, 16)  # 16-step bins (2 chunks)
        bin_means = []
        bin_stds = []
        bin_centers = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (timesteps >= lo) & (timesteps < hi)
            if mask.sum() > 0:
                bin_means.append(recovery[mask].mean())
                bin_stds.append(recovery[mask].std())
                bin_centers.append((lo + hi) / 2)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        bin_centers = np.array(bin_centers)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(bin_centers, bin_means, width=14, color="#4C72B0", alpha=0.7, label="Mean recovery")
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="none", color="k", capsize=3, lw=1.5)
        ax.axhline(0.5, color="red", ls="--", lw=1, label="50% threshold")
        ax.set_xlabel("Episode timestep (binned)")
        ax.set_ylabel("Mean recovery rate")
        ax.set_title(f"{label}  —  Recovery by episode phase  (noise_std={noise_std})")
        ax.set_ylim(0, 1.1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "recovery_by_timestep_bin.png", dpi=150)
        plt.close(fig)

    # ── 5. Print summary ─────────────────────────────────────────────────────
    n_crit = is_critical.sum()
    n_red = (~is_critical).sum()
    print("\n" + "=" * 60)
    print(f"SUMMARY  |  {label}  |  noise_std={noise_std}  n_noisy={n_noisy}")
    print("=" * 60)
    print(f"Total states evaluated : {len(states)}")
    print(f"Critical (recovery<50%): {n_crit} ({100*n_crit/len(states):.1f}%)")
    print(f"Redundant (recovery≥50%): {n_red} ({100*n_red/len(states):.1f}%)")
    print(f"Mean recovery rate      : {recovery.mean():.3f} ± {recovery.std():.3f}")
    print(f"Timestep range          : [{timesteps.min()}, {timesteps.max()}]")

    # Top 5 critical states
    crit_idx = np.argsort(recovery)[:5]
    print("\nTop 5 most critical states (lowest recovery):")
    for idx in crit_idx:
        s = states[idx]
        print(
            f"  ep={s['episode']:3d}  t={s['timestep']:4d}  "
            f"recovery={s['recovery_rate']:.2f}  eef={s['eef_pos'].round(3)}"
        )

    print(f"\nFigures saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Identify critical vs redundant states via observation noise injection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        required=False,
        metavar="label:ckpt_path[:override...]",
        help=(
            "Run spec: 'label:ckpt_path[:hydra_override ...]'. "
            "Required unless --analyze-only is set."
        ),
    )
    parser.add_argument(
        "--n-rollouts", type=int, default=20,
        help="Number of reference episodes to collect states from (default: 20)",
    )
    parser.add_argument(
        "--n-noisy", type=int, default=10,
        help="Noisy re-rollouts per saved state (default: 10)",
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.1,
        help="Gaussian noise std in normalized obs space (default: 0.1). "
             "Typical: 0.05=subtle, 0.1=moderate, 0.3=strong",
    )
    parser.add_argument(
        "--n-future-steps", type=int, default=100,
        help="Steps to run forward per noisy re-rollout (default: 100)",
    )
    parser.add_argument(
        "--subsample-every", type=int, default=1,
        help="Only keep every N-th reference state to reduce Phase 2 compute (default: 1 = keep all)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="Output .pkl path")
    parser.add_argument("--seed", type=int, default=42)
    # Analysis-only mode
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Skip data collection; load --out and generate figures only",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory for output figures (default: <out>.analysis/)",
    )
    args = parser.parse_args()

    # ── Analysis-only mode ───────────────────────────────────────────────────
    if args.analyze_only:
        with open(args.out, "rb") as f:
            data = pickle.load(f)
        out_dir = args.out_dir or (Path(args.out).stem + "_analysis")
        analyze_and_plot(data, out_dir)
        return

    # ── Collection mode ──────────────────────────────────────────────────────
    if args.run is None:
        parser.error("--run is required unless --analyze-only is set")

    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    label, ckpt_path, overrides = parse_run_spec(args.run)
    overrides_full = overrides + [
        f"optimization.device={args.device}",
        "task.num_envs=1",          # one env for sequential restore
        f"optimization.seed={args.seed}",
    ]

    config = load_config(overrides_full)
    envs = make_vec_env(config, seed=args.seed)
    setup_config_for_env(config, envs)
    dataset = make_dataset(config)
    maybe_register_libero_pro_objects(config)

    agent, _ = load_model(ckpt_path, config, dataset, args.device)
    agent.eval()
    if isinstance(agent, TrainingAgent):
        agent = BaselineAgentAdapter(agent, config)

    print(f"\nModel  : {label}  |  ckpt: {ckpt_path}")
    print(f"Task   : {config.task.env_name} ({config.task.obs_type})")
    print(f"Noise  : std={args.noise_std}  (normalized obs space)")
    print(f"Params : n_rollouts={args.n_rollouts}  n_noisy={args.n_noisy}  "
          f"n_future_steps={args.n_future_steps}")

    # Phase 1
    ref_states = collect_reference_states(
        config, agent, dataset, envs, args, args.device
    )

    # Subsample to reduce Phase 2 cost
    if args.subsample_every > 1:
        ref_states = ref_states[:: args.subsample_every]
        print(f"Subsampled to {len(ref_states)} states (every {args.subsample_every})")

    # Phase 2
    results = evaluate_critical_states(
        ref_states, envs, agent, config, dataset, args.device, args
    )

    envs.close()

    output = {
        "label": label,
        "noise_std": args.noise_std,
        "n_noisy": args.n_noisy,
        "n_future_steps": args.n_future_steps,
        "states": results,
    }
    with open(args.out, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved to: {args.out}")

    # Auto-analyze
    out_dir = args.out_dir or (Path(args.out).stem + "_analysis")
    analyze_and_plot(output, out_dir, label=label)


if __name__ == "__main__":
    main()
