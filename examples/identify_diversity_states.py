"""Identify behavioral-diversity states: where does stochastic policy branch into multiple modes?

Question: which states in a trajectory are genuine branching points for behavioral diversity?

At each saved state, run K stochastic rollouts from the SAME physical state (clean env
restore, no perturbation). Each rollout draws a fresh ODE noise vector → potentially
different action trajectory. Measure how much the trajectories diverge.

  High trajectory spread → "diversity state": policy has multiple valid continuations here.
  Low trajectory spread  → "committed state" : all rollouts converge to the same behavior.

For flow_intent (Config A) this directly captures intent-induced diversity: each sample
draws a different intent vector, revealing which timesteps the intent still matters.
For baseline (Config B) it captures the action-space stochasticity of the flow model.

Algorithm
---------
Phase 1 — Reference rollouts
  Run N episodes. At every chunk boundary save:
    sim_state  : full MuJoCo qpos/qvel
    obs_buf    : raw obs buffer (1, obs_steps, obs_dim)
    timestep   : episode step counter
    eef_pos    : EEF xyz at that state

Phase 2 — Stochastic re-rollouts
  For each saved state, repeat K times:
    1. Restore env to the clean sim_state (no noise added).
    2. Run the full remaining horizon with stochastic policy (fresh ODE noise each call).
    3. Record EEF trajectory (eef_pos at each chunk boundary).
  Compute per-state diversity metrics:
    final_eef_spread        — std of final EEF position across K rollouts (metres)
    mean_traj_divergence    — mean pairwise trajectory distance (metres)
    success_rate            — fraction of K rollouts that succeed
    n_modes                 — k-means estimated number of distinct trajectory clusters

Output pkl
----------
{
  "label": str,
  "n_rollouts_ref": int,
  "n_stochastic": int,
  "states": [
    {
      "episode": int,
      "timestep": int,
      "eef_pos": np.ndarray (3,),      # reference EEF at this state
      "sim_state": np.ndarray,
      "trajectories": list[np.ndarray], # K arrays, each (n_chunks, 3) EEF sequence
      "successes": list[bool],
      "intents": list[np.ndarray] | None, # intent vectors per rollout (flow_intent only)
      "final_eef_spread": float,
      "mean_traj_divergence": float,
      "success_rate": float,
      "n_modes": int,
    },
    ...
  ]
}

Usage
-----
    # Baseline
    python examples/identify_diversity_states.py \\
        --run "baseline:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt:task=lift_mh_state:network=mlp:optimization.sample_mode=stochastic" \\
        --n-rollouts 20 --n-stochastic 10 \\
        --device cuda --out rollouts/diversity_lift_mh_baseline.pkl

    # Flow-intent
    python examples/identify_diversity_states.py \\
        --run "flow_intent:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \\
        --n-rollouts 20 --n-stochastic 10 \\
        --device cuda --out rollouts/diversity_lift_mh_fi.pkl

    # Analyze only
    python examples/identify_diversity_states.py --analyze-only \\
        --out rollouts/diversity_lift_mh_fi.pkl --out-dir rollouts/diversity_fi_figs
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
# Phase 1: collect reference states
# ─────────────────────────────────────────────────────────────────────────────

def collect_reference_states(config, agent, dataset, envs, args, device):
    """Run n_rollouts episodes and save (sim_state, obs_buf, timestep, eef_pos) at each chunk boundary."""
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
# Phase 2: stochastic re-rollout from clean state
# ─────────────────────────────────────────────────────────────────────────────

def run_stochastic_rollout(envs, obs_buf, agent, config, dataset, device, n_future_steps, num_steps=9):
    """Run policy forward from current env state, collecting EEF trajectory.

    Each call draws fresh ODE noise (stochastic). The env must already be
    restored to the desired state before calling.

    Returns:
        eef_traj : np.ndarray (n_chunks, 3) — EEF xyz at each chunk boundary
        success  : bool
        intent   : np.ndarray or None — intent vector from first chunk (flow_intent only)
    """
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    done = False
    info = {}
    remaining = n_future_steps
    eef_traj = []
    intent_vec = None
    first_chunk = True

    while not done and remaining > 0:
        fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
        with torch.no_grad():
            if first_chunk and isinstance(agent, (FlowIntentAgent, ResidualPARLWrapper,
                                                   DSRLWrapper, ResidualSACWrapper)):
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps,
                                        return_intent=True)
                if isinstance(act_norm, tuple):
                    act_norm, intent_t = act_norm
                    intent_vec = intent_t[0].cpu().numpy()
            else:
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
        first_chunk = False

        act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
        for a_i in range(min(act_steps, remaining)):
            act = undo_action(act_un[0, s + a_i], config, dataset)
            _, done, info = _step_inner(inner, act)
            if done:
                break

        eef_traj.append(get_eef_from_envs(envs, config))
        obs_buf = update_obs_buf(obs_buf, envs, config)
        remaining -= act_steps

    return {
        "eef_traj": np.array(eef_traj),   # (n_chunks, 3)
        "success": bool(info.get("success", False)),
        "intent": intent_vec,
    }


def _pairwise_traj_distance(trajs):
    """Mean pairwise L2 distance between trajectory endpoints."""
    finals = np.array([t[-1] for t in trajs if len(t) > 0])
    if len(finals) < 2:
        return 0.0
    n = len(finals)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(finals[i] - finals[j])
            count += 1
    return total / count if count > 0 else 0.0


def _estimate_n_modes(trajs, max_k=4):
    """Estimate number of distinct trajectory clusters via k-means on final EEF."""
    from sklearn.cluster import KMeans
    finals = np.array([t[-1] for t in trajs if len(t) > 0])
    if len(finals) < 3:
        return 1
    best_k = 1
    best_inertia = None
    for k in range(1, min(max_k + 1, len(finals))):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(finals)
        if best_inertia is None or km.inertia_ < 0.5 * best_inertia:
            best_inertia = km.inertia_
            best_k = k
        else:
            break
    return best_k


def evaluate_diversity_states(ref_states, envs, agent, config, dataset, device, args):
    """For each saved state, run n_stochastic clean rollouts and measure trajectory diversity."""
    results = []
    n_states = len(ref_states)
    print(f"Phase 2: {n_states} states × {args.n_stochastic} stochastic rollouts each ...\n")

    for i, state in enumerate(ref_states):
        remaining_steps = config.task.max_episode_steps - state["timestep"]
        trajs, successes, intents = [], [], []

        for _ in range(args.n_stochastic):
            obs_buf = restore_env_obs(envs, state["sim_state"], config)
            result = run_stochastic_rollout(
                envs, obs_buf, agent, config, dataset, device,
                n_future_steps=remaining_steps,
            )
            trajs.append(result["eef_traj"])
            successes.append(result["success"])
            if result["intent"] is not None:
                intents.append(result["intent"])

        finals = np.array([t[-1] for t in trajs if len(t) > 0])
        final_spread = float(finals.std(axis=0).mean()) if len(finals) > 1 else 0.0
        mean_div = _pairwise_traj_distance(trajs)
        n_modes = _estimate_n_modes(trajs)
        success_rate = float(sum(successes) / len(successes))

        results.append({
            "episode": state["episode"],
            "timestep": state["timestep"],
            "eef_pos": state["eef_pos"],
            "sim_state": state["sim_state"],
            "trajectories": trajs,
            "successes": successes,
            "intents": intents if intents else None,
            "final_eef_spread": final_spread,
            "mean_traj_divergence": mean_div,
            "success_rate": success_rate,
            "n_modes": n_modes,
        })

        if (i + 1) % max(1, n_states // 20) == 0 or i == n_states - 1:
            print(
                f"  [{i + 1:4d}/{n_states}]  "
                f"ep={state['episode']:3d}  t={state['timestep']:4d}  "
                f"spread={final_spread:.4f}m  modes={n_modes}  sr={success_rate:.2f}"
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis & plotting
# ─────────────────────────────────────────────────────────────────────────────

def analyze_and_plot(data, out_dir, label="policy"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states = data["states"]
    label = data.get("label", label)
    n_stochastic = data.get("n_stochastic", "?")

    timesteps = np.array([s["timestep"] for s in states])
    spread = np.array([s["final_eef_spread"] for s in states])
    divergence = np.array([s["mean_traj_divergence"] for s in states])
    n_modes = np.array([s["n_modes"] for s in states])
    success_rate = np.array([s["success_rate"] for s in states])
    eef = np.array([s["eef_pos"] for s in states])

    # Threshold: spread > 2cm is "diverse"
    spread_threshold = 0.02
    is_diverse = spread > spread_threshold

    # ── 1. Trajectory spread vs timestep ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(timesteps, spread, c=spread, cmap="plasma",
                    s=20, alpha=0.6, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Final EEF spread (m)")
    if len(timesteps) > 10:
        sort_idx = np.argsort(timesteps)
        ts_s, sp_s = timesteps[sort_idx], spread[sort_idx]
        window = max(5, len(ts_s) // 20)
        rm = np.convolve(sp_s, np.ones(window) / window, mode="valid")
        ax.plot(ts_s[window // 2: window // 2 + len(rm)], rm, "k-", lw=2,
                label=f"Rolling mean (w={window})")
        ax.legend(fontsize=9)
    ax.axhline(spread_threshold, color="red", ls="--", lw=1, label=f"{spread_threshold*100:.0f}cm threshold")
    ax.set_xlabel("Episode timestep")
    ax.set_ylabel("Final EEF spread across rollouts (m)")
    ax.set_title(f"{label}  |  trajectory diversity  n_stochastic={n_stochastic}")
    fig.tight_layout()
    fig.savefig(out_dir / "spread_vs_timestep.png", dpi=150)
    plt.close(fig)

    # ── 2. EEF position colored by spread ────────────────────────────────────
    if eef.shape[1] >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (xi, yi, xl, yl) in zip(
            axes, [(0, 1, "EEF x", "EEF y"), (0, 2, "EEF x", "EEF z")]
        ):
            sc = ax.scatter(eef[:, xi], eef[:, yi], c=spread, cmap="plasma",
                            s=20, alpha=0.7, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="EEF spread (m)")
            ax.set_xlabel(xl); ax.set_ylabel(yl)
        fig.suptitle(f"{label}  —  EEF position vs trajectory spread", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "eef_spread_scatter.png", dpi=150)
        plt.close(fig)

    # ── 3. Spread distribution ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(spread, bins=30, color="#4C72B0", edgecolor="white", lw=0.5)
    ax.axvline(spread_threshold, color="red", ls="--", lw=1.5,
               label=f">{spread_threshold*100:.0f}cm threshold")
    n_div = is_diverse.sum()
    ax.set_xlabel("Final EEF spread (m)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{label}  |  trajectory spread distribution\n"
        f"diverse: {n_div}/{len(states)} ({100*n_div/len(states):.0f}%)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spread_distribution.png", dpi=150)
    plt.close(fig)

    # ── 4. Mean spread by timestep bin ───────────────────────────────────────
    if len(timesteps) > 10:
        bins = np.arange(0, timesteps.max() + 16, 16)
        bin_means, bin_stds, bin_centers = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (timesteps >= lo) & (timesteps < hi)
            if mask.sum() > 0:
                bin_means.append(spread[mask].mean())
                bin_stds.append(spread[mask].std())
                bin_centers.append((lo + hi) / 2)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        bin_centers = np.array(bin_centers)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(bin_centers, bin_means, width=14, color="#4C72B0", alpha=0.7)
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="none", color="k", capsize=3)
        ax.axhline(spread_threshold, color="red", ls="--", lw=1)
        ax.set_xlabel("Episode timestep (binned)")
        ax.set_ylabel("Mean EEF spread (m)")
        ax.set_title(f"{label}  —  trajectory diversity by episode phase")
        fig.tight_layout()
        fig.savefig(out_dir / "spread_by_timestep_bin.png", dpi=150)
        plt.close(fig)

    # ── 5. Number of trajectory modes by timestep ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(timesteps, n_modes, c=n_modes, cmap="viridis",
               s=20, alpha=0.6, edgecolors="none", vmin=1, vmax=4)
    ax.set_xlabel("Episode timestep")
    ax.set_ylabel("Estimated trajectory modes (k-means)")
    ax.set_title(f"{label}  —  behavioral modes per state")
    ax.set_ylim(0.5, 4.5)
    fig.tight_layout()
    fig.savefig(out_dir / "n_modes_vs_timestep.png", dpi=150)
    plt.close(fig)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    n_div = is_diverse.sum()
    top_idx = np.argsort(spread)[::-1][:5]
    print("\n" + "=" * 60)
    print(f"SUMMARY  |  {label}  |  n_stochastic={n_stochastic}")
    print("=" * 60)
    print(f"Total states evaluated      : {len(states)}")
    print(f"Diverse (spread>{spread_threshold*100:.0f}cm): {n_div} ({100*n_div/len(states):.1f}%)")
    print(f"Mean EEF spread             : {spread.mean()*100:.2f} ± {spread.std()*100:.2f} cm")
    print(f"Mean pairwise divergence    : {divergence.mean()*100:.2f} cm")
    print(f"Mean #modes per state       : {n_modes.mean():.2f}")
    print(f"Timestep range              : [{timesteps.min()}, {timesteps.max()}]")
    print(f"\nTop 5 most diverse states (highest spread):")
    for idx in top_idx:
        s = states[idx]
        print(
            f"  ep={s['episode']:3d}  t={s['timestep']:4d}  "
            f"spread={s['final_eef_spread']*100:.2f}cm  modes={s['n_modes']}  "
            f"eef={s['eef_pos'].round(3)}"
        )
    print(f"\nFigures saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Identify behavioral-diversity states via stochastic re-rollouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", required=False, metavar="label:ckpt_path[:override...]")
    parser.add_argument("--n-rollouts", type=int, default=20,
                        help="Reference episodes for Phase 1 (default: 20)")
    parser.add_argument("--n-stochastic", type=int, default=10,
                        help="Stochastic re-rollouts per state (default: 10)")
    parser.add_argument("--subsample-every", type=int, default=1,
                        help="Keep every N-th reference state (default: 1 = all)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="Output .pkl path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip collection; load --out and plot only")
    parser.add_argument("--out-dir", default=None,
                        help="Figure output directory (default: <out_stem>_analysis/)")
    args = parser.parse_args()

    if args.analyze_only:
        with open(args.out, "rb") as f:
            data = pickle.load(f)
        out_dir = args.out_dir or (Path(args.out).stem + "_analysis")
        analyze_and_plot(data, out_dir)
        return

    if args.run is None:
        parser.error("--run is required unless --analyze-only is set")

    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    label, ckpt_path, overrides = parse_run_spec(args.run)
    overrides_full = overrides + [
        f"optimization.device={args.device}",
        "task.num_envs=1",
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
    print(f"Params : n_rollouts={args.n_rollouts}  n_stochastic={args.n_stochastic}")

    ref_states = collect_reference_states(config, agent, dataset, envs, args, args.device)

    if args.subsample_every > 1:
        ref_states = ref_states[:: args.subsample_every]
        print(f"Subsampled to {len(ref_states)} states (every {args.subsample_every})")

    results = evaluate_diversity_states(ref_states, envs, agent, config, dataset, args.device, args)

    envs.close()

    output = {
        "label": label,
        "n_rollouts_ref": args.n_rollouts,
        "n_stochastic": args.n_stochastic,
        "states": results,
    }
    with open(args.out, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved to: {args.out}")

    out_dir = args.out_dir or (Path(args.out).stem + "_analysis")
    analyze_and_plot(output, out_dir, label=label)


if __name__ == "__main__":
    main()
