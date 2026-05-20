"""Combined experiment: behavioral diversity vs. physical criticality.

For each state visited during reference rollouts:
  1. Behavioral diversity  — sample K_steer independent action chunks from the
     flow-intent policy (different ODE noise draws from the same observation).
     Diversity score = mean pairwise L2 distance between those K chunks.
  2. Physical criticality  — add Gaussian noise to robot joint angles, restore
     the env, run N_noisy re-rollouts with clean policy observations.
     Recovery rate = fraction that succeed.

Research question: do states where the policy shows HIGH behavioral diversity
(multimodal action distribution) also have LOW recovery (i.e. are "critical")?

Output pkl
----------
{
  "label"       : str,
  "noise_std"   : float,
  "k_steer"     : int,
  "n_noisy"     : int,
  "n_arm_joints": int,
  "states": [
    {
      "episode"        : int,
      "timestep"       : int,
      "eef_pos"        : np.ndarray (3,),
      "sim_state"      : np.ndarray,
      "diversity_score": float,      # mean pairwise L2 of K action chunks
      "action_samples" : np.ndarray, # (K, flat_act_dim)
      "n_noisy"        : int,
      "n_success"      : int,
      "recovery_rate"  : float,
      "outcomes"       : list[bool],
    }, ...
  ]
}

Usage
-----
    # Flow-intent (same agent for diversity + recovery)
    python examples/critical_vs_diverse_states.py \\
        --run "flow_intent:checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt:task=lift_mh_state_flow_intent:network=mlp_flow_intent" \\
        --n-rollouts 20 --n-noisy 10 --noise-std 0.05 --k-steer 10 \\
        --device cuda --out rollouts/div_vs_crit_lift_mh_fi.pkl

    # Analyze existing pkl
    python examples/critical_vs_diverse_states.py --analyze-only \\
        --out rollouts/div_vs_crit_lift_mh_fi.pkl \\
        --out-dir rollouts/div_vs_crit_lift_mh_fi_figs
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
from scipy import stats as scipy_stats

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
from identify_critical_states import (
    perturb_sim_state,
    run_clean_rollout,
)
from mip.agent import TrainingAgent
from mip.flow_intent_agent import FlowIntentAgent
from mip.torch_utils import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Core: collect diversity + criticality at every chunk boundary
# ─────────────────────────────────────────────────────────────────────────────

def collect_diverse_critical_states(config, agent, dataset, envs, args, device):
    """Run reference rollouts; at every (subsampled) state measure diversity + recovery.

    For each saved state:
      - Diversity : sample args.k_steer independent action chunks; compute mean
                    pairwise L2 distance in unnormalized action space.
      - Recovery  : run args.n_noisy perturbed rollouts (Gaussian joint noise).

    Returns list of per-state dicts.
    """
    num_steps = 9
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    rng = np.random.default_rng(args.seed + 1)

    results = []
    n_episodes = 0
    global_state_idx = 0

    print(f"\nCollecting {args.n_rollouts} reference episodes "
          f"(k_steer={args.k_steer}, n_noisy={args.n_noisy}, "
          f"noise_std={args.noise_std} rad) ...")

    while n_episodes < args.n_rollouts:
        obs, _ = envs.reset()
        t = 0
        ep_idx = n_episodes
        ep_saved = []

        # ── Phase 1: roll out one episode, record states ──────────────────────
        while t < config.task.max_episode_steps:
            if t + act_steps <= config.task.max_episode_steps:
                sim_state = get_sim_state(envs, config)
                obs_snap = (
                    obs.copy() if config.task.obs_type == "state"
                    else {k: v.copy() for k, v in obs.items()}
                )
                eef_pos = get_eef_from_envs(envs, config)
                ep_saved.append({
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

        # Subsample
        if args.subsample_every > 1:
            ep_saved = ep_saved[:: args.subsample_every]

        print(f"  ep {ep_idx}: {len(ep_saved)} states to evaluate")

        # ── Phase 2+3: diversity then recovery at each saved state ─────────────
        for state in ep_saved:
            fi_obs, _ = to_fi_obs(state["obs_buf"], config, dataset, device)

            # --- Behavioral diversity (K independent ODE samples) ---------------
            action_samples = []
            with torch.no_grad():
                for _ in range(args.k_steer):
                    act_k = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
                    act_k_un = dataset.normalizer["action"].unnormalize(act_k.cpu().numpy())
                    # slice act_steps starting at obs_steps-1 (matches collect_rollouts)
                    chunk = act_k_un[0, s: s + act_steps].flatten()
                    action_samples.append(chunk)
            action_samples = np.stack(action_samples)  # (K, flat_dim)

            # Mean pairwise L2 distance
            diffs = action_samples[:, None, :] - action_samples[None, :, :]  # (K, K, D)
            pairwise_dists = np.linalg.norm(diffs, axis=-1)                  # (K, K)
            upper = pairwise_dists[np.triu_indices(args.k_steer, k=1)]
            diversity_score = float(upper.mean())

            # --- Physical recovery (N_noisy perturbed re-rollouts) --------------
            remaining_steps = config.task.max_episode_steps - state["timestep"]
            outcomes = []
            for _ in range(args.n_noisy):
                noised = perturb_sim_state(
                    state["sim_state"], args.noise_std, args.n_arm_joints, rng=rng
                )
                obs_buf = restore_env_obs(envs, noised, config)
                res = run_clean_rollout(
                    envs, obs_buf, agent, config, dataset, device,
                    n_future_steps=remaining_steps,
                )
                outcomes.append(res["success"])

            recovery_rate = sum(outcomes) / max(1, args.n_noisy)

            results.append({
                "episode": state["episode"],
                "timestep": state["timestep"],
                "eef_pos": state["eef_pos"],
                "sim_state": state["sim_state"],
                "diversity_score": diversity_score,
                "action_samples": action_samples,
                "n_noisy": args.n_noisy,
                "n_success": sum(outcomes),
                "recovery_rate": recovery_rate,
                "outcomes": outcomes,
            })

            global_state_idx += 1
            if global_state_idx % 10 == 0:
                print(
                    f"    [{global_state_idx:4d}]  "
                    f"ep={state['episode']:3d}  t={state['timestep']:4d}  "
                    f"div={diversity_score:.4f}  recovery={recovery_rate:.2f}"
                )

        n_episodes += config.task.num_envs

    print(f"\nDone. Total states evaluated: {len(results)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis & plotting
# ─────────────────────────────────────────────────────────────────────────────

def analyze_and_plot(data, out_dir, label="policy"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states     = data["states"]
    noise_std  = data["noise_std"]
    k_steer    = data["k_steer"]
    n_noisy    = data["n_noisy"]
    label      = data.get("label", label)

    diversity  = np.array([s["diversity_score"] for s in states])
    recovery   = np.array([s["recovery_rate"]   for s in states])
    timesteps  = np.array([s["timestep"]         for s in states])
    is_critical = recovery < 0.5

    # ── 1. Diversity vs. recovery scatter (the main plot) ────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        diversity, recovery,
        c=timesteps, cmap="viridis", s=25, alpha=0.6, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Episode timestep")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="50% recovery threshold")

    # Pearson + Spearman correlation
    r_pearson, p_pearson   = scipy_stats.pearsonr(diversity, recovery)
    r_spearman, p_spearman = scipy_stats.spearmanr(diversity, recovery)
    ax.set_xlabel(f"Behavioral diversity (mean pairwise L2, K={k_steer})")
    ax.set_ylabel(f"Recovery rate (noise_std={noise_std} rad)")
    ax.set_title(
        f"{label}  —  Diversity vs. Recovery\n"
        f"Pearson r={r_pearson:+.3f} (p={p_pearson:.3f})  "
        f"Spearman ρ={r_spearman:+.3f} (p={p_spearman:.3f})"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "diversity_vs_recovery.png", dpi=150)
    plt.close(fig)

    # ── 2. Diversity distribution: critical vs. redundant ────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(diversity.min(), diversity.max(), 25)
    ax.hist(diversity[is_critical],  bins=bins, alpha=0.6, label="Critical (recovery<50%)", color="#d62728")
    ax.hist(diversity[~is_critical], bins=bins, alpha=0.6, label="Redundant (recovery≥50%)", color="#2ca02c")
    ax.set_xlabel(f"Diversity score (K={k_steer})")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}  —  Diversity distribution by criticality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "diversity_by_criticality.png", dpi=150)
    plt.close(fig)

    # ── 3. Recovery vs. timestep (as before) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(
        timesteps, recovery,
        c=recovery, cmap="RdYlGn", vmin=0, vmax=1,
        s=20, alpha=0.6, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Recovery rate")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("Episode timestep")
    ax.set_ylabel("Recovery rate after joint-angle noise")
    ax.set_title(f"{label}  |  noise_std={noise_std} rad  n_noisy={n_noisy}")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_vs_timestep.png", dpi=150)
    plt.close(fig)

    # ── 4. Diversity vs. timestep ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(
        timesteps, diversity,
        c=timesteps, cmap="viridis",
        s=20, alpha=0.6, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Timestep")
    ax.set_xlabel("Episode timestep")
    ax.set_ylabel(f"Diversity score (K={k_steer})")
    ax.set_title(f"{label}  —  Behavioral diversity across episode")
    fig.tight_layout()
    fig.savefig(out_dir / "diversity_vs_timestep.png", dpi=150)
    plt.close(fig)

    # ── 5. Top-half diversity: recovery comparison ────────────────────────────
    median_div = np.median(diversity)
    hi_div = diversity >= median_div
    lo_div = ~hi_div
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Low diversity\n(bottom 50%)", "High diversity\n(top 50%)"],
        [recovery[lo_div].mean(), recovery[hi_div].mean()],
        color=["#2ca02c", "#d62728"], alpha=0.8,
        yerr=[recovery[lo_div].std(), recovery[hi_div].std()],
        capsize=5,
    )
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("Mean recovery rate")
    ax.set_title(
        f"{label}  —  Recovery by diversity quartile\n"
        f"(median diversity = {median_div:.4f})"
    )
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_by_diversity_half.png", dpi=150)
    plt.close(fig)

    # ── 6. EEF scatter colored by diversity ──────────────────────────────────
    eef = np.array([s["eef_pos"] for s in states])
    if eef.shape[1] >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, (xi, yi, xl, yl) in zip(
            axes,
            [(0, 1, "EEF x", "EEF y"), (0, 2, "EEF x", "EEF z")],
        ):
            sc = ax.scatter(
                eef[:, xi], eef[:, yi],
                c=diversity, cmap="plasma",
                s=20, alpha=0.7, edgecolors="none",
            )
            plt.colorbar(sc, ax=ax, label="Diversity score")
            ax.set_xlabel(xl); ax.set_ylabel(yl)
        fig.suptitle(f"{label}  —  EEF position colored by behavioral diversity (K={k_steer})", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "eef_diversity_scatter.png", dpi=150)
        plt.close(fig)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    n_crit = is_critical.sum()
    print("\n" + "=" * 65)
    print(f"SUMMARY  |  {label}")
    print(f"  noise_std={noise_std} rad  k_steer={k_steer}  n_noisy={n_noisy}")
    print("=" * 65)
    print(f"Total states            : {len(states)}")
    print(f"Critical (recovery<50%) : {n_crit} ({100*n_crit/len(states):.1f}%)")
    print(f"Redundant               : {len(states)-n_crit} ({100*(len(states)-n_crit)/len(states):.1f}%)")
    print(f"Mean recovery rate      : {recovery.mean():.3f} ± {recovery.std():.3f}")
    print(f"Mean diversity          : {diversity.mean():.4f} ± {diversity.std():.4f}")
    print(f"Diversity(critical)     : {diversity[is_critical].mean():.4f} ± {diversity[is_critical].std():.4f}")
    print(f"Diversity(redundant)    : {diversity[~is_critical].mean():.4f} ± {diversity[~is_critical].std():.4f}")
    print(f"Pearson r (div, rec)    : {r_pearson:+.3f}  p={p_pearson:.3f}")
    print(f"Spearman ρ (div, rec)   : {r_spearman:+.3f}  p={p_spearman:.3f}")
    print(f"Recovery: hi-div states : {recovery[hi_div].mean():.3f}")
    print(f"Recovery: lo-div states : {recovery[lo_div].mean():.3f}")
    print(f"\nFigures → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral diversity vs. physical criticality experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run", required=False,
        metavar="label:ckpt_path[:override...]",
        help="Run spec (required unless --analyze-only).",
    )
    parser.add_argument("--n-rollouts",    type=int,   default=20)
    parser.add_argument("--n-noisy",       type=int,   default=10,
                        help="Perturbed re-rollouts per state (default: 10)")
    parser.add_argument("--k-steer",       type=int,   default=10,
                        help="Independent ODE samples for diversity (default: 10)")
    parser.add_argument("--noise-std",     type=float, default=0.02,
                        help="Joint-angle noise std in radians (default: 0.02 ~1deg)")
    parser.add_argument("--n-arm-joints",  type=int,   default=2,
                        help="Number of arm joints to perturb (default: 2 — avoids compounding 7-joint displacement)")
    parser.add_argument("--subsample-every", type=int, default=1,
                        help="Keep every N-th chunk boundary (default: 1 = all)")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--out",           required=True,  help="Output .pkl path")
    parser.add_argument("--out-dir",       default=None,   help="Figure output directory")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument(
        "--agent-type", default=None,
        choices=["residual_sac", "dsrl", "plain_dsrl", "residual_parl"],
    )
    parser.add_argument("--flow-intent-ckpt", default=None)
    parser.add_argument("--analyze-only",  action="store_true")
    args = parser.parse_args()

    # ── Analysis-only mode ────────────────────────────────────────────────────
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

    config  = load_config(overrides_full)
    envs    = make_vec_env(config, seed=args.seed)
    setup_config_for_env(config, envs)
    dataset = make_dataset(config)
    maybe_register_libero_pro_objects(config)

    agent, _ = load_model(
        ckpt_path, config, dataset, args.device,
        flow_intent_ckpt=args.flow_intent_ckpt,
        agent_type=args.agent_type,
    )
    agent.eval()
    if isinstance(agent, TrainingAgent):
        agent = BaselineAgentAdapter(agent, config)

    print(f"\nModel  : {label}  |  ckpt: {ckpt_path}")
    print(f"Task   : {config.task.env_name} ({config.task.obs_type})")
    print(f"Noise  : std={args.noise_std} rad on {args.n_arm_joints} arm joints")
    print(f"Params : n_rollouts={args.n_rollouts}  k_steer={args.k_steer}  n_noisy={args.n_noisy}")

    results = collect_diverse_critical_states(
        config, agent, dataset, envs, args, args.device
    )
    envs.close()

    output = {
        "label"       : label,
        "noise_std"   : args.noise_std,
        "k_steer"     : args.k_steer,
        "n_noisy"     : args.n_noisy,
        "n_arm_joints": args.n_arm_joints,
        "states"      : results,
    }
    with open(args.out, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved to: {args.out}")

    out_dir = args.out_dir or (Path(args.out).stem + "_analysis")
    analyze_and_plot(output, out_dir, label=label)


if __name__ == "__main__":
    main()
