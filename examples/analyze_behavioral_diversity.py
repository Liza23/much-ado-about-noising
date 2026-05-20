"""Analyze behavioral diversity from a critical_vs_diverse_states pkl.

Treats diversity score as the criticality measure:
  high diversity = decision fork (critical)
  low diversity  = committed trajectory (redundant)

Usage
-----
    python examples/analyze_behavioral_diversity.py \
        --pkl rollouts/div_vs_crit_lift_mh_fi_j2_n002.pkl \
        --out-dir rollouts/behavioral_diversity_analysis \
        --act-dim 10 --act-steps 8
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ACT_DIM_NAMES = ["pos_x", "pos_y", "pos_z",
                 "rot_0", "rot_1", "rot_2", "rot_3", "rot_4", "rot_5",
                 "gripper"]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def threshold_critical(diversity, percentile=75):
    """States above this percentile diversity are 'critical'."""
    return np.percentile(diversity, percentile)


def plot_diversity_histogram(diversity, thresh, label, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diversity, bins=40, color="#4C72B0", edgecolor="white", lw=0.4, alpha=0.85)
    ax.axvline(thresh, color="red", ls="--", lw=1.5,
               label=f"75th pct threshold ({thresh:.3f})")
    n_crit = (diversity >= thresh).sum()
    ax.set_xlabel("Diversity score (mean pairwise L2 of K action chunks)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{label}  —  Behavioral diversity distribution\n"
        f"Critical (top 25%): {n_crit}/{len(diversity)} states"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "diversity_histogram.png", dpi=150)
    plt.close(fig)


def plot_diversity_vs_timestep(diversity, timesteps, thresh, label, out_dir):
    is_crit = diversity >= thresh
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(timesteps[~is_crit], diversity[~is_crit],
               c="#2ca02c", s=15, alpha=0.5, label="Redundant")
    ax.scatter(timesteps[is_crit], diversity[is_crit],
               c="#d62728", s=25, alpha=0.8, label="Critical")
    ax.axhline(thresh, color="red", ls="--", lw=1)

    # rolling mean
    sort_idx = np.argsort(timesteps)
    ts_s = timesteps[sort_idx]
    div_s = diversity[sort_idx]
    w = max(5, len(ts_s) // 15)
    rm = np.convolve(div_s, np.ones(w) / w, mode="valid")
    ax.plot(ts_s[w // 2: w // 2 + len(rm)], rm, "k-", lw=2, label=f"Rolling mean (w={w})")

    ax.set_xlabel("Episode timestep")
    ax.set_ylabel("Diversity score")
    ax.set_title(f"{label}  —  Behavioral diversity across episode")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "diversity_vs_timestep.png", dpi=150)
    plt.close(fig)


def plot_eef_workspace(eef, diversity, thresh, label, out_dir):
    is_crit = diversity >= thresh
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (xi, yi, xl, yl) in zip(
        axes, [(0, 1, "EEF x", "EEF y"), (0, 2, "EEF x", "EEF z")]
    ):
        sc = ax.scatter(
            eef[:, xi], eef[:, yi],
            c=diversity, cmap="plasma", s=20, alpha=0.7, edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="Diversity score")
        # overlay critical states with red ring
        ax.scatter(eef[is_crit, xi], eef[is_crit, yi],
                   s=40, facecolors="none", edgecolors="red", lw=0.8, alpha=0.7)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    fig.suptitle(
        f"{label}  —  EEF workspace colored by diversity  (red ring = critical)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "eef_diversity_workspace.png", dpi=150)
    plt.close(fig)


def plot_action_dim_variance(action_samples_list, diversity, thresh, act_dim, act_steps, label, out_dir):
    """Per action-dimension std: critical states vs redundant states."""
    is_crit = diversity >= thresh

    flat_dim = act_dim * act_steps
    crit_samples = np.vstack([s for s, c in zip(action_samples_list, is_crit) if c])
    redu_samples = np.vstack([s for s, c in zip(action_samples_list, is_crit) if not c])

    # std across all samples within each group, per flat dimension
    crit_std = crit_samples.std(axis=0) if len(crit_samples) else np.zeros(flat_dim)
    redu_std = redu_samples.std(axis=0) if len(redu_samples) else np.zeros(flat_dim)

    # Average over act_steps to get per-act-dim summary
    crit_std_per_dim = crit_std.reshape(act_steps, act_dim).mean(axis=0)
    redu_std_per_dim = redu_std.reshape(act_steps, act_dim).mean(axis=0)

    names = ACT_DIM_NAMES[:act_dim]
    x = np.arange(act_dim)
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, crit_std_per_dim, w, label="Critical (high div)", color="#d62728", alpha=0.8)
    ax.bar(x + w / 2, redu_std_per_dim, w, label="Redundant (low div)",  color="#2ca02c", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Mean std across samples (action units)")
    ax.set_title(f"{label}  —  Which action dims drive diversity?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "action_dim_variance.png", dpi=150)
    plt.close(fig)


def plot_pca_action_samples(action_samples_list, diversity, thresh, label, out_dir, n_show=6):
    """PCA of K action samples at the N most diverse states — shows cluster structure."""
    is_crit = diversity >= thresh
    crit_indices = np.where(is_crit)[0]
    top_idx = crit_indices[np.argsort(diversity[crit_indices])[::-1][:n_show]]

    if len(top_idx) == 0:
        return

    all_samples = np.vstack([action_samples_list[i] for i in top_idx])
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaler.fit_transform(all_samples))
    K = action_samples_list[0].shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for plot_i, state_i in enumerate(top_idx):
        ax = axes[plot_i]
        c = coords[plot_i * K: (plot_i + 1) * K]

        # Fit GMM k=2 to detect bimodality
        gmm = GaussianMixture(n_components=2, random_state=0).fit(c)
        labels_gmm = gmm.predict(c)

        ax.scatter(c[labels_gmm == 0, 0], c[labels_gmm == 0, 1],
                   c="#1f77b4", s=60, label="Mode A", zorder=3)
        ax.scatter(c[labels_gmm == 1, 0], c[labels_gmm == 1, 1],
                   c="#ff7f0e", s=60, label="Mode B", zorder=3)

        div_val = diversity[state_i]
        t = [s["timestep"] for s in _states_ref][state_i]
        ep = [s["episode"] for s in _states_ref][state_i]
        ax.set_title(f"ep={ep} t={t}  div={div_val:.3f}", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=7)

    for ax in axes[len(top_idx):]:
        ax.axis("off")

    var = pca.explained_variance_ratio_
    fig.suptitle(
        f"{label}  —  PCA of K=10 action samples at top-{n_show} diverse states\n"
        f"PCA variance: {var[0]:.1%} + {var[1]:.1%}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "pca_action_samples_critical.png", dpi=150)
    plt.close(fig)


def print_summary(diversity, timesteps, thresh, label):
    is_crit = diversity >= thresh
    print("\n" + "=" * 60)
    print(f"BEHAVIORAL DIVERSITY ANALYSIS  |  {label}")
    print("=" * 60)
    print(f"Total states        : {len(diversity)}")
    print(f"Threshold (75th pct): {thresh:.4f}")
    print(f"Critical  (≥thresh) : {is_crit.sum()} ({100*is_crit.mean():.0f}%)")
    print(f"Redundant (<thresh) : {(~is_crit).sum()} ({100*(~is_crit).mean():.0f}%)")
    print(f"Diversity mean±std  : {diversity.mean():.4f} ± {diversity.std():.4f}")
    print(f"Diversity range     : [{diversity.min():.4f}, {diversity.max():.4f}]")

    # When do critical states occur?
    crit_ts = timesteps[is_crit]
    redu_ts = timesteps[~is_crit]
    print(f"\nMean timestep (critical) : {crit_ts.mean():.1f}")
    print(f"Mean timestep (redundant): {redu_ts.mean():.1f}")

    # Top 5 most diverse states
    top5 = np.argsort(diversity)[::-1][:5]
    print("\nTop 5 most diverse states:")
    for i in top5:
        s = _states_ref[i]
        print(f"  ep={s['episode']:3d}  t={s['timestep']:4d}  div={diversity[i]:.4f}")


# module-level ref for plot_pca (needs state metadata)
_states_ref = []


def main():
    global _states_ref

    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--act-dim",   type=int, default=10)
    parser.add_argument("--act-steps", type=int, default=8)
    parser.add_argument("--thresh-pct", type=float, default=75,
                        help="Percentile threshold for critical/redundant split (default: 75)")
    args = parser.parse_args()

    data = load_pkl(args.pkl)
    states = data["states"]
    label = data.get("label", Path(args.pkl).stem)
    _states_ref = states

    out_dir = Path(args.out_dir or (Path(args.pkl).stem + "_div_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)

    diversity       = np.array([s["diversity_score"] for s in states])
    timesteps       = np.array([s["timestep"]         for s in states])
    eef             = np.array([s["eef_pos"]           for s in states])
    action_samples  = [s["action_samples"]             for s in states]

    thresh = threshold_critical(diversity, args.thresh_pct)

    print_summary(diversity, timesteps, thresh, label)

    print("\nGenerating plots...")
    plot_diversity_histogram(diversity, thresh, label, out_dir)
    plot_diversity_vs_timestep(diversity, timesteps, thresh, label, out_dir)
    plot_eef_workspace(eef, diversity, thresh, label, out_dir)
    plot_action_dim_variance(action_samples, diversity, thresh,
                             args.act_dim, args.act_steps, label, out_dir)
    plot_pca_action_samples(action_samples, diversity, thresh, label, out_dir)

    print(f"\nFigures → {out_dir}")


if __name__ == "__main__":
    main()
