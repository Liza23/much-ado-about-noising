"""Plot ghost trajectory comparison across model variants.

Reads a .pkl produced by collect_ghost_rollouts.py and renders one figure per
critical state.  Each figure has one 3-D PCA subplot per model variant.
Trajectories are coloured by spectral cluster label, matching the style of
rollouts/libero_pudding_cluster_figs produced by analyze_multimodal_clustering.py.

Usage:
    python examples/plot_ghost.py \\
        --in  rollouts/ghost_rollouts.pkl \\
        --out rollouts/ghost_plot.png \\
        [--n-critical 5] [--n-ghost-plot 8] [--select-by variance] [--k 3]
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))


# ── helpers ──────────────────────────────────────────────────────────────────

def _equalize_3d(ax):
    """Make 3D axes equal scale."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ranges = [abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), abs(zlim[1] - zlim[0])]
    max_range = max(ranges) if max(ranges) > 0 else 1.0
    for setter, lim in [(ax.set_xlim3d, xlim), (ax.set_ylim3d, ylim), (ax.set_zlim3d, zlim)]:
        mid = np.mean(lim)
        setter([mid - max_range / 2, mid + max_range / 2])


def fit_global_pca(trajs_per_model):
    """Fit PCA on all trajectory points from all models."""
    pts = np.vstack([t for trajs in trajs_per_model.values() for t in trajs if len(t) > 1])
    n_comp = min(3, pts.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(pts)
    return pca


def spectral_cluster_trajs(trajs, k):
    """Spectral-cluster a list of (T, D) trajectories. Returns (N,) labels."""
    N = len(trajs)
    if N < 2 or k < 2:
        return np.zeros(N, dtype=int)

    # Pairwise average pointwise Euclidean distance
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            ti, tj = trajs[i], trajs[j]
            min_len = min(len(ti), len(tj))
            if min_len < 1:
                d = 0.0
            else:
                d = float(np.mean(np.linalg.norm(ti[:min_len] - tj[:min_len], axis=-1)))
            D[i, j] = D[j, i] = d

    pos = D[D > 0]
    sigma = float(np.median(pos)) if pos.size > 0 else 1.0
    A = np.exp(-D ** 2 / (2.0 * sigma ** 2))
    np.fill_diagonal(A, 1.0)

    k_actual = min(k, N)
    try:
        labels = SpectralClustering(
            n_clusters=k_actual, affinity="precomputed",
            assign_labels="kmeans", random_state=42,
        ).fit_predict(A)
    except Exception:
        labels = np.zeros(N, dtype=int)
    return labels


def plot_model_panel(ax, trajs, labels, pca, label):
    """Draw ghost trajectories on a 3D axis, coloured by cluster."""
    n_clusters = max(int(labels.max()) + 1, 1)
    cmap = plt.cm.tab10

    for i, traj in enumerate(trajs):
        if len(traj) < 2:
            continue
        proj = pca.transform(traj)
        if proj.shape[1] < 3:
            proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
        color = cmap(labels[i] / max(n_clusters - 1, 1))
        ax.plot(proj[:, 0], proj[:, 1], proj[:, 2],
                linewidth=0.8, alpha=0.5, color=color)
        # start marker only (open circle)
        ax.scatter([proj[0, 0]], [proj[0, 1]], [proj[0, 2]],
                   s=15, c="none", edgecolor="k", linewidths=0.5, marker="o")

    ax.set_title(f"{label}\n(k={n_clusters})", fontsize=9, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=7)
    ax.set_ylabel("PC2", fontsize=7)
    ax.set_zlabel("PC3", fontsize=7)
    ax.tick_params(labelsize=5)
    _equalize_3d(ax)


def main():
    parser = argparse.ArgumentParser(
        description="Plot ghost trajectory comparison from collect_ghost_rollouts.pkl"
    )
    parser.add_argument("--in", required=True, dest="pkl_path", help="Input .pkl path")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--n-critical", type=int, default=None,
                        help="Max critical states to plot (default: all)")
    parser.add_argument("--n-ghost-plot", type=int, default=8,
                        help="Max ghost trajectories per model per panel")
    parser.add_argument("--select-by", choices=("variance", "index"), default="variance",
                        help="How to order critical states")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of spectral clusters per panel")
    parser.add_argument("--tag", default=None,
                        help="Optional tag prepended to figure title")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Subset of run labels to plot (default: all)")
    parser.add_argument("--n-cols", type=int, default=None,
                        help="Number of columns per row (default: all in one row)")
    args = parser.parse_args()

    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    runs = data["runs"]           # label → {"per_state": [[rollouts], ...]}
    labels = list(runs.keys())
    if args.labels is not None:
        labels = [l for l in args.labels if l in runs]
    mode = data.get("mode", "model_comparison")
    is_perturb = mode == "perturb"

    out_base = Path(args.out)

    if is_perturb:
        # ── Perturb mode: each run has its own critical_states ────────────────
        # We plot one figure per "state rank" (rank-0 = highest variance state
        # in each env independently).  Columns = env variants.
        model_label = data.get("model_label", "model")
        n_show = args.n_critical or max(
            len(v.get("critical_states", [])) for v in runs.values()
        )

        for rank in range(n_show):
            # Gather trajectories and per-env metadata for this rank
            trajs_per_env = {}
            state_metas = {}
            for lbl in labels:
                run_data = runs[lbl]
                crit = run_data.get("critical_states", [])
                per_state = run_data.get("per_state", [])

                # Sort by variance descending (same as select_by=variance)
                indexed_crit = sorted(enumerate(crit), key=lambda x: x[1]["variance"], reverse=True)
                if rank >= len(indexed_crit):
                    trajs_per_env[lbl] = []
                    state_metas[lbl] = {"timestep": -1, "variance": 0.0}
                    continue

                ci, state_meta = indexed_crit[rank]
                state_metas[lbl] = state_meta
                trajs = []
                if ci < len(per_state):
                    for r in per_state[ci][:args.n_ghost_plot]:
                        traj = np.asarray(r["eef_trajectory"], dtype=np.float32)
                        if len(traj) > 1:
                            trajs.append(traj)
                trajs_per_env[lbl] = trajs

            all_trajs_flat = [t for trajs in trajs_per_env.values() for t in trajs]
            if not all_trajs_flat:
                print(f"Rank {rank + 1}: no trajectories, skipping.")
                continue

            pca = fit_global_pca(trajs_per_env)

            labels_per_env = {}
            for lbl in labels:
                trajs = trajs_per_env[lbl]
                if not trajs:
                    labels_per_env[lbl] = np.zeros(0, dtype=int)
                    continue
                proj_trajs = []
                for t in trajs:
                    proj = pca.transform(t)
                    if proj.shape[1] < 3:
                        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
                    proj_trajs.append(proj)
                labels_per_env[lbl] = spectral_cluster_trajs(proj_trajs, args.k)

            n_panels = len(labels)
            n_cols = args.n_cols or n_panels
            n_rows = math.ceil(n_panels / n_cols)
            fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

            tag = f"[{args.tag}] " if args.tag else ""
            fig.suptitle(
                f"{tag}Model: {model_label} — State rank #{rank + 1}\n"
                f"Ghost trajectories per env variant (global PCA)",
                fontsize=10, fontweight="bold",
            )

            for col, lbl in enumerate(labels):
                trajs = trajs_per_env[lbl]
                cluster_lbs = labels_per_env[lbl]
                sm = state_metas[lbl]
                ax = fig.add_subplot(n_rows, n_cols, col + 1, projection="3d")
                panel_label = f"{lbl}\nt={sm['timestep']} var={sm['variance']:.4f}"
                if trajs:
                    plot_model_panel(ax, trajs, cluster_lbs, pca, panel_label)
                else:
                    ax.set_title(f"{lbl}\n(no data)", fontsize=9)

            fig.tight_layout(rect=[0, 0, 1, 0.90])

            if n_show == 1:
                out_path = out_base
            else:
                out_path = out_base.with_stem(f"{out_base.stem}_r{rank + 1:02d}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

    else:
        # ── Model-comparison mode (original behaviour) ────────────────────────
        critical_states = data["critical_states"]

        indexed = list(enumerate(critical_states))
        if args.select_by == "variance":
            indexed = sorted(indexed, key=lambda x: x[1]["variance"], reverse=True)
        n_show = min(args.n_critical or len(indexed), len(indexed))

        for fig_i, (ci, state_meta) in enumerate(indexed[:n_show]):
            ts = state_meta["timestep"]
            var = state_meta["variance"]

            # Collect trajectories per model
            trajs_per_model = {}
            for lbl in labels:
                per_state = runs[lbl].get("per_state", [])
                if ci >= len(per_state):
                    trajs_per_model[lbl] = []
                    continue
                trajs = []
                for r in per_state[ci][:args.n_ghost_plot]:
                    traj = np.asarray(r["eef_trajectory"], dtype=np.float32)
                    if len(traj) > 1:
                        trajs.append(traj)
                trajs_per_model[lbl] = trajs

            all_trajs_flat = [t for trajs in trajs_per_model.values() for t in trajs]
            if not all_trajs_flat:
                print(f"State {ci}: no trajectories, skipping.")
                continue

            pca = fit_global_pca(trajs_per_model)

            # Cluster per model
            labels_per_model = {}
            for lbl in labels:
                trajs = trajs_per_model[lbl]
                if not trajs:
                    labels_per_model[lbl] = np.zeros(0, dtype=int)
                    continue
                proj_trajs = []
                for t in trajs:
                    proj = pca.transform(t)
                    if proj.shape[1] < 3:
                        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
                    proj_trajs.append(proj)
                labels_per_model[lbl] = spectral_cluster_trajs(proj_trajs, args.k)

            n_models = len(labels)
            n_cols = args.n_cols or n_models
            n_rows = math.ceil(n_models / n_cols)
            fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

            tag = f"[{args.tag}] " if args.tag else ""
            fig.suptitle(
                f"{tag}State #{fig_i + 1} (t={ts}, probe_var={var:.4f})\n"
                f"Ghost action trajectories per variant (global PCA)",
                fontsize=10, fontweight="bold",
            )

            for col, lbl in enumerate(labels):
                trajs = trajs_per_model[lbl]
                cluster_labels = labels_per_model[lbl]
                ax = fig.add_subplot(n_rows, n_cols, col + 1, projection="3d")
                if trajs:
                    plot_model_panel(ax, trajs, cluster_labels, pca, lbl)
                else:
                    ax.set_title(f"{lbl}\n(no data)", fontsize=9)

            fig.tight_layout(rect=[0, 0, 1, 0.90])

            if n_show == 1:
                out_path = out_base
            else:
                out_path = out_base.with_stem(f"{out_base.stem}_s{fig_i + 1:02d}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
