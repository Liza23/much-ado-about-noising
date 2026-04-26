"""Stage 2 (coverage): Analyze state-space coverage from rollout data.

Loads the .pkl produced by collect_diversity_rollouts.py and generates:
    state_pca.png           — PCA of visited obs states, colored by variant
    state_umap.png          — UMAP of visited obs states (if umap installed)
    coverage_metrics.png    — bar chart: sliced-Wasserstein vs GT, 2D bin coverage
    coverage_metrics.txt    — markdown table of the same metrics

Usage:
    python examples/analyze_coverage.py \
        --data rollouts/lift_mh_coverage.pkl \
        --task lift_mh \
        --out-dir figs/coverage/lift_mh
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance


# ──────────────────────────────────────────────────────────────────────────────
# Color palette (matches analyze_diversity.py)
# ──────────────────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "baseline":         "#1f77b4",
    "flow_intent":      "#2ca02c",
    "hierarchical_emb": "#ff7f0e",
    "flow_intent_emb":  "#17becf",
    "gt_demos":         "#e377c2",
}

def get_color(method: str, idx: int, n: int):
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    return plt.cm.tab10(idx / max(n - 1, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ──────────────────────────────────────────────────────────────────────────────

def fit_pca(X: np.ndarray, n_components: int = 2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca


def fit_umap(X: np.ndarray, n_components: int = 2):
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)
        return reducer.fit_transform(X), reducer
    except ImportError:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Sliced Wasserstein distance
# ──────────────────────────────────────────────────────────────────────────────

def sliced_wasserstein(A: np.ndarray, B: np.ndarray,
                       n_projections: int = 200, seed: int = 0) -> float:
    """Sliced Wasserstein distance between two point clouds A and B.

    Projects both onto n_projections random unit directions, computes 1D
    Wasserstein on each projection, and returns the mean.  Only requires
    scipy — no POT library needed.
    """
    rng = np.random.default_rng(seed)
    d = A.shape[1]
    dirs = rng.standard_normal((n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dists = [wasserstein_distance(A @ v, B @ v) for v in dirs]
    return float(np.mean(dists))


# ──────────────────────────────────────────────────────────────────────────────
# Coverage metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_coverage(rollout_states: np.ndarray, gt_states: np.ndarray,
                     pca: PCA, bins: int = 20) -> float:
    """Fraction of GT-occupied 2D PCA bins also visited by the rollout policy.

    Uses the PCA fitted on the joint (GT + all rollout) distribution so that
    comparisons across methods are on the same coordinate system.
    """
    gt_proj = pca.transform(gt_states)
    ro_proj = pca.transform(rollout_states)

    # Define grid from GT extent
    x_min, x_max = gt_proj[:, 0].min(), gt_proj[:, 0].max()
    y_min, y_max = gt_proj[:, 1].min(), gt_proj[:, 1].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, bins + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, bins + 1)

    gt_H, _, _ = np.histogram2d(gt_proj[:, 0], gt_proj[:, 1],
                                 bins=[x_edges, y_edges])
    ro_H, _, _ = np.histogram2d(ro_proj[:, 0], ro_proj[:, 1],
                                 bins=[x_edges, y_edges])

    gt_mask = gt_H > 0
    if gt_mask.sum() == 0:
        return 0.0
    covered = (ro_H > 0) & gt_mask
    return float(covered.sum()) / float(gt_mask.sum())


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: State embedding (PCA or UMAP), colored by variant
# ──────────────────────────────────────────────────────────────────────────────

def fig_state_embedding(method_states: dict, gt_states: np.ndarray,
                         emb_name: str, out_dir: Path, task_name: str = ""):
    """Plot 2D state embedding for all variants + GT demos side by side."""
    method_list = list(method_states.keys())
    n_methods = len(method_list)

    # Build joint array for fitting (subsample GT so it doesn't dominate)
    gt_sub = gt_states[np.random.choice(len(gt_states),
                                         min(len(gt_states), 2000), replace=False)]
    all_states = np.concatenate(
        [method_states[m] for m in method_list] + [gt_sub], axis=0
    )

    if emb_name == "PCA":
        emb_all, pca = fit_pca(all_states)
    else:
        emb_all, reducer = fit_umap(all_states)
        if emb_all is None:
            return None  # umap not installed
        pca = None

    # Split back
    splits = {}
    offset = 0
    for m in method_list:
        n = len(method_states[m])
        splits[m] = emb_all[offset: offset + n]
        offset += n
    splits["gt_demos"] = emb_all[offset:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: all variants + GT
    ax = axes[0]
    gt_e = splits["gt_demos"]
    ax.scatter(gt_e[:, 0], gt_e[:, 1], c=METHOD_COLORS["gt_demos"],
               s=12, alpha=0.35, marker="x", linewidths=0.7, label="gt_demos", zorder=1)
    for mi, m in enumerate(method_list):
        e = splits[m]
        color = get_color(m, mi, n_methods)
        ax.scatter(e[:, 0], e[:, 1], c=color, s=15, alpha=0.55, linewidths=0, label=m, zorder=2)
    prefix = f"[{task_name}] " if task_name else ""
    ax.set_title(f"{prefix}{emb_name} of visited states — all variants", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_aspect("equal", "datalim")

    # Right: one panel per variant, GT shown as background
    ax2 = axes[1]
    ax2.scatter(gt_e[:, 0], gt_e[:, 1], c=METHOD_COLORS["gt_demos"],
                s=8, alpha=0.20, marker="x", linewidths=0.5, label="gt_demos", zorder=1)
    for mi, m in enumerate(method_list):
        e = splits[m]
        color = get_color(m, mi, n_methods)
        ax2.scatter(e[:, 0], e[:, 1], c=color, s=20, alpha=0.65, linewidths=0,
                    label=m, zorder=3)
    ax2.set_title(f"{prefix}{emb_name} — variants vs GT (GT=background)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, markerscale=2)
    ax2.set_aspect("equal", "datalim")

    fig.suptitle(f"{prefix}State-Space Coverage: {emb_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"state_{emb_name.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return pca  # return PCA fitted on joint distribution (used for coverage metric)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Coverage metrics bar chart
# ──────────────────────────────────────────────────────────────────────────────

def fig_coverage_metrics(method_states: dict, gt_states: np.ndarray,
                          joint_pca: PCA, out_dir: Path, task_name: str = ""):
    """Bar charts: sliced Wasserstein + 2D PCA bin coverage vs GT."""
    method_list = list(method_states.keys())
    n = len(method_list)
    colors = [get_color(m, i, n) for i, m in enumerate(method_list)]

    sw_dists = []
    coverages = []
    for m in method_list:
        states = method_states[m]
        gt_sub = gt_states[np.random.choice(len(gt_states),
                                              min(len(gt_states), len(states)), replace=False)]
        sw = sliced_wasserstein(states, gt_sub, n_projections=300)
        cov = compute_coverage(states, gt_states, joint_pca, bins=20)
        sw_dists.append(sw)
        coverages.append(cov)
        print(f"  {m:25s}  SW={sw:.4f}  coverage={cov:.3f}")

    x = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(x, sw_dists, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(method_list, rotation=30, ha="right", fontsize=9)
    ax.set_title("Sliced Wasserstein ↓ vs GT demos\n(lower = more similar to demo distribution)",
                 fontsize=10, fontweight="bold")
    for bar, val in zip(bars, sw_dists):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    bars2 = ax2.bar(x, coverages, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_list, rotation=30, ha="right", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("State coverage ↑ (fraction of GT bins visited)\n(higher = more of MDP explored)",
                  fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, coverages):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    prefix = f"[{task_name}] " if task_name else ""
    fig.suptitle(f"{prefix}State-Space Coverage Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "coverage_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Markdown table
    header = "| Variant | Sliced-Wasserstein ↓ | Coverage ↑ |"
    sep    = "|---------|----------------------|------------|"
    rows = [header, sep]
    for m, sw, cov in zip(method_list, sw_dists, coverages):
        rows.append(f"| {m} | {sw:.4f} | {cov:.4f} |")
    table = "\n".join(rows)
    print("\n" + table)
    txt_path = out_dir / "coverage_metrics.txt"
    txt_path.write_text(table + "\n")
    print(f"Saved: {txt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze state-space coverage")
    parser.add_argument("--data", required=True, help="Path to .pkl from collect_diversity_rollouts.py")
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-states", type=int, default=3000,
                        help="Max states per variant to subsample for embedding (speed)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, "rb") as f:
        all_data = pickle.load(f)

    if args.task not in all_data:
        raise ValueError(f"Task '{args.task}' not in data. Available: {list(all_data.keys())}")

    task_data = all_data[args.task]
    print(f"Loaded task '{args.task}' with variants: {list(task_data.keys())}")

    gt_entry = task_data.get("gt_demos", {})
    gt_states = gt_entry.get("obs_states", None)
    if gt_states is None:
        raise ValueError(
            "No 'obs_states' found in gt_demos. "
            "Re-run collect_diversity_rollouts.py with the updated script."
        )
    print(f"GT demo states: {gt_states.shape}")

    # Collect per-variant rollout states (exclude gt_demos)
    method_states = {}
    for variant, data in task_data.items():
        if variant == "gt_demos":
            continue
        states = data.get("obs_states", None)
        if states is None or len(states) == 0:
            print(f"[SKIP] {variant}: no obs_states — re-run collection script")
            continue
        # Subsample for speed
        if len(states) > args.max_states:
            idx = np.random.default_rng(0).choice(len(states), args.max_states, replace=False)
            states = states[idx]
        method_states[variant] = states
        print(f"  {variant}: {states.shape}")

    if not method_states:
        raise ValueError("No variants with obs_states found.")

    # PCA embedding (always available, used for coverage metric)
    print("\nFitting PCA embedding...")
    joint_pca = fig_state_embedding(method_states, gt_states, "PCA", out_dir, task_name=args.task)

    # UMAP embedding (optional)
    print("\nFitting UMAP embedding (may take a moment)...")
    fig_state_embedding(method_states, gt_states, "UMAP", out_dir, task_name=args.task)

    # Coverage metrics
    print("\nComputing coverage metrics...")
    fig_coverage_metrics(method_states, gt_states, joint_pca, out_dir, task_name=args.task)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
