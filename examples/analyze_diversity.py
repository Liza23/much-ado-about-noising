"""Stage 2: Analyze action distribution diversity from collected rollout data.

Loads the .pkl produced by collect_diversity_rollouts.py and generates:
    action_umap_by_method.png   — UMAP of all action chunks, colored by variant
    action_umap_by_success.png  — same UMAP colored by success/failure
    steer_umap.png              — steerability: K intent samples per obs (flow_intent only)
    diversity_metrics.png       — bar chart of per-dim std, mean pairwise L2, coverage
    diversity_metrics.txt       — markdown table of the same metrics

Usage:
    python examples/analyze_diversity.py \
        --data rollouts/lift_ph_diversity.pkl \
        --task lift_ph \
        --out-dir figs/diversity/lift_ph
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# ──────────────────────────────────────────────────────────────────────────────
# UMAP / PCA helper
# ──────────────────────────────────────────────────────────────────────────────

def fit_embedding(X: np.ndarray, n_components: int = 2):
    """Fit UMAP (or PCA fallback) and return (embedding, method_name)."""
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)
        emb = reducer.fit_transform(X)
        return emb, "UMAP"
    except ImportError:
        reducer = PCA(n_components=n_components)
        emb = reducer.fit_transform(X)
        return emb, "PCA"


def fit_all_embeddings(X: np.ndarray, n_components: int = 2):
    """Fit both PCA and UMAP (if available). Returns list of (emb, name)."""
    results = []
    pca = PCA(n_components=n_components)
    results.append((pca.fit_transform(X), "PCA"))
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)
        results.append((reducer.fit_transform(X), "UMAP"))
    except ImportError:
        pass
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "baseline":         "#1f77b4",  # strong blue
    "hierarchical_emb": "#ff7f0e",  # vivid orange
    "flow_intent":      "#2ca02c",  # strong green
    "flow_intent_emb":  "#17becf",  # teal
    "intent_learned":   "#d62728",  # strong red
    "intent_sequence":  "#8c564b",  # brown
    "gt_demos":         "#e377c2",  # pink — distinct from policy variants
}

def get_color(method: str, idx: int, n_methods: int):
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    cmap = plt.cm.tab10
    return cmap(idx / max(n_methods - 1, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Action UMAP — colored by method
# ──────────────────────────────────────────────────────────────────────────────

def fig_action_umap_method(task_data: dict, out_dir: Path, embed_method: str,
                            emb: np.ndarray, labels_method: list, method_list: list,
                            task_name: str = ""):
    fig, ax = plt.subplots(figsize=(8, 6))
    for mi, method in enumerate(method_list):
        mask = np.array(labels_method) == mi
        color = get_color(method, mi, len(method_list))
        alpha = 0.5 if method == "gt_demos" else 0.7
        size = 18 if method == "gt_demos" else 20
        marker = "x" if method == "gt_demos" else "o"
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, label=method, alpha=alpha, s=size,
                   marker=marker, linewidths=0.8 if method == "gt_demos" else 0)
    prefix = f"[{task_name}] " if task_name else ""
    ax.set_title(f"{prefix}{embed_method} of Action Chunks — by variant", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{embed_method} dim 1")
    ax.set_ylabel(f"{embed_method} dim 2")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_aspect("equal", "datalim")
    plt.tight_layout()
    tag = embed_method.lower()
    path = out_dir / f"action_{tag}_by_method.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Action UMAP — colored by success/failure
# ──────────────────────────────────────────────────────────────────────────────

def fig_action_umap_success(task_data: dict, out_dir: Path, embed_method: str,
                             emb: np.ndarray, labels_success: list, labels_method: list,
                             method_list: list, task_name: str = ""):
    # Only non-gt_demos rows have meaningful success labels
    has_success = np.array([m != method_list.index("gt_demos")
                             if "gt_demos" in method_list else True
                             for m in labels_method])
    suc = np.array(labels_success)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: success/failure overlay
    ax = axes[0]
    gt_mask = np.array(labels_method) == (method_list.index("gt_demos") if "gt_demos" in method_list else -1)
    ax.scatter(emb[gt_mask, 0], emb[gt_mask, 1], c="#e377c2", s=18, alpha=0.5, marker="x", linewidths=0.8, label="gt_demos")
    fail_mask = (~gt_mask) & (suc == 0)
    succ_mask = (~gt_mask) & (suc == 1)
    ax.scatter(emb[fail_mask, 0], emb[fail_mask, 1], c="tomato", s=12, alpha=0.5, linewidths=0, label="failure")
    ax.scatter(emb[succ_mask, 0], emb[succ_mask, 1], c="steelblue", s=12, alpha=0.6, linewidths=0, label="success")
    prefix = f"[{task_name}] " if task_name else ""
    ax.set_title(f"{prefix}{embed_method} — success vs failure")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_aspect("equal", "datalim")

    # Right: per-method success-only (fall back to all episodes if no successes)
    ax2 = axes[1]
    any_success = succ_mask.any()
    for mi, method in enumerate(method_list):
        if method == "gt_demos":
            continue
        if any_success:
            mask = (np.array(labels_method) == mi) & (suc == 1)
        else:
            mask = np.array(labels_method) == mi
        if mask.any():
            color = get_color(method, mi, len(method_list))
            ax2.scatter(emb[mask, 0], emb[mask, 1], c=color, label=method,
                        alpha=0.7, s=15, linewidths=0)
    title_suffix = "successful episodes only" if any_success else "all episodes (0% success)"
    ax2.set_title(f"{prefix}{embed_method} — {title_suffix}, by variant")
    ax2.legend(fontsize=9, markerscale=2)
    ax2.set_aspect("equal", "datalim")

    fig.suptitle(f"{prefix}Action Chunk Embedding — Success Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    tag = embed_method.lower()
    path = out_dir / f"action_{tag}_by_success.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Steerability UMAP (flow_intent only)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_convex_hull(ax, points, color, alpha=0.15):
    """Draw convex hull polygon around a set of 2D points if >= 3 points."""
    if len(points) < 3:
        return
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        vertices = np.append(hull.vertices, hull.vertices[0])  # close the polygon
        ax.fill(points[vertices, 0], points[vertices, 1],
                color=color, alpha=alpha, linewidth=0)
        ax.plot(points[vertices, 0], points[vertices, 1],
                color=color, alpha=alpha * 3, linewidth=0.8)
    except Exception:
        pass  # degenerate hull (collinear points) — skip silently


def fig_steer_umap(task_data: dict, out_dir: Path, task_name: str = ""):
    if "flow_intent" not in task_data:
        return
    fi = task_data["flow_intent"]
    if "steer_actions" not in fi:
        print("[SKIP] steer_umap: no steerability data in flow_intent entry")
        return

    steer_acts = fi["steer_actions"]   # (N_eps, K, flat_dim)
    steer_ints = fi["steer_intents"]   # (N_eps, K, intent_dim)
    N_eps, K, flat_dim = steer_acts.shape

    # Flatten to (N_eps*K, flat_dim) and create episode labels
    X = steer_acts.reshape(N_eps * K, flat_dim)
    ep_labels = np.repeat(np.arange(N_eps), K)

    emb, method_name = fit_embedding(X)

    n_show = min(N_eps, 20)  # cap for legibility
    cmap = plt.cm.tab20
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: action space — color by episode, convex hull per episode
    ax = axes[0]
    for ep in range(n_show):
        mask = ep_labels == ep
        color = cmap(ep / n_show)
        pts = emb[mask]
        # Scatter individual samples
        ax.scatter(pts[:, 0], pts[:, 1], c=[color] * mask.sum(),
                   s=40, alpha=0.9, linewidths=0.5, edgecolors="white", zorder=3)
        # Star at centroid
        ax.scatter(pts[:, 0].mean(), pts[:, 1].mean(),
                   c=[color], s=90, marker="*", alpha=1.0, linewidths=0, zorder=4)
        # Convex hull showing the spread of this episode's K samples
        _draw_convex_hull(ax, pts, color=color, alpha=0.12)

    ax.set_title(f"{method_name} of steer actions — color by episode\n"
                 f"(same color = same obs, K={K} samples, hull = spread)")
    ax.set_xlabel(f"{method_name} dim 1"); ax.set_ylabel(f"{method_name} dim 2")
    ax.set_aspect("equal", "datalim")

    # Right: intent space — same coloring + hulls
    X_int = steer_ints.reshape(N_eps * K, -1)
    emb_int, method_int = fit_embedding(X_int)
    ax2 = axes[1]
    for ep in range(n_show):
        mask = ep_labels == ep
        color = cmap(ep / n_show)
        pts_int = emb_int[mask]
        ax2.scatter(pts_int[:, 0], pts_int[:, 1], c=[color] * mask.sum(),
                    s=40, alpha=0.9, linewidths=0.5, edgecolors="white", zorder=3)
        ax2.scatter(pts_int[:, 0].mean(), pts_int[:, 1].mean(),
                    c=[color], s=90, marker="*", alpha=1.0, linewidths=0, zorder=4)
        _draw_convex_hull(ax2, pts_int, color=color, alpha=0.12)

    ax2.set_title(f"{method_int} of sampled intent vectors — color by episode\n"
                  f"(hull = spread of K intent samples in 2D)")
    ax2.set_xlabel(f"{method_int} dim 1"); ax2.set_ylabel(f"{method_int} dim 2")
    ax2.set_aspect("equal", "datalim")

    prefix = f"[{task_name}] " if task_name else ""
    fig.suptitle(f"{prefix}Steerability: {K} intent samples per observation (flow_intent)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "steer_umap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Diversity metrics
# ──────────────────────────────────────────────────────────────────────────────

def _mean_pairwise_l2(A: np.ndarray, B: np.ndarray | None = None, n_sample: int = 500):
    """Mean pairwise L2 distance, subsampled for speed.

    If B is None: within-set distances (upper triangle only).
    If B is given: across-set distances (all pairs A_i vs B_j).
    """
    A = A[np.random.choice(len(A), min(len(A), n_sample), replace=False)]
    if B is None:
        diffs = A[:, None, :] - A[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        upper = dists[np.triu_indices(len(A), k=1)]
        return float(upper.mean()) if len(upper) > 0 else 0.0
    else:
        B = B[np.random.choice(len(B), min(len(B), n_sample), replace=False)]
        diffs = A[:, None, :] - B[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        return float(dists.mean())


def compute_diversity_metrics(chunks: np.ndarray):
    """Compute diversity metrics for a set of action chunks.

    Args:
        chunks: (N, flat_dim)

    Returns:
        dict with keys: per_dim_std, within_l2, coverage
    """
    N, D = chunks.shape

    # 1. Per-dim std (mean over dims)
    per_dim_std = float(chunks.std(axis=0).mean())

    # 2. Within-method mean pairwise L2
    within_l2 = _mean_pairwise_l2(chunks)

    # 3. Coverage: PCA to 2D, discretize into 10×10 grid, fraction of non-empty bins
    pca = PCA(n_components=2)
    proj = pca.fit_transform(chunks)
    bins = 10
    H, _, _ = np.histogram2d(proj[:, 0], proj[:, 1], bins=bins)
    coverage = float((H > 0).sum()) / (bins * bins)

    return {"per_dim_std": per_dim_std, "within_l2": within_l2, "coverage": coverage}


def fig_diversity_metrics(task_data: dict, out_dir: Path, task_name: str = ""):
    # Compute per-variant metrics
    variant_chunks = {}
    for variant, data in task_data.items():
        if "action_chunks" not in data or len(data["action_chunks"]) == 0:
            continue
        variant_chunks[variant] = data["action_chunks"]

    if not variant_chunks:
        print("[SKIP] No action chunks available for diversity metrics")
        return

    # Truncate all chunks to min flat_dim so cross-variant comparisons work
    # even when variants have different act_steps (e.g. flow_intent vs baseline)
    min_dim = min(c.shape[1] for c in variant_chunks.values())
    variant_chunks = {v: c[:, :min_dim] for v, c in variant_chunks.items()}

    metrics = {v: compute_diversity_metrics(c) for v, c in variant_chunks.items()}

    # Within vs across pairwise L2:
    # For each non-gt variant, compute mean L2 to every other variant's chunks.
    # "across_l2" = mean of pairwise distances to all other variants pooled.
    non_gt = [v for v in variant_chunks if v != "gt_demos"]
    for v in non_gt:
        others = np.concatenate([variant_chunks[u] for u in variant_chunks if u != v], axis=0)
        metrics[v]["across_l2"] = _mean_pairwise_l2(variant_chunks[v], others)

    variants = list(metrics.keys())
    colors = [get_color(v, i, len(variants)) for i, v in enumerate(variants)]
    x = np.arange(len(variants))

    metric_keys   = ["per_dim_std", "within_l2", "coverage"]
    metric_labels = ["Per-dim std ↑", "Within-method L2 ↑", "Coverage (2D bins) ↑"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, key, label in zip(axes[:3], metric_keys, metric_labels):
        vals = [metrics[v][key] for v in variants]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # 4th panel: within vs across L2 grouped bar (non-gt variants only)
    ax4 = axes[3]
    non_gt_colors = [get_color(v, i, len(non_gt)) for i, v in enumerate(non_gt)]
    x4 = np.arange(len(non_gt))
    w = 0.35
    within_vals = [metrics[v]["within_l2"] for v in non_gt]
    across_vals  = [metrics[v].get("across_l2", 0.0) for v in non_gt]
    bars_w = ax4.bar(x4 - w/2, within_vals, w, label="within-method", alpha=0.85,
                     color=non_gt_colors, edgecolor="white")
    bars_a = ax4.bar(x4 + w/2, across_vals, w, label="across-method", alpha=0.45,
                     color=non_gt_colors, edgecolor="white", hatch="//")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(non_gt, rotation=30, ha="right", fontsize=9)
    ax4.set_title("Within vs Across method L2 ↑", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=8)
    for bar, val in list(zip(bars_w, within_vals)) + list(zip(bars_a, across_vals)):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    prefix = f"[{task_name}] " if task_name else ""
    fig.suptitle(f"{prefix}Action Diversity Metrics by Variant", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "diversity_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Markdown table
    header = "| Variant | Per-dim std | Within L2 | Across L2 | Coverage |"
    sep    = "|---------|-------------|-----------|-----------|----------|"
    rows = [header, sep]
    for v in variants:
        m = metrics[v]
        across = f"{m['across_l2']:.4f}" if "across_l2" in m else "—"
        rows.append(
            f"| {v} | {m['per_dim_std']:.4f} | {m['within_l2']:.4f} | {across} | {m['coverage']:.4f} |"
        )
    table = "\n".join(rows)
    print("\n" + table)
    txt_path = out_dir / "diversity_metrics.txt"
    txt_path.write_text(table + "\n")
    print(f"Saved: {txt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze action distribution diversity")
    parser.add_argument("--data", required=True, help="Path to .pkl from collect_diversity_rollouts.py")
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, "rb") as f:
        all_data = pickle.load(f)

    if args.task not in all_data:
        raise ValueError(f"Task '{args.task}' not in data file. Available: {list(all_data.keys())}")

    task_data = all_data[args.task]
    print(f"Loaded task '{args.task}' with variants: {list(task_data.keys())}")

    # Build joint embedding across all variants (including GT demos)
    method_list = [v for v in task_data if "action_chunks" in task_data[v]]
    rows, labels_method, labels_success = [], [], []
    ep_chunk_counts = {m: [] for m in method_list}  # track chunks per episode for success labels

    for mi, method in enumerate(method_list):
        data = task_data[method]
        chunks = data["action_chunks"]   # (N_total, flat_dim)
        successes = data.get("success", [])

        # Build per-chunk success labels: repeat each episode's success for its chunk count
        if successes and method != "gt_demos":
            # We don't track per-chunk episode membership; use episode mean
            # For simplicity: assign success label uniformly per episode.
            # Since we don't store chunk→episode mapping, assign majority success.
            n_eps = len(successes)
            chunks_per_ep = max(len(chunks) // n_eps, 1)
            chunk_success = []
            for ep_i, suc in enumerate(successes):
                start = ep_i * chunks_per_ep
                end = start + chunks_per_ep if ep_i < n_eps - 1 else len(chunks)
                chunk_success.extend([int(suc)] * (end - start))
            # Pad/trim to match actual chunk count
            chunk_success = (chunk_success + [0] * len(chunks))[:len(chunks)]
        else:
            chunk_success = [0] * len(chunks)

        for i, chunk in enumerate(chunks):
            rows.append(chunk)
            labels_method.append(mi)
            labels_success.append(chunk_success[i])

    # Truncate all chunks to the minimum size to handle horizon mismatches
    min_dim = min(r.shape[0] for r in rows)
    rows = [r[:min_dim] for r in rows]
    X = np.array(rows)
    print(f"Fitting embedding on {X.shape[0]} action chunks × {X.shape[1]} dims...")
    embeddings = fit_all_embeddings(X)

    # Generate figures for each embedding method
    for emb, embed_method in embeddings:
        print(f"Embedding method: {embed_method}")
        fig_action_umap_method(task_data, out_dir, embed_method, emb, labels_method, method_list, task_name=args.task)
        fig_action_umap_success(task_data, out_dir, embed_method, emb, labels_success, labels_method, method_list, task_name=args.task)

    fig_steer_umap(task_data, out_dir, task_name=args.task)
    fig_diversity_metrics(task_data, out_dir, task_name=args.task)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
