"""Stage 2: Analyze and visualize multimodality from collected data.

Loads the .pkl produced by collect_multimodal_viz.py and generates:
  1. action_pca_by_method.png       — PCA/UMAP of all rollout action chunks by variant
  2. action_pca_by_success.png      — same colored by success/failure
  3. critical_state_samples.png     — per-critical-state scatter: N samples per variant
  4. critical_state_grid.png        — grid view of all critical states
  5. steer_umap.png                 — steerability (flow_intent only)
  6. diversity_metrics.png          — bar chart of per-dim std, pairwise L2, coverage
  7. diversity_metrics.txt          — markdown table

Usage:
    python examples/analyze_multimodal_viz.py \
        --data rollouts/lift_ph_state_multimodal.pkl \
        --task lift_ph_state \
        --out-dir rollouts/lift_ph_state_multimodal_figs
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.decomposition import PCA

# Reuse helpers from analyze_diversity.py
import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "examples"))
from analyze_diversity import (
    fit_all_embeddings,
    get_color,
    fig_action_umap_method,
    fig_action_umap_success,
    fig_steer_umap,
    fig_diversity_metrics,
    METHOD_COLORS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Critical state visualization
# ──────────────────────────────────────────────────────────────────────────────

def fig_critical_state_samples(task_data: dict, out_dir: Path, task_name: str = ""):
    """For each critical state, plot PCA of N samples per variant side by side."""
    critical = task_data.get("critical_states")
    if critical is None:
        print("[SKIP] No critical states data")
        return

    n_critical = len(critical["timestep"])
    variants = [k for k in task_data
                if isinstance(task_data[k], dict) and "critical_samples" in task_data[k]]

    if not variants:
        print("[SKIP] No variants with critical_samples")
        return

    # Determine grid: one row per critical state, one col per variant + 1 for overlay
    n_cols = len(variants) + 1  # extra col for overlay
    n_rows = min(n_critical, 5)  # show top 5 critical states

    fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3)

    for si in range(n_rows):
        # Collect all samples at this critical state across variants for joint PCA
        all_flat = []
        variant_ranges = {}
        per_variant_flat = {}
        for vi, var in enumerate(variants):
            samples = task_data[var]["critical_samples"][si]  # (n_samples, act_steps, act_dim)
            flat = samples.reshape(samples.shape[0], -1)
            per_variant_flat[var] = flat
        min_dim = min(f.shape[1] for f in per_variant_flat.values())
        for vi, var in enumerate(variants):
            flat = per_variant_flat[var][:, :min_dim]
            start = len(all_flat)
            all_flat.extend(flat)
            variant_ranges[var] = (start, start + len(flat))

        X = np.array(all_flat)
        pca = PCA(n_components=2)
        emb = pca.fit_transform(X)

        var_explained = pca.explained_variance_ratio_.sum()
        eef = critical["eef_pos"][si]
        ts = critical["timestep"][si]
        var_score = critical["variance_score"][si]

        # Per-variant panels
        for vi, var in enumerate(variants):
            ax = fig.add_subplot(gs[si, vi])
            start, end = variant_ranges[var]
            pts = emb[start:end]
            color = get_color(var, vi, len(variants))
            ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.6, s=15, linewidths=0)
            # Mark centroid
            ax.scatter(pts[:, 0].mean(), pts[:, 1].mean(),
                       c=color, s=80, marker="*", edgecolors="white", linewidths=0.5, zorder=5)

            if si == 0:
                ax.set_title(var, fontsize=10, fontweight="bold")
            if vi == 0:
                ax.set_ylabel(f"State #{si}\nt={ts} var={var_score:.4f}", fontsize=8)
            ax.set_aspect("equal", "datalim")
            ax.tick_params(labelsize=6)

        # Overlay panel
        ax = fig.add_subplot(gs[si, -1])
        for vi, var in enumerate(variants):
            start, end = variant_ranges[var]
            pts = emb[start:end]
            color = get_color(var, vi, len(variants))
            ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.4, s=12,
                       linewidths=0, label=var if si == 0 else None)
        if si == 0:
            ax.set_title("Overlay", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, markerscale=2, loc="upper right")
        ax.set_aspect("equal", "datalim")
        ax.tick_params(labelsize=6)

    prefix = f"[{task_name}] " if task_name else ""
    fig.suptitle(f"{prefix}Critical State Samples — PCA of action chunks at high-variance states\n"
                 f"(each dot = one action sample from the same observation)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "critical_state_samples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_critical_state_variance_comparison(task_data: dict, out_dir: Path, task_name: str = ""):
    """Bar chart comparing action variance at each critical state across variants."""
    critical = task_data.get("critical_states")
    if critical is None:
        return

    n_critical = len(critical["timestep"])
    variants = [k for k in task_data
                if isinstance(task_data[k], dict) and "critical_samples" in task_data[k]]

    if not variants:
        return

    n_show = min(n_critical, 10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-state variance comparison
    ax = axes[0]
    x = np.arange(n_show)
    width = 0.8 / len(variants)
    for vi, var in enumerate(variants):
        variances = []
        for si in range(n_show):
            samples = task_data[var]["critical_samples"][si]
            flat = samples.reshape(samples.shape[0], -1)
            variances.append(float(flat.var(axis=0).mean()))
        color = get_color(var, vi, len(variants))
        ax.bar(x + vi * width, variances, width, label=var, color=color, alpha=0.85)
    ax.set_xticks(x + width * len(variants) / 2)
    ax.set_xticklabels([f"S{i}\nt={critical['timestep'][i]}" for i in range(n_show)],
                       fontsize=7, rotation=30)
    ax.set_ylabel("Mean per-dim variance")
    ax.set_title("Action variance at critical states")
    ax.legend(fontsize=8)

    # Right: spread (mean pairwise L2) comparison
    ax2 = axes[1]
    for vi, var in enumerate(variants):
        spreads = []
        for si in range(n_show):
            samples = task_data[var]["critical_samples"][si]
            flat = samples.reshape(samples.shape[0], -1)
            # Subsample pairwise distances
            n = min(len(flat), 50)
            sub = flat[:n]
            diffs = sub[:, None, :] - sub[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            upper = dists[np.triu_indices(n, k=1)]
            spreads.append(float(upper.mean()) if len(upper) > 0 else 0.0)
        color = get_color(var, vi, len(variants))
        ax2.bar(x + vi * width, spreads, width, label=var, color=color, alpha=0.85)
    ax2.set_xticks(x + width * len(variants) / 2)
    ax2.set_xticklabels([f"S{i}\nt={critical['timestep'][i]}" for i in range(n_show)],
                        fontsize=7, rotation=30)
    ax2.set_ylabel("Mean pairwise L2")
    ax2.set_title("Action spread at critical states")
    ax2.legend(fontsize=8)

    prefix = f"[{task_name}] " if task_name else ""
    fig.suptitle(f"{prefix}Multimodality Comparison at Critical States", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "critical_state_variance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_critical_intents(task_data: dict, out_dir: Path, task_name: str = ""):
    """Visualize intent diversity at critical states (flow_intent only)."""
    critical = task_data.get("critical_states")
    if critical is None:
        return

    intent_variants = [k for k in task_data
                       if isinstance(task_data[k], dict)
                       and task_data[k].get("critical_intents") is not None]

    if not intent_variants:
        return

    n_critical = len(critical["timestep"])
    n_show = min(n_critical, 5)

    for var in intent_variants:
        intents = task_data[var]["critical_intents"]  # (n_critical, n_samples, intent_dim)
        actions = task_data[var]["critical_samples"]   # (n_critical, n_samples, act_steps, act_dim)

        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        if n_show == 1:
            axes = axes.reshape(2, 1)

        for si in range(n_show):
            int_si = intents[si]   # (n_samples, intent_dim)
            act_si = actions[si].reshape(actions.shape[1], -1)  # (n_samples, flat_dim)

            # PCA of intents
            if int_si.shape[1] > 2:
                pca_int = PCA(n_components=2)
                int_emb = pca_int.fit_transform(int_si)
            else:
                int_emb = int_si[:, :2]

            # PCA of actions
            pca_act = PCA(n_components=2)
            act_emb = pca_act.fit_transform(act_si)

            # Color by intent cluster (simple k-means with k=3)
            from sklearn.cluster import KMeans
            k = min(3, len(int_si))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(int_si)
            colors = plt.cm.tab10(labels / max(k - 1, 1))

            axes[0, si].scatter(int_emb[:, 0], int_emb[:, 1], c=colors, alpha=0.7, s=20)
            axes[0, si].set_title(f"S{si} Intents (t={critical['timestep'][si]})", fontsize=9)
            axes[0, si].set_aspect("equal", "datalim")
            axes[0, si].tick_params(labelsize=6)

            axes[1, si].scatter(act_emb[:, 0], act_emb[:, 1], c=colors, alpha=0.7, s=20)
            axes[1, si].set_title(f"S{si} Actions (colored by intent)", fontsize=9)
            axes[1, si].set_aspect("equal", "datalim")
            axes[1, si].tick_params(labelsize=6)

        axes[0, 0].set_ylabel("Intent PCA", fontsize=9)
        axes[1, 0].set_ylabel("Action PCA", fontsize=9)

        prefix = f"[{task_name}] " if task_name else ""
        fig.suptitle(f"{prefix}{var}: Intent-to-Action mapping at critical states\n"
                     f"(same color = same intent cluster)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        path = out_dir / f"critical_intents_{var}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def fig_intent_positions_overlay(task_data: dict, out_dir: Path, task_name: str = ""):
    """Plot sampled intent x,y,z positions at each critical state (flow_intent only).

    For each critical state shows 50 sampled future EEF positions in workspace
    coordinates (XY top-down and XZ side views), colored by k-means cluster.
    Red star marks the current EEF position.  Reveals whether multimodality
    corresponds to different spatial goals vs just timing/velocity variation.
    """
    critical = task_data.get("critical_states")
    if critical is None:
        print("[SKIP] intent_positions: no critical_states")
        return

    intent_variants = [k for k in task_data
                       if isinstance(task_data[k], dict)
                       and task_data[k].get("critical_intents") is not None]
    if not intent_variants:
        print("[SKIP] intent_positions: no critical_intents found")
        return

    from sklearn.cluster import KMeans

    n_critical = len(critical["timestep"])
    n_show = min(n_critical, 10)
    print(f"[intent_positions] n_critical={n_critical}, variants={intent_variants}")

    for var in intent_variants:
        raw = task_data[var]["critical_intents"]
        # Normalise to numpy array — pkl may store as list-of-arrays
        if not isinstance(raw, np.ndarray):
            raw = np.array([np.array(x) for x in raw])
        intents = raw  # (n_critical, n_samples, intent_dim)
        print(f"[intent_positions] {var}: intents shape={intents.shape}")
        intent_dim = intents.shape[2]
        # Use XY + XZ for 3D intents; only XY for 2D (e.g. pusht)
        if intent_dim >= 3:
            projections = [
                (0, 1, "X (m)", "Y (m)", "XY"),
                (0, 2, "X (m)", "Z (m)", "XZ"),
            ]
        else:
            projections = [(0, 1, "X", "Y", "XY")]

        n_rows = len(projections)
        xyz_all = intents[:, :, :max(3, intent_dim)]  # keep all dims up to 3

        fig, axes = plt.subplots(n_rows, n_show, figsize=(3.5 * n_show, 4 * n_rows),
                                 sharex=False, sharey=False)
        if n_rows == 1:
            axes = axes.reshape(1, n_show)
        if n_show == 1:
            axes = axes.reshape(n_rows, 1)

        # Build a shared colormap for clusters
        cmap = plt.cm.tab10
        k = 3

        for si in range(n_show):
            pts = intents[si]          # (n_samples, intent_dim)
            eef = critical["eef_pos"][si]  # (3,) or zeros
            ts  = critical["timestep"][si]
            vs  = critical["variance_score"][si]

            # K-means on available dims to identify spatial modes
            n_pts = len(pts)
            n_clusters = min(k, n_pts)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(pts)
            colors = cmap(labels / max(n_clusters - 1, 1))

            for row, (xi, yi, xlabel, ylabel, proj_label) in enumerate(projections):
                ax = axes[row, si]
                ax.scatter(pts[:, xi], pts[:, yi], c=colors, alpha=0.65, s=18, linewidths=0)
                # Current EEF position as reference (only if eef has enough dims)
                if eef is not None and len(eef) > max(xi, yi):
                    ax.scatter(eef[xi], eef[yi], c="red", s=90, marker="*",
                               zorder=5, label="current EEF" if si == 0 else None)
                if si == 0:
                    ax.set_ylabel(f"{proj_label}\n{ylabel}", fontsize=8)
                if row == 0:
                    ax.set_title(f"S{si}  t={ts}\nvar={vs:.4f}", fontsize=8)
                ax.set_xlabel(xlabel, fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_aspect("equal", "datalim")

        # Shared legend
        axes[0, 0].legend(fontsize=7, markerscale=1.2, loc="upper left")

        prefix = f"[{task_name}] " if task_name else ""
        fig.suptitle(
            f"{prefix}{var}: Sampled intent positions at critical states\n"
            f"(each dot = sampled future EEF mean pos; color = k-means cluster; ★ = current EEF)",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        path = out_dir / f"intent_positions_{var}.png"
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        except Exception as e:
            print(f"[ERROR] Could not save {path}: {e}")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze multimodality visualization data")
    parser.add_argument("--data", required=True, help="Path to .pkl from collect_multimodal_viz.py")
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, "rb") as f:
        all_data = pickle.load(f)

    if args.task not in all_data:
        raise ValueError(f"Task '{args.task}' not in data. Available: {list(all_data.keys())}")

    task_data = all_data[args.task]
    print(f"Loaded task '{args.task}' with keys: {list(task_data.keys())}")

    # ── Rollout-level analysis (same as analyze_diversity.py) ─────────────────
    # Build compatible task_data dict for reuse
    compat_data = {}
    method_list = []
    for key, val in task_data.items():
        if not isinstance(val, dict):
            continue
        if "rollout_chunks" in val:
            compat_data[key] = {
                "action_chunks": val["rollout_chunks"],
                "success": val.get("rollout_success", []),
            }
            if "steer_actions" in val:
                compat_data[key]["steer_actions"] = val["steer_actions"]
                compat_data[key]["steer_intents"] = val["steer_intents"]
            method_list.append(key)
        elif "action_chunks" in val:
            compat_data[key] = val
            method_list.append(key)

    if method_list:
        # Build joint embedding
        rows, labels_method, labels_success = [], [], []
        for mi, method in enumerate(method_list):
            data = compat_data[method]
            chunks = data["action_chunks"]
            successes = data.get("success", [])

            if successes and method != "gt_demos":
                n_eps = len(successes)
                chunks_per_ep = max(len(chunks) // n_eps, 1)
                chunk_success = []
                for ep_i, suc in enumerate(successes):
                    start = ep_i * chunks_per_ep
                    end = start + chunks_per_ep if ep_i < n_eps - 1 else len(chunks)
                    chunk_success.extend([int(suc)] * (end - start))
                chunk_success = (chunk_success + [0] * len(chunks))[:len(chunks)]
            else:
                chunk_success = [0] * len(chunks)

            for i, chunk in enumerate(chunks):
                rows.append(chunk)
                labels_method.append(mi)
                labels_success.append(chunk_success[i])

        min_dim = min(r.shape[0] for r in rows)
        rows = [r[:min_dim] for r in rows]
        X = np.array(rows)
        print(f"Fitting embedding on {X.shape[0]} chunks x {X.shape[1]} dims...")
        embeddings = fit_all_embeddings(X)

        for emb, embed_method in embeddings:
            fig_action_umap_method(compat_data, out_dir, embed_method, emb,
                                    labels_method, method_list, task_name=args.task)
            fig_action_umap_success(compat_data, out_dir, embed_method, emb,
                                     labels_success, labels_method, method_list,
                                     task_name=args.task)

        fig_steer_umap(compat_data, out_dir, task_name=args.task)
        fig_diversity_metrics(compat_data, out_dir, task_name=args.task)

    # ── Critical state analysis (NEW) ─────────────────────────────────────────
    print("\n--- Critical State Analysis ---")
    fig_critical_state_samples(task_data, out_dir, task_name=args.task)
    fig_critical_state_variance_comparison(task_data, out_dir, task_name=args.task)
    fig_critical_intents(task_data, out_dir, task_name=args.task)
    fig_intent_positions_overlay(task_data, out_dir, task_name=args.task)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
