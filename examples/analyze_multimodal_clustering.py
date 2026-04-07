"""Spectral-clustering multimodality analysis for MIP policy rollouts.

Mirrors the analysis from viz/mass_droid_clustering.py and
viz/analyze_task_multimodality_trajectories.py, adapted to work on
the .pkl data produced by collect_multimodal_viz.py.

For each policy variant, at each critical state:
  1. Spectral-cluster the N action samples (varying k)
  2. Compute variance-drop-ratio (VDR), silhouette, Calinski-Harabasz
  3. Produce 3D ghost trajectory plots (Plotly HTML + Matplotlib PNG)
  4. Cross-variant comparison: VDR bar charts, violin distributions

Usage:
    python examples/analyze_multimodal_clustering.py \
        --data rollouts/lift_ph_state_multimodal.pkl \
        --task lift_ph_state \
        --out-dir rollouts/lift_ph_state_cluster_figs \
        --k-min 2 --k-max 6
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viz"))

from clustering_shared import (
    compute_minkowski_distance_matrix,
    total_variance_minkowski,
    weighted_incluster_variance_minkowski,
    integrate_joint_velocity_chunk,
    prepare_trajectories_and_features_from_actions,
)

# Color palette matching analyze_diversity.py
METHOD_COLORS = {
    # MLP variants
    "baseline":                  "#1f77b4",
    "hierarchical_emb":          "#ff7f0e",
    "flow_intent":               "#2ca02c",
    "flow_intent_emb":           "#17becf",
    # ChiUNet variants
    "baseline_chiunet":          "#6baed6",
    "hierarchical_emb_chiunet":  "#e6550d",
    "flow_intent_chiunet_action":"#756bb1",
    # Other
    "gt_demos":                  "#e377c2",
}


def get_color(method, idx=0, n=3):
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    return plt.cm.tab10(idx / max(n - 1, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Per-state spectral clustering (adapted from mass_droid_clustering._process_state)
# ──────────────────────────────────────────────────────────────────────────────

def cluster_action_samples(actions, k_min=2, k_max=6, dt=0.1, vel_dims=7,
                           include_gripper=True, random_state=42, abs_action=False):
    """Spectral-cluster action chunks from the same state.

    Args:
        actions: (N, T, A) — N action samples, each (T, A)
        k_min, k_max: range of cluster counts to try
        abs_action: if True, actions are absolute positions (skip cumsum)

    Returns:
        results: list of dicts (one per k) with metrics
        best: dict with best_k info + cluster labels
    """
    N, T, A = actions.shape

    # Build trajectories
    trajectories, X_feat, plot_matrix, chunk_ids, time_ids = \
        prepare_trajectories_and_features_from_actions(
            actions, dt=dt, vel_dims=vel_dims, include_gripper=include_gripper,
            abs_action=abs_action,
        )

    # Minkowski distance matrix
    D = compute_minkowski_distance_matrix(trajectories)
    tv = total_variance_minkowski(D)

    # Affinity matrix
    pos = D[D > 0]
    sigma = float(np.median(pos)) if pos.size > 0 else 1.0
    A_aff = np.exp(-D ** 2 / (2.0 * sigma ** 2))
    np.fill_diagonal(A_aff, 1.0)

    results = []
    best = {"score": None, "k": None, "labels": None, "vdr": 0.0}

    for k in range(k_min, k_max + 1):
        if N < k or k < 1:
            continue
        try:
            cl = SpectralClustering(
                n_clusters=k, affinity="precomputed",
                assign_labels="kmeans", random_state=random_state,
            )
            labels = cl.fit_predict(A_aff)
        except Exception:
            continue

        wvar = weighted_incluster_variance_minkowski(D, labels)
        drop = tv - wvar
        vdr = float(np.clip(drop / tv, 0, 1)) if tv > 0 else 0.0

        # Calinski-Harabasz
        ch = np.nan
        if k > 1 and N > k and wvar > 0:
            ch = (drop / (k - 1)) / (wvar / (N - k))

        # Silhouette
        sil = np.nan
        try:
            if N >= 10 and len(np.unique(labels)) > 1:
                sil = silhouette_score(X_feat, labels, metric="euclidean")
        except Exception:
            pass

        row = {
            "k": k, "total_variance": float(tv),
            "weighted_incluster_variance": float(wvar),
            "variance_drop": float(drop), "vdr": float(vdr),
            "calinski_harabasz": float(ch), "silhouette": float(sil),
        }
        results.append(row)

        if best["score"] is None or vdr > best["score"] + 1e-12:
            best = {"score": vdr, "k": k, "labels": labels.copy(), "vdr": vdr,
                    "ch": ch, "sil": sil}

    return results, best, {
        "trajectories": trajectories,
        "X_feat": X_feat,
        "plot_matrix": plot_matrix,
        "chunk_ids": chunk_ids,
        "time_ids": time_ids,
        "D": D,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3D ghost trajectory plot (adapted from mass_droid_clustering.plot_actions_xyz
# + analyze_task_multimodality_trajectories.plot_episode_trajectory_with_ghosts)
# ──────────────────────────────────────────────────────────────────────────────

def plot_state_clusters_3d(trajectories, labels, n_clusters, out_prefix,
                           title="", pca_model=None):
    """3D scatter + trajectory lines per cluster. Outputs PNG + Plotly HTML."""
    # Flatten all trajectories for plotting
    all_pts = np.vstack([t for t in trajectories])
    chunk_ids = np.concatenate([np.full(len(t), i) for i, t in enumerate(trajectories)])
    time_ids = np.concatenate([np.arange(len(t)) for t in trajectories])
    per_point_labels = np.concatenate([np.full(len(t), labels[i]) for i, t in enumerate(trajectories)])

    # PCA to 3D (or fewer if data is lower-dimensional), pad to 3D for plotting
    if pca_model is None:
        n_comp = min(3, all_pts.shape[1])
        pca_model = PCA(n_components=n_comp, random_state=0)
        xyz = pca_model.fit_transform(all_pts)
    else:
        xyz = pca_model.transform(all_pts)
    if xyz.shape[1] < 3:
        xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])))

    # ── Matplotlib static PNG ──
    n_clusters_actual = len(np.unique(labels))
    cols = min(n_clusters_actual + 1, 4)
    rows_grid = int(np.ceil((n_clusters_actual + 1) / cols))
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows_grid))

    # Overview panel
    ax = fig.add_subplot(rows_grid, cols, 1, projection="3d")
    cmap = plt.cm.tab10
    for ci in np.unique(labels):
        mask = per_point_labels == ci
        ax.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2],
                   s=2, alpha=0.7, color=cmap(ci / max(n_clusters - 1, 1)),
                   label=f"C{ci}")
    # Draw trajectory lines
    for i in range(len(trajectories)):
        pts_i = xyz[chunk_ids == i]
        if len(pts_i) >= 2:
            color = cmap(labels[i] / max(n_clusters - 1, 1))
            ax.plot(pts_i[:, 0], pts_i[:, 1], pts_i[:, 2],
                    linewidth=0.6, alpha=0.5, color=color)
    ax.set_title("All clusters", fontsize=9)
    ax.legend(fontsize=7, markerscale=3)
    _equalize_3d(ax)

    # Per-cluster panels
    for ci_panel, ci in enumerate(np.unique(labels)):
        if ci_panel + 2 > rows_grid * cols:
            break
        ax = fig.add_subplot(rows_grid, cols, ci_panel + 2, projection="3d")
        cluster_chunks = np.where(labels == ci)[0]
        for i in cluster_chunks:
            pts_i = xyz[chunk_ids == i]
            if len(pts_i) >= 2:
                ax.plot(pts_i[:, 0], pts_i[:, 1], pts_i[:, 2],
                        linewidth=1.0, alpha=0.6)
            ax.scatter(pts_i[:, 0], pts_i[:, 1], pts_i[:, 2],
                       s=3, alpha=0.8)
            # Start/end markers
            ax.scatter([pts_i[0, 0]], [pts_i[0, 1]], [pts_i[0, 2]],
                       s=30, c="none", edgecolor="k", linewidths=0.8, marker="o")
            ax.scatter([pts_i[-1, 0]], [pts_i[-1, 1]], [pts_i[-1, 2]],
                       s=40, c="k", marker="x", linewidths=1.2)
        ax.set_title(f"Cluster {ci} ({len(cluster_chunks)} chunks)", fontsize=9)
        _equalize_3d(ax)

    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = f"{out_prefix}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # ── Plotly interactive HTML ──
    if HAS_PLOTLY:
        fig_p = go.Figure()
        for ci in np.unique(labels):
            mask = per_point_labels == ci
            fig_p.add_trace(go.Scatter3d(
                x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
                mode="markers",
                marker=dict(size=2, opacity=0.7),
                name=f"Cluster {ci}",
            ))
        # Trajectory lines
        for i in range(len(trajectories)):
            pts_i = xyz[chunk_ids == i]
            if len(pts_i) >= 2:
                fig_p.add_trace(go.Scatter3d(
                    x=pts_i[:, 0], y=pts_i[:, 1], z=pts_i[:, 2],
                    mode="lines",
                    line=dict(width=3, color=f"rgba(100,100,100,0.3)"),
                    showlegend=False, hoverinfo="none",
                ))
        fig_p.update_layout(
            title=title,
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig_p.write_html(f"{out_prefix}.html", include_plotlyjs="cdn")

    return pca_model


def _equalize_3d(ax):
    """Make 3D axes equal scale."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ranges = [abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), abs(zlim[1] - zlim[0])]
    max_range = max(ranges)
    for setter, lim in [(ax.set_xlim3d, xlim), (ax.set_ylim3d, ylim), (ax.set_zlim3d, zlim)]:
        mid = np.mean(lim)
        setter([mid - max_range / 2, mid + max_range / 2])


# ──────────────────────────────────────────────────────────────────────────────
# Cross-variant comparison plots
# ──────────────────────────────────────────────────────────────────────────────

def fig_vdr_comparison(all_vdr: dict, critical_meta: dict, out_dir: Path, task_name: str):
    """Bar chart of VDR per critical state, grouped by variant."""
    variants = list(all_vdr.keys())
    n_states = len(next(iter(all_vdr.values())))
    n_show = min(n_states, 10)

    fig, ax = plt.subplots(figsize=(max(12, n_show * 1.5), 5))
    x = np.arange(n_show)
    width = 0.8 / len(variants)

    for vi, var in enumerate(variants):
        vdrs = all_vdr[var][:n_show]
        color = get_color(var, vi, len(variants))
        ax.bar(x + vi * width, vdrs, width, label=var, color=color, alpha=0.85)

    ax.set_xticks(x + width * len(variants) / 2)
    labels = [f"S{i}\nt={critical_meta['timestep'][i]}" for i in range(n_show)]
    ax.set_xticklabels(labels, fontsize=8, rotation=30)
    ax.set_ylabel("Variance Drop Ratio (VDR)")
    ax.set_title(f"[{task_name}] Multimodality (VDR) at Critical States — per variant",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = out_dir / "vdr_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_vdr_violin(all_vdr: dict, out_dir: Path, task_name: str):
    """Violin plot of VDR distribution per variant."""
    variants = list(all_vdr.keys())
    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 2), 5))

    positions = np.arange(1, len(variants) + 1)
    groups = [np.array(all_vdr[v]) for v in variants]

    vp = ax.violinplot(groups, positions=positions, showmedians=True, showextrema=True)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(get_color(variants[i], i, len(variants)))
        body.set_alpha(0.6)

    rng = np.random.RandomState(0)
    for i, vals in enumerate(groups):
        if vals.size == 0:
            continue
        xj = (i + 1) + 0.06 * rng.randn(vals.size)
        ax.scatter(xj, vals, s=15, alpha=0.7,
                   color=get_color(variants[i], i, len(variants)), zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylabel("Variance Drop Ratio")
    ax.set_title(f"[{task_name}] VDR Distribution Across Critical States",
                 fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "vdr_violin.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_best_k_distribution(all_best_k: dict, out_dir: Path, task_name: str):
    """Histogram of best-k per state, per variant."""
    variants = list(all_best_k.keys())
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4),
                             sharey=True)
    if len(variants) == 1:
        axes = [axes]

    for ax, var in zip(axes, variants):
        ks = np.array(all_best_k[var])
        color = get_color(var)
        if ks.size > 0:
            bins = np.arange(ks.min() - 0.5, ks.max() + 1.5, 1)
            ax.hist(ks, bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Best k")
        ax.set_ylabel("Count")
        ax.set_title(var, fontweight="bold")

    fig.suptitle(f"[{task_name}] Best Cluster Count Distribution", fontweight="bold")
    plt.tight_layout()
    path = out_dir / "best_k_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_metrics_table(all_metrics: dict, out_dir: Path, task_name: str):
    """Markdown table of mean metrics per variant."""
    rows = []
    for var, states_metrics in all_metrics.items():
        vdrs = [m["vdr"] for m in states_metrics if m is not None]
        chs = [m["ch"] for m in states_metrics if m is not None and not np.isnan(m.get("ch", np.nan))]
        sils = [m["sil"] for m in states_metrics if m is not None and not np.isnan(m.get("sil", np.nan))]
        best_ks = [m["k"] for m in states_metrics if m is not None]
        rows.append({
            "variant": var,
            "mean_vdr": np.mean(vdrs) if vdrs else 0,
            "std_vdr": np.std(vdrs) if vdrs else 0,
            "mean_ch": np.mean(chs) if chs else 0,
            "mean_sil": np.mean(sils) if sils else 0,
            "mean_k": np.mean(best_ks) if best_ks else 0,
        })

    header = "| Variant | Mean VDR | Std VDR | Mean CH | Mean Sil | Mean k |"
    sep =    "|---------|----------|---------|---------|----------|--------|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['mean_vdr']:.4f} | {r['std_vdr']:.4f} | "
            f"{r['mean_ch']:.2f} | {r['mean_sil']:.4f} | {r['mean_k']:.1f} |"
        )
    table = "\n".join(lines)
    print(f"\n[{task_name}] Clustering Metrics Summary:")
    print(table)

    path = out_dir / "clustering_metrics.txt"
    path.write_text(table + "\n")
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Ghost overlay: side-by-side variant comparison at a single critical state
# ──────────────────────────────────────────────────────────────────────────────

def fig_ghost_comparison(variant_data: dict, state_idx: int, critical_meta: dict,
                         out_dir: Path, task_name: str, dt=0.1, vel_dims=7,
                         abs_action=False, include_umap_ghosts=False):
    """Side-by-side 3D ghost plots for each variant at one critical state.

    Fits a single global PCA across all variants so the projections are comparable.
    UMAP ghost plots are optional because the nonlinear projection can materially
    distort apparent cross-variant spread when every trajectory point is embedded.
    """
    variants = list(variant_data.keys())
    all_trajs = {}
    all_labels = {}
    all_flat_pts = []

    for var in variants:
        actions = variant_data[var]["actions"]  # (N, T, A)
        labels = variant_data[var]["labels"]    # (N,) cluster labels
        N, T, A = actions.shape
        trajs = []
        for i in range(N):
            traj = integrate_joint_velocity_chunk(
                actions[i], dt=dt, vel_dims=vel_dims,
                include_gripper=True, anchor_start=True, abs_action=abs_action,
            )
            trajs.append(traj)
        all_trajs[var] = trajs
        all_labels[var] = labels
        for t in trajs:
            all_flat_pts.append(t)

    # Global PCA (up to 3 components, fewer if data is lower-dimensional)
    all_pts = np.vstack(all_flat_pts)
    n_comp = min(3, all_pts.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(all_pts)

    ts = critical_meta["timestep"][state_idx]
    var_score = critical_meta["variance_score"][state_idx]

    # Optional global UMAP basis. This is disabled by default because the
    # nonlinear projection can make trajectory spread look inverted relative to
    # the underlying clustering geometry.
    umap_model = None
    if include_umap_ghosts:
        if HAS_UMAP and all_pts.shape[0] >= 3:
            n_neighbors = max(2, min(30, all_pts.shape[0] - 1))
            umap_model = umap.UMAP(
                n_components=n_comp, random_state=42, n_neighbors=n_neighbors, n_jobs=1
            ).fit(all_pts)
        elif not HAS_UMAP:
            print("[warn] UMAP not installed; skipping ghost UMAP figure.")

    def _plot_variant_panel(ax, projector, axis_prefix: str, var: str, point_cloud_only: bool = False):
        trajs = all_trajs[var]
        labels = all_labels[var]
        n_clusters = len(np.unique(labels))
        cmap = plt.cm.tab10

        for i, traj in enumerate(trajs):
            xyz = projector(traj)
            if xyz.shape[1] < 3:
                xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])))
            color = cmap(labels[i] / max(n_clusters - 1, 1))
            if point_cloud_only:
                ax.scatter(
                    xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    s=6, alpha=0.18, color=color,
                )
                ax.scatter(
                    [xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]],
                    s=18, c="none", edgecolor="k", linewidths=0.6, marker="o",
                )
                ax.scatter(
                    [xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]],
                    s=22, c="k", linewidths=0.8, marker="x",
                )
                continue

            ax.plot(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                linewidth=0.8, alpha=0.5, color=color,
            )
            ax.scatter(
                [xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]],
                s=15, c="none", edgecolor="k", linewidths=0.5, marker="o",
            )

        ax.set_title(f"{var}\n(k={n_clusters})", fontsize=9, fontweight="bold")
        ax.set_xlabel(f"{axis_prefix}1", fontsize=7)
        ax.set_ylabel(f"{axis_prefix}2", fontsize=7)
        ax.set_zlabel(f"{axis_prefix}3", fontsize=7)
        ax.tick_params(labelsize=5)
        _equalize_3d(ax)

    def _save_family_ghost(
        family_variants,
        projector,
        axis_prefix: str,
        title_suffix: str,
        file_name: str,
        family_label: str,
        point_cloud_only: bool = False,
    ) -> Path | None:
        family_variants = [v for v in family_variants if v in variants]
        if not family_variants:
            return None

        fig = plt.figure(figsize=(5 * len(family_variants), 5))
        for ci, var in enumerate(family_variants):
            ax = fig.add_subplot(1, len(family_variants), ci + 1, projection="3d")
            _plot_variant_panel(
                ax, projector, axis_prefix, var, point_cloud_only=point_cloud_only
            )

        fig.suptitle(
            f"[{task_name}] State #{state_idx} (t={ts}, probe_var={var_score:.4f})\n"
            f"Ghost action trajectories per variant [{family_label}] ({title_suffix})",
            fontsize=10, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        out_path = out_dir / file_name
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        return out_path

    def _render_family_plotly(
        family_variants,
        projector,
        file_name: str,
        family_label: str,
        axis_prefix: str,
        point_cloud_only: bool = False,
    ) -> None:
        if not HAS_PLOTLY:
            return

        family_variants = [v for v in family_variants if v in variants]
        if not family_variants:
            return

        fig_p = go.Figure()
        for var in family_variants:
            trajs = all_trajs[var]
            labels = all_labels[var]
            n_clusters = len(np.unique(labels))
            colors_plotly = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))

            for i, traj in enumerate(trajs):
                xyz = projector(traj)
                if xyz.shape[1] < 3:
                    xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])))
                ci = labels[i]
                r, g, b, _ = colors_plotly[ci % len(colors_plotly)]
                color_rgba = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.18)"
                if point_cloud_only:
                    fig_p.add_trace(go.Scatter3d(
                        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                        mode="markers",
                        marker=dict(size=2.5, color=color_rgba),
                        name=f"{var} C{ci}",
                        legendgroup=f"{var}_C{ci}",
                        showlegend=(i == np.where(labels == ci)[0][0]),
                        hoverinfo="name",
                    ))
                    fig_p.add_trace(go.Scatter3d(
                        x=[xyz[0, 0]], y=[xyz[0, 1]], z=[xyz[0, 2]],
                        mode="markers",
                        marker=dict(size=4, color="rgba(0,0,0,0)", line=dict(color="black", width=2), symbol="circle-open"),
                        name=f"{var} start",
                        legendgroup=f"{var}_start",
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    fig_p.add_trace(go.Scatter3d(
                        x=[xyz[-1, 0]], y=[xyz[-1, 1]], z=[xyz[-1, 2]],
                        mode="markers",
                        marker=dict(size=4, color="black", symbol="x"),
                        name=f"{var} end",
                        legendgroup=f"{var}_end",
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                else:
                    fig_p.add_trace(go.Scatter3d(
                        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                        mode="lines",
                        line=dict(width=3, color=f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.4)"),
                        name=f"{var} C{ci}",
                        legendgroup=f"{var}_C{ci}",
                        showlegend=(i == np.where(labels == ci)[0][0]),
                        hoverinfo="name",
                    ))

        fig_p.update_layout(
            title=f"[{task_name}] State #{state_idx} (t={ts}) — ghost trajectories [{family_label}]",
            scene=dict(
                xaxis_title=f"{axis_prefix}1",
                yaxis_title=f"{axis_prefix}2",
                zaxis_title=f"{axis_prefix}3",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig_p.write_html(str(out_dir / file_name), include_plotlyjs="cdn")

    kitchen_standard = ["baseline", "hierarchical_emb", "flow_intent"]
    kitchen_chiunet = ["baseline_chiunet", "hierarchical_emb_chiunet", "flow_intent_chiunet"]
    is_kitchen_split = task_name == "kitchen_state"

    if is_kitchen_split:
        family_specs = [
            ("standard", "standard", kitchen_standard),
            ("chiunet", "chiunet", kitchen_chiunet),
        ]
        for family_slug, family_label, family_variants in family_specs:
            family_present = [v for v in family_variants if v in variants]
            if not family_present:
                continue

            family_pts = np.vstack([t for var in family_present for t in all_trajs[var]])
            n_comp_family = min(3, family_pts.shape[1])
            pca_family = PCA(n_components=n_comp_family, random_state=0).fit(family_pts)

            umap_family = None
            if include_umap_ghosts and HAS_UMAP and family_pts.shape[0] >= 3:
                n_neighbors = max(2, min(30, family_pts.shape[0] - 1))
                umap_family = umap.UMAP(
                    n_components=n_comp_family, random_state=42, n_neighbors=n_neighbors, n_jobs=1
                ).fit(family_pts)

            _save_family_ghost(
                family_present,
                projector=pca_family.transform,
                axis_prefix="PC",
                title_suffix="global PCA",
                file_name=f"ghost_comparison_{family_slug}_state{state_idx:02d}.png",
                family_label=family_label,
            )
            _render_family_plotly(
                family_present,
                projector=pca_family.transform,
                file_name=f"ghost_comparison_{family_slug}_state{state_idx:02d}.html",
                family_label=family_label,
                axis_prefix="PC",
            )

            if umap_family is not None:
                _save_family_ghost(
                    family_present,
                    projector=umap_family.transform,
                    axis_prefix="UMAP",
                    title_suffix="global UMAP point cloud",
                    file_name=f"ghost-umap_comparison_{family_slug}_state{state_idx:02d}.png",
                    family_label=family_label,
                    point_cloud_only=True,
                )
                _render_family_plotly(
                    family_present,
                    projector=umap_family.transform,
                    file_name=f"ghost-umap_comparison_{family_slug}_state{state_idx:02d}.html",
                    family_label=family_label,
                    axis_prefix="UMAP",
                    point_cloud_only=True,
                )
        return

    # Fallback for non-kitchen tasks and legacy layouts.
    def _save_matplotlib_ghost(
        projector, axis_prefix: str, title_suffix: str, file_name: str,
        point_cloud_only: bool = False,
    ) -> Path:
        fig = plt.figure(figsize=(5 * len(variants), 5))
        for ci, var in enumerate(variants):
            ax = fig.add_subplot(1, len(variants), ci + 1, projection="3d")
            _plot_variant_panel(
                ax, projector, axis_prefix, var, point_cloud_only=point_cloud_only
            )

        fig.suptitle(
            f"[{task_name}] State #{state_idx} (t={ts}, probe_var={var_score:.4f})\n"
            f"Ghost action trajectories per variant ({title_suffix})",
            fontsize=10, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        out_path = out_dir / file_name
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
        return out_path

    _save_matplotlib_ghost(
        projector=pca.transform,
        axis_prefix="PC",
        title_suffix="global PCA",
        file_name=f"ghost_comparison_state{state_idx:02d}.png",
    )

    if umap_model is not None:
        _save_matplotlib_ghost(
            projector=umap_model.transform,
            axis_prefix="UMAP",
            title_suffix="global UMAP point cloud",
            file_name=f"ghost-umap_comparison_state{state_idx:02d}.png",
            point_cloud_only=True,
        )

    _render_family_plotly(
        variants,
        projector=pca.transform,
        file_name=f"ghost_comparison_state{state_idx:02d}.html",
        family_label="all variants",
        axis_prefix="PC",
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spectral-clustering multimodality analysis for MIP policy rollouts"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Time step for velocity integration (1/control_freq)")
    parser.add_argument("--vel-dims", type=int, default=7,
                        help="Number of velocity dimensions in action vector")
    parser.add_argument(
        "--abs-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat actions as absolute positions instead of relative velocities.",
    )
    parser.add_argument("--n-ghost-states", type=int, default=5,
                        help="Number of top critical states to render ghost plots for")
    parser.add_argument(
        "--include-umap-ghosts",
        action="store_true",
        help=(
            "Also render UMAP ghost plots. Disabled by default because the "
            "nonlinear embedding can invert apparent trajectory spread."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ghost_dir = out_dir / "ghost_plots"
    ghost_dir.mkdir(exist_ok=True)
    per_variant_dir = out_dir / "per_variant"
    per_variant_dir.mkdir(exist_ok=True)

    with open(args.data, "rb") as f:
        all_data = pickle.load(f)

    if args.task not in all_data:
        raise ValueError(f"Task '{args.task}' not found. Available: {list(all_data.keys())}")

    task_data = all_data[args.task]
    critical = task_data.get("critical_states")
    if critical is None:
        print("[ERROR] No critical_states in data. Run collect_multimodal_viz.py first.")
        return

    variants = [k for k in task_data
                if isinstance(task_data[k], dict)
                and "critical_samples" in task_data[k]]
    n_critical = len(critical["timestep"])

    print(f"Task: {args.task}")
    print(f"Variants: {variants}")
    print(f"Critical states: {n_critical}")
    print(f"Clustering k range: [{args.k_min}, {args.k_max}]")
    if not args.include_umap_ghosts:
        print("Ghost UMAP plots disabled by default; use --include-umap-ghosts to enable.")

    # ── Phase 1: Cluster per variant x per state ──────────────────────────────
    all_vdr = {v: [] for v in variants}
    all_best_k = {v: [] for v in variants}
    all_best_metrics = {v: [] for v in variants}
    all_cluster_data = {v: [] for v in variants}  # for ghost plots

    for si in range(n_critical):
        ts = critical["timestep"][si]
        var_score = critical["variance_score"][si]
        print(f"\n  State #{si}: t={ts}, probe_variance={var_score:.6f}")

        for var in variants:
            actions = task_data[var]["critical_samples"][si]  # (N, T, A)
            N, T, A = actions.shape

            results, best, extras = cluster_action_samples(
                actions, k_min=args.k_min, k_max=args.k_max,
                dt=args.dt, vel_dims=args.vel_dims, abs_action=args.abs_action,
            )

            if best["k"] is not None:
                vdr = best["vdr"]
                print(f"    {var}: best_k={best['k']} VDR={vdr:.4f} "
                      f"CH={best.get('ch', np.nan):.2f} Sil={best.get('sil', np.nan):.4f}")
                all_vdr[var].append(vdr)
                all_best_k[var].append(best["k"])
                all_best_metrics[var].append(best)
                all_cluster_data[var].append({
                    "actions": actions,
                    "labels": best["labels"],
                    "trajectories": extras["trajectories"],
                })
            else:
                print(f"    {var}: clustering failed (N={N})")
                all_vdr[var].append(0.0)
                all_best_k[var].append(1)
                all_best_metrics[var].append(None)
                all_cluster_data[var].append(None)

            # Per-variant per-state 3D cluster plot
            if best["k"] is not None and si < args.n_ghost_states:
                var_state_dir = per_variant_dir / var
                var_state_dir.mkdir(exist_ok=True)
                plot_state_clusters_3d(
                    extras["trajectories"], best["labels"], best["k"],
                    out_prefix=str(var_state_dir / f"state{si:02d}"),
                    title=f"[{args.task}] {var} — State #{si} (t={ts}, k={best['k']}, VDR={best['vdr']:.3f})",
                )

    # ── Phase 2: Cross-variant comparison plots ───────────────────────────────
    print(f"\n{'='*60}")
    print("Generating comparison plots...")

    fig_vdr_comparison(all_vdr, critical, out_dir, args.task)
    fig_vdr_violin(all_vdr, out_dir, args.task)
    fig_best_k_distribution(all_best_k, out_dir, args.task)
    fig_metrics_table(all_best_metrics, out_dir, args.task)

    # ── Phase 3: Ghost comparison plots (top states) ──────────────────────────
    n_ghost = min(args.n_ghost_states, n_critical)
    for si in range(n_ghost):
        variant_data = {}
        for var in variants:
            cd = all_cluster_data[var][si]
            if cd is not None:
                variant_data[var] = cd
        if variant_data:
            fig_ghost_comparison(
                variant_data, si, critical, ghost_dir, args.task,
                dt=args.dt, vel_dims=args.vel_dims, abs_action=args.abs_action,
                include_umap_ghosts=args.include_umap_ghosts,
            )

    # ── Save clustering results CSV ───────────────────────────────────────────
    csv_rows = []
    for var in variants:
        for si in range(n_critical):
            m = all_best_metrics[var][si]
            if m is None:
                continue
            csv_rows.append({
                "variant": var,
                "state_idx": si,
                "timestep": critical["timestep"][si],
                "probe_variance": critical["variance_score"][si],
                "best_k": m["k"],
                "vdr": m["vdr"],
                "ch": m.get("ch", np.nan),
                "sil": m.get("sil", np.nan),
            })
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_path = out_dir / "clustering_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
