#!/usr/bin/env python3
"""
analyze_action_multimodality.py

Spectral version, parallelized, true symmetrized Minkowski distance on chunks,
and per-state checkpointing.

UPDATED (joint-velocity integration):
- DROID action chunks are JOINT VELOCITIES (7 dims) + GRIPPER POSITION (1 dim).
- Before computing chunk distances / clustering, we INTEGRATE joint velocities
  to get integrated joint displacement trajectories (Δq) using dt = 1/fps.
- For plotting, we also plot the integrated trajectories (Δq plus gripper).
  (If you want absolute joint positions q(t)=q0+Δq(t), you must provide q0
   from state; this script does not have joint_position by default.)

What we do concretely:
- For each chunk (T, 8):
    v = chunk[:, :7]                  # joint velocities
    dq = cumsum(v * dt)               # integrated displacement
    dq -= dq[0]  (optional, only when T>1; keeps shape but anchors start at 0)
    g = chunk[:, 7:8]                 # gripper position (NOT integrated)
    traj_used = concat([dq, g], axis=1)  # (T,8)
- All Minkowski distances / variances / spectral clustering use traj_used.

Notes:
- If your velocities are normalized (clipped [-1,1]) and env scales internally,
  then dq is in “normalized displacement units”. It’s still consistent for
  comparing shapes, but not physical radians unless you apply the same scaling.

Outputs:
- Writes per-state checkpoints under outdir/per_state/state_XXXXXX.npz containing
  rows (metrics for all tried k) and best (dict for the best k only).
- Cluster labels are saved only for the best k, in best["labels"] inside the
  per-state .npz; per-k rows do not include labels.
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go
import pdb

from clustering_shared import (
    as_2d as _as_2d,
    compute_minkowski_distance_matrix,
    load_actions_with_fallback as _load_actions_with_fallback,
    prepare_trajectories_and_features_from_actions as _prepare_trajectories_and_features_from_actions,
    total_variance_minkowski,
    weighted_incluster_variance_minkowski,
)

# ------------------------------------------------------------------------------------
# Globals for worker processes (set once by initializer)
# ------------------------------------------------------------------------------------
_G_ACTIONS = None
_G_META = None
_G_ARGS = None


def _init_worker(actions_arr, meta_records, args_dict):
    """Initializer for each worker; stores big arrays in globals to avoid re-pickling."""
    global _G_ACTIONS, _G_META, _G_ARGS
    _G_ACTIONS = actions_arr
    _G_META = meta_records
    _G_ARGS = args_dict


def _list_actions_from_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "actions" not in data:
        return []
    arr = data["actions"]
    if arr.size == 0:
        return []
    return [arr[i] for i in range(len(arr))]


# ---- DTW support (unchanged) ----
def dtw_distance_band(X, Y, window=None):
    X = _as_2d(X)
    Y = _as_2d(Y)
    len_x, len_y = X.shape[0], Y.shape[0]
    if len_x == 0 and len_y == 0:
        return 0.0
    if len_x == 0 or len_y == 0:
        return np.inf
    if window is None:
        window = max(len_x, len_y) // 10
    window = int(window)
    w = max(window, abs(len_x - len_y))
    prev = np.full(len_y + 1, np.inf)
    curr = np.full(len_y + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, len_x + 1):
        curr.fill(np.inf)
        j_start = max(1, i - w)
        j_end = min(len_y, i + w)
        xi = X[i - 1]
        for j in range(j_start, j_end + 1):
            dist = np.dot(xi - Y[j - 1], xi - Y[j - 1])
            curr[j] = dist + min(curr[j - 1], prev[j], prev[j - 1])
        prev, curr = curr, prev
    return float(np.sqrt(prev[len_y]))


_TRAJ = None


def _init_dtw_worker(trajectories):  # noqa
    global _TRAJ
    _TRAJ = trajectories


def _pair_dtw(args):  # noqa
    i, j, win = args
    return i, j, dtw_distance_band(_TRAJ[i], _TRAJ[j], window=win)


def compute_dtw_matrix(trajectories, window=None, parallelize=False, max_workers=None):
    from concurrent.futures import ProcessPoolExecutor as PPE
    N = len(trajectories)
    D = np.zeros((N, N), dtype=float)
    if not parallelize or N < 64:
        for i in range(N):
            for j in range(i + 1, N):
                d = dtw_distance_band(trajectories[i], trajectories[j], window=window)
                D[i, j] = D[j, i] = d
        return D
    args = [(i, j, window) for i in range(N) for j in range(i + 1, N)]
    with PPE(max_workers=max_workers, initializer=_init_dtw_worker, initargs=(trajectories,)) as ex:
        for i, j, d in ex.map(_pair_dtw, args, chunksize=32):
            D[i, j] = D[j, i] = d
    return D


# ------------------------------------------------------------------------------------
# plotting helpers
# ------------------------------------------------------------------------------------

def to_xyz(actions, mode="pca", pca_model=None, kinematic_map=None):
    X = np.asarray(actions, dtype=float)
    if mode == "first3":
        if X.shape[1] < 3:
            pad = np.zeros((X.shape[0], 3 - X.shape[1]))
            return np.hstack([X, pad])[:, :3]
        return X[:, :3]
    elif mode == "custom":
        if kinematic_map is None:
            raise ValueError("custom mode requires kinematic_map")
        return np.asarray(kinematic_map(X), dtype=float)
    else:
        if pca_model is None:
            pca_model = PCA(n_components=3, random_state=0)
            Z = pca_model.fit_transform(X)
        else:
            Z = pca_model.transform(X)
        return Z


def plot_actions_xyz(xyz, labels, index_rows, png_path="", title="",
                     point_size=1, line_width=1.0, line_alpha=0.6,
                     line_color_mode="by_dominant_cluster",
                     start_marker_size=10, end_marker_size=20):
    """
    3D overview plot.

    - Writes a static Matplotlib PNG at `png_path`.
    - Also writes an interactive Plotly HTML with the same basename.
    """
    import matplotlib.lines as mlines
    xyz = np.asarray(xyz)
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != xyz.shape[0]:
        raise ValueError("labels length mismatch")

    # ---------------- Matplotlib static PNG ----------------
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                         c=labels, s=point_size, alpha=0.95)
    start_h = end_h = None

    if index_rows is not None and len(index_rows) == xyz.shape[0]:
        sample_idx = np.array([r[1] for r in index_rows], dtype=int)
        action_idx = np.array([r[2] for r in index_rows], dtype=int)
        unique_samples = np.unique(sample_idx)
        for s in unique_samples:
            m = (sample_idx == s)
            order = np.argsort(action_idx[m])
            pts = xyz[m][order]
            labs = labels[m][order]
            if pts.shape[0] >= 2:
                if line_color_mode == "by_dominant_cluster":
                    vals, counts = np.unique(labs, return_counts=True)
                    dom = vals[np.argmax(counts)]
                    line_color = scatter.cmap(scatter.norm(dom))
                else:
                    line_color = "k"
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        linewidth=line_width, alpha=line_alpha, color=line_color)
            p0, p1 = pts[0], pts[-1]
            ax.scatter([p0[0]], [p0[1]], [p0[2]],
                       s=start_marker_size, c="none",
                       edgecolor="k", linewidths=1.0, marker="o")
            ax.scatter([p1[0]], [p1[1]], [p1[2]],
                       s=end_marker_size, c="k",
                       marker="x", linewidths=1.5)
        start_h = mlines.Line2D([], [], color="k", marker="o", markersize=6,
                                markerfacecolor="none", linestyle="None", label="chunk start")
        end_h = mlines.Line2D([], [], color="k", marker="x", markersize=7,
                              linestyle="None", label="chunk end")

    legend_items = [mlines.Line2D([], [], color=scatter.cmap(scatter.norm(lab)),
                                  marker="s", linestyle="None", markersize=6,
                                  label=f"cluster {lab}") for lab in np.unique(labels)]
    if start_h is not None:
        legend_items.append(start_h)
    if end_h is not None:
        legend_items.append(end_h)
    if legend_items:
        ax.legend(handles=legend_items, loc="best", frameon=False, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    xm, ym, zm = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    ax.set_xlim3d([xm - max_range / 2, xm + max_range / 2])
    ax.set_ylim3d([ym - max_range / 2, ym + max_range / 2])
    ax.set_zlim3d([zm - max_range / 2, zm + max_range / 2])
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # ---------------- Plotly interactive HTML ----------------
    fig_p = go.Figure()

    fig_p.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(
                size=max(point_size, 2),
                color=labels,
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title="cluster"),
            ),
            name="points",
        )
    )

    if index_rows is not None and len(index_rows) == xyz.shape[0]:
        sample_idx = np.array([r[1] for r in index_rows], dtype=int)
        action_idx = np.array([r[2] for r in index_rows], dtype=int)
        unique_samples = np.unique(sample_idx)
        first = True
        for s in unique_samples:
            m = (sample_idx == s)
            order = np.argsort(action_idx[m])
            pts = xyz[m][order]
            if pts.shape[0] >= 2:
                fig_p.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="lines",
                        line=dict(width=5.0),
                        opacity=line_alpha,
                        showlegend=False,
                    )
                )
            p0, p1 = pts[0], pts[-1]
            fig_p.add_trace(
                go.Scatter3d(
                    x=[p0[0]],
                    y=[p0[1]],
                    z=[p0[2]],
                    mode="markers",
                    marker=dict(size=3,
                                symbol="circle-open",
                                line=dict(width=2)),
                    name="chunk start" if first else None,
                    showlegend=first,
                )
            )
            fig_p.add_trace(
                go.Scatter3d(
                    x=[p1[0]],
                    y=[p1[1]],
                    z=[p1[2]],
                    mode="markers",
                    marker=dict(size=3, symbol="x"),
                    name="chunk end" if first else None,
                    showlegend=first,
                )
            )
            first = False

    fig_p.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    base, _ = os.path.splitext(png_path)
    html_path = base + ".html"
    fig_p.write_html(html_path, include_plotlyjs="cdn")


def plot_per_cluster_panels(xyz, per_point_labels, index_rows, n_clusters,
                            png_path, title_prefix="State",
                            point_size=3, line_width=1.0, line_alpha=0.6):
    import matplotlib.lines as mlines
    xyz = np.asarray(xyz)
    labels = np.asarray(per_point_labels).reshape(-1)
    assert xyz.shape[0] == labels.shape[0] == len(index_rows)
    sample_idx = np.array([r[1] for r in index_rows], dtype=int)
    action_idx = np.array([r[2] for r in index_rows], dtype=int)
    uniq_clusters = np.unique(labels)
    C = int(max(n_clusters, uniq_clusters.max() + 1))
    cols = min(C, 4)
    rows = int(np.ceil(C / cols))
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))

    def _equal(ax):
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        xr = abs(xlim[1] - xlim[0])
        yr = abs(ylim[1] - ylim[0])
        zr = abs(zlim[1] - zlim[0])
        mr = max(xr, yr, zr)
        xm = np.mean(xlim)
        ym = np.mean(ylim)
        zm = np.mean(zlim)
        ax.set_xlim3d([xm - mr / 2, xm + mr / 2])
        ax.set_ylim3d([ym - mr / 2, ym + mr / 2])
        ax.set_zlim3d([zm - mr / 2, zm + mr / 2])

    # assign each chunk to its dominant cluster
    chunk_ids = np.unique(sample_idx)
    chunk_to_cluster = {}
    for k in chunk_ids:
        m = (sample_idx == k)
        if not np.any(m):
            continue
        vals, counts = np.unique(labels[m], return_counts=True)
        chunk_to_cluster[k] = int(vals[np.argmax(counts)])

    for ci in range(C):
        ax = fig.add_subplot(rows, cols, ci + 1, projection="3d")
        ax.set_title(f"Cluster {ci}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        chunks_in_ci = [k for k, lab in chunk_to_cluster.items() if lab == ci]
        if not chunks_in_ci:
            ax.text(0.5, 0.5, 0.5, "No chunks", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        for k in chunks_in_ci:
            m = (sample_idx == k)
            pts = xyz[m]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=point_size, alpha=0.95)
        for k in chunks_in_ci:
            m = (sample_idx == k)
            order = np.argsort(action_idx[m])
            pts = xyz[m][order]
            if pts.shape[0] >= 2:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        linewidth=line_width, alpha=line_alpha)
            p0, p1 = pts[0], pts[-1]
            ax.scatter([p0[0]], [p0[1]], [p0[2]], s=36, c="none",
                       edgecolor="k", linewidths=1.0, marker="o")
            ax.scatter([p1[0]], [p1[1]], [p1[2]], s=48, c="k",
                       marker="x", linewidths=1.5)
        _equal(ax)

    start_h = mlines.Line2D([], [], color="k", marker="o", markersize=6,
                            markerfacecolor="none", linestyle="None", label="chunk start")
    end_h = mlines.Line2D([], [], color="k", marker="x", markersize=7,
                          linestyle="None", label="chunk end")
    fig.legend(handles=[start_h, end_h], loc="upper right", frameon=False)
    fig.suptitle(f"{title_prefix}: per-cluster chunk views", y=0.995)
    fig.tight_layout(rect=[0, 0.00, 1, 0.96])
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------
# per-state worker (with checkpoint)
# ------------------------------------------------------------------------------------

def _state_file_path(per_state_dir, i):
    return os.path.join(per_state_dir, f"state_{i:06d}.npz")


def _load_state_file(path):
    data = np.load(path, allow_pickle=True)
    rows_arr = data["rows"]
    rows = list(rows_arr.tolist())

    best_arr = data["best"]

    if best_arr.size == 0:
        best = None
    else:
        b = best_arr[0]
        if isinstance(b, dict):
            best = b
        else:
            try:
                best = b.item()
            except AttributeError:
                best = b

    return rows, best


def _save_state_file(path, rows, best, labels_by_k=None):
    tmp_path = path + ".tmp"
    payload = {
        "rows": np.array(rows, dtype=object),
        "best": np.array([] if best is None else [best], dtype=object),
    }
    if labels_by_k is not None:
        payload["labels_by_k"] = np.array([labels_by_k], dtype=object)
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp_path, path)


def _process_state(i):
    """
    Run the whole clustering pipeline on state i.
    If per-state file exists, load and return it.
    Otherwise compute, save, and return.
    """
    actions_arr = _G_ACTIONS
    meta_records = _G_META
    args = _G_ARGS
    per_state_dir = args["per_state_dir"]
    debug_enabled = bool(args.get("debug_wvar", False))
    debug_vdr_threshold = float(args.get("debug_vdr_threshold", 0.9))

    state_path = _state_file_path(per_state_dir, i)
    if os.path.exists(state_path):
        rows, best_cache = _load_state_file(state_path)
        return rows, best_cache

    acts_raw = actions_arr[i]
    dbg_input_line = None
    dbg_dist_line = None
    dbg_aff_line = None
    try:
        arr = np.asarray(acts_raw)

        # NEW: integrate velocities into Δq trajectories
        trajectories, X_feat, _, _, _ = _prepare_trajectories_and_features_from_actions(
            arr,
            dt=args["dt"],
            vel_dims=args["vel_dims"],
            include_gripper=args["include_gripper"],
        )
        if debug_enabled:
            traj_lens = [int(t.shape[0]) for t in trajectories]
            traj_dims = [int(t.shape[1]) for t in trajectories]
            dbg_input_line = (
                f"[DEBUG] state={i} input: arr.shape={arr.shape} "
                f"N_chunks={len(trajectories)} X_feat.shape={X_feat.shape} "
                f"traj_len[min,max]=({min(traj_lens) if traj_lens else -1},"
                f"{max(traj_lens) if traj_lens else -1}) "
                f"traj_dim_set={sorted(set(traj_dims))}"
            )
    except Exception as e:
        sys.stderr.write(f"[skip] state {i}: {e}\n")
        rows = []
        _save_state_file(state_path, rows, None)
        return rows, None

    N_chunks = len(trajectories)

    # Optional subsample
    used_idx = None
    if args["sample_cap"] is not None and N_chunks > args["sample_cap"]:
        rng = np.random.RandomState(args["random_state"] + i)
        used_idx = rng.choice(N_chunks, args["sample_cap"], replace=False)
        trajectories = [trajectories[j] for j in used_idx]
        X_feat = X_feat[used_idx]
        N_chunks = len(trajectories)

    # metadata
    if i < len(meta_records):
        row_meta = meta_records[i]
        episode_id = row_meta.get("episode_id", f"ep{i:06d}")
        t_in_episode = int(row_meta.get("t_in_episode", -1))
        task_name = str(row_meta.get("task_name", ""))
        instruction = str(row_meta.get("instruction", ""))
    else:
        episode_id = f"ep{i:06d}"
        t_in_episode = -1
        task_name = ""
        instruction = ""

    # build affinity, distances + variance
    try:
        if args["method"] == "minkowski":
            D_mink = compute_minkowski_distance_matrix(trajectories)
        else:
            D_mink = compute_dtw_matrix(
                trajectories,
                window=args["dtw_window"],
                parallelize=args["parallel_dtw"],
                max_workers=args["max_workers"],
            )

        if debug_enabled:
            pos_dm = D_mink[D_mink > 0]
            asym = float(np.max(np.abs(D_mink - D_mink.T))) if D_mink.size else 0.0
            diag_abs_max = float(np.max(np.abs(np.diag(D_mink)))) if D_mink.size else 0.0
            dbg_dist_line = (
                f"[DEBUG] state={i} D: shape={D_mink.shape} min={float(np.min(D_mink)) if D_mink.size else np.nan} "
                f"max={float(np.max(D_mink)) if D_mink.size else np.nan} "
                f"mean_pos={float(np.mean(pos_dm)) if pos_dm.size else np.nan} "
                f"asym_max={asym} diag_abs_max={diag_abs_max}"
            )

        pos = D_mink[D_mink > 0]
        sigma_used = float(np.median(pos)) if (args["sigma"] is None and pos.size) else (args["sigma"] or 1.0)
        A = np.exp(-D_mink ** 2 / (2.0 * sigma_used ** 2))
        np.fill_diagonal(A, 1.0)

        tv = total_variance_minkowski(D_mink)
        if debug_enabled:
            dbg_aff_line = (
                f"[DEBUG] state={i} affinity: sigma_used={sigma_used} "
                f"A[min,max]=({float(np.min(A)) if A.size else np.nan},{float(np.max(A)) if A.size else np.nan}) "
                f"A_diag[min,max]=({float(np.min(np.diag(A))) if A.size else np.nan},"
                f"{float(np.max(np.diag(A))) if A.size else np.nan}) tv={tv}"
            )
    except Exception as e:
        sys.stderr.write(f"[skip] state {i}: affinity failed ({e})\n")
        rows = []
        _save_state_file(state_path, rows, None)
        return rows, None

    rows = []
    labels_by_k = {}
    best = {
        "score": None,
        "k": None,
        "wvar": None,
        "drop": None,
        "vdr": None,
        "r2": None,
        "ch": None,
        "sil": None,
        "labels": None,
        "used_idx": used_idx,
    }

    for k in range(args["k_min"], args["k_max"] + 1):
        if N_chunks < k or k < 1:
            continue
        try:
            cl = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=args["random_state"],
            )
            labels = cl.fit_predict(A)
        except Exception as e:
            sys.stderr.write(f"[warn] state {i}: spectral failed for k={k}: {e}\n")
            continue

        labels_by_k[int(k)] = labels.copy()

        # Minkowski-based variance
        wvar = weighted_incluster_variance_minkowski(D_mink, labels, debug=False)
        drop = tv - wvar
        tss = tv
        bss = drop

        var_drop_ratio = 0.0 if tss <= 0 else float(np.clip(drop / tss, 0.0, 1.0))
        r2 = var_drop_ratio
        if debug_enabled and var_drop_ratio >= debug_vdr_threshold:
            if dbg_input_line is not None:
                print(dbg_input_line, flush=True)
            if dbg_dist_line is not None:
                print(dbg_dist_line, flush=True)
            if dbg_aff_line is not None:
                print(dbg_aff_line, flush=True)
            _ = weighted_incluster_variance_minkowski(D_mink, labels, debug=True)
            u, cts = np.unique(labels, return_counts=True)
            sizes = dict(zip(u.tolist(), cts.tolist()))
            print(
                f"[DEBUG] state={i} k={k} N_chunks={N_chunks} tv={tss} wvar={wvar} "
                f"drop={drop} vdr={var_drop_ratio} threshold={debug_vdr_threshold} "
                f"cluster_sizes={sizes}",
                flush=True,
            )

        Np = N_chunks
        ch = np.nan
        if k > 1 and Np > k and wvar > 0:
            ch = (bss / (k - 1)) / (wvar / (Np - k))

        # silhouette on Euclidean features (flattened integrated chunks)
        sil = np.nan
        try:
            if Np >= 10 and len(np.unique(labels)) > 1:
                if args["silhouette_sample_cap"] and Np > args["silhouette_sample_cap"]:
                    rng = np.random.RandomState(args["random_state"] + i + k)
                    idx = rng.choice(Np, args["silhouette_sample_cap"], replace=False)
                    sil = silhouette_score(X_feat[idx], labels[idx], metric="euclidean")
                else:
                    sil = silhouette_score(X_feat, labels, metric="euclidean")
        except Exception:
            sil = np.nan

        rows.append({
            "state_index": i,
            "episode_id": episode_id,
            "t_in_episode": t_in_episode,
            "task_name": task_name,
            "instruction": instruction,
            "num_points": int(N_chunks),
            "action_dim": int(X_feat.shape[1]),
            "k": int(k),
            "total_variance": float(tss),
            "weighted_incluster_variance": float(wvar),
            "variance_drop": float(drop),
            "variance_drop_ratio": float(var_drop_ratio),
            "r2": float(r2),
            "calinski_harabasz": float(ch),
            "silhouette": float(sil),
            "best_k": False,
        })

        metric_value = {
            "variance_drop": drop,
            "variance_drop_ratio": var_drop_ratio,
            "r2": r2,
            "ch": ch,
            "silhouette": sil,
        }[args["best_metric"]]
        cmp_val = -np.inf if (metric_value is None or np.isnan(metric_value)) else metric_value

        if (best["score"] is None) or (cmp_val > best["score"] + 1e-12) or \
           (abs(cmp_val - best["score"]) <= 1e-12 and (best["k"] is None or k < best["k"])):
            best.update({
                "score": cmp_val,
                "k": k,
                "wvar": wvar,
                "drop": drop,
                "vdr": var_drop_ratio,
                "r2": r2,
                "ch": ch,
                "sil": sil,
                "labels": labels.copy(),
            })

    best_cache = None
    if best["k"] is not None:
        rows.append({
            "state_index": i,
            "episode_id": episode_id,
            "t_in_episode": t_in_episode,
            "task_name": task_name,
            "instruction": instruction,
            "num_points": int(N_chunks),
            "action_dim": int(X_feat.shape[1]),
            "k": int(best["k"]),
            "total_variance": float(tv),
            "weighted_incluster_variance": float(best["wvar"]),
            "variance_drop": float(best["drop"]),
            "variance_drop_ratio": float(best.get("vdr", 0.0)),
            "r2": float(best["r2"]),
            "calinski_harabasz": float(best["ch"]),
            "silhouette": float(best["sil"]),
            "best_k": True,
        })
        best_cache = {
            "state_index": i,
            "labels": best["labels"],     # per-chunk labels
            "idx": None if used_idx is None else used_idx.copy(),
            "k": int(best["k"]),
        }

    _save_state_file(state_path, rows, best_cache, labels_by_k=labels_by_k)
    return rows, best_cache


# ------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--actions_npz", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="multimodality_out")

    ap.add_argument("--k_min", type=int, default=5)
    ap.add_argument("--k_max", type=int, default=8)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--sample_cap", type=int, default=None)
    ap.add_argument("--best_metric", type=str,
                    choices=["variance_drop", "variance_drop_ratio", "r2", "ch", "silhouette"],
                    default="variance_drop_ratio")
    ap.add_argument("--k_selection", type=str, choices=["best", "auc"], default="best",
                    help="How to rank states for top_states.csv: "
                         "'best' uses best single-k row; "
                         "'auc' uses area-under-curve of variance_drop_ratio over k (k-independent).")
    ap.add_argument("--silhouette_sample_cap", type=int, default=5000)

    ap.add_argument("--method", type=str, choices=["minkowski", "dtw"], default="minkowski")
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--dtw_window", type=int, default=None)
    ap.add_argument("--parallel_dtw", action="store_true")
    ap.add_argument("--max_workers", type=int, default=None)

    ap.add_argument("--plot_top_n", type=int, default=20)
    ap.add_argument("--ee_mode", type=str, choices=["pca", "first3", "custom"], default="first3")

    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Number of processes to use for per-state clustering (>=1).")

    ap.add_argument("--max_states", type=int, default=None,
                    help="Process at most this many states (by index from 0).")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="Skip per-state clustering and aggregate outputs from existing "
                         "outdir/per_state checkpoints only.")

    # NEW: integration params
    ap.add_argument("--fps", type=float, default=15.0,
                    help="Control frequency for DROID (Hz). dt=1/fps used to integrate velocities.")
    ap.add_argument("--vel_dims", type=int, default=7,
                    help="Number of joint-velocity dims at start of action vector.")
    ap.add_argument("--no_gripper", action="store_true",
                    help="If set, ignore gripper dimension in clustering/plotting.")
    ap.add_argument("--debug_wvar", action="store_true",
                    help="Print debug info for suspicious high-variance-drop rows.")
    ap.add_argument("--debug_vdr_threshold", type=float, default=0.9,
                    help="Only print per-state debug when variance_drop_ratio >= this threshold.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    per_state_dir = os.path.join(args.outdir, "per_state")
    os.makedirs(per_state_dir, exist_ok=True)

    # load data
    meta_df = pd.read_csv(args.summary_csv)
    actions_arr = _load_actions_with_fallback(args.actions_npz)
    N_states = len(actions_arr)

    if len(meta_df) != N_states:
        sys.stderr.write(f"[warn] summary_csv rows ({len(meta_df)}) "
                         f"!= actions entries ({N_states}); proceeding by index.\n")

    if args.max_states is not None:
        N_effective = min(N_states, args.max_states)
    else:
        N_effective = N_states

    meta_records = meta_df.to_dict(orient="records")

    worker_args = dict(
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        sample_cap=args.sample_cap,
        best_metric=args.best_metric,
        silhouette_sample_cap=args.silhouette_sample_cap,
        method=args.method,
        sigma=args.sigma,
        dtw_window=args.dtw_window,
        parallel_dtw=args.parallel_dtw,
        max_workers=args.max_workers,
        per_state_dir=per_state_dir,

        # NEW: dt/integration controls
        dt=(1.0 / float(args.fps) if args.fps > 0 else 1.0 / 15.0),
        vel_dims=int(args.vel_dims),
        include_gripper=(not args.no_gripper),
        debug_wvar=bool(args.debug_wvar),
        debug_vdr_threshold=float(args.debug_vdr_threshold),
    )

    print(f"Processing {N_effective} states (out of {N_states}) with {args.n_jobs} worker(s)...")
    print(f"Integration: fps={args.fps} -> dt={worker_args['dt']:.6f} | vel_dims={worker_args['vel_dims']} | "
          f"include_gripper={worker_args['include_gripper']}")
    if args.debug_wvar:
        print(f"Debug filter enabled: variance_drop_ratio >= {args.debug_vdr_threshold}")

    best_labels_cache = {}

    if args.aggregate_only:
        print("Aggregate-only mode: skipping per-state clustering and using existing checkpoints.")
    elif args.n_jobs == 1:
        _init_worker(actions_arr, meta_records, worker_args)
        for i in range(N_effective):
            if i % 50 == 0:
                print(f"  state {i}/{N_effective}...")
            _, best_i = _process_state(i)
            if best_i is not None:
                best_labels_cache[best_i["state_index"]] = best_i
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.n_jobs,
                                 initializer=_init_worker,
                                 initargs=(actions_arr, meta_records, worker_args)) as ex:
            futures = {ex.submit(_process_state, i): i for i in range(N_effective)}
            for fut in as_completed(futures):
                i = futures[fut]
                if i % 50 == 0:
                    print(f"  state {i}/{N_effective}...")
                rows_i, best_i = fut.result()
                if best_i is not None:
                    best_labels_cache[best_i["state_index"]] = best_i

    # --------------------------------------------------
    # Aggregation
    # --------------------------------------------------
    all_rows = []
    for i in range(N_effective):
        state_path = _state_file_path(per_state_dir, i)
        if not os.path.exists(state_path):
            continue
        rows_i, best_i = _load_state_file(state_path)
        all_rows.extend(rows_i)
        if best_i is not None and "state_index" in best_i:
            best_labels_cache[int(best_i["state_index"])] = best_i

    if not all_rows:
        print("No rows computed — check inputs.")
        return

    metrics_df = pd.DataFrame(all_rows)
    best_df = metrics_df[metrics_df["best_k"] == True].copy()  # noqa: E712

    sort_col = {
        "variance_drop": "variance_drop",
        "variance_drop_ratio": "variance_drop_ratio",
        "r2": "r2",
        "ch": "calinski_harabasz",
        "silhouette": "silhouette",
    }[args.best_metric]

    if args.k_selection == "auc":
        eval_df_auc = metrics_df[metrics_df["best_k"] == False].copy()  # noqa: E712

        def _vdr_auc_over_k(g):
            g = g.sort_values("k")
            k = g["k"].to_numpy(dtype=float)
            v = g["variance_drop_ratio"].to_numpy(dtype=float)
            m = np.isfinite(k) & np.isfinite(v)
            if not np.any(m):
                return np.nan
            k = k[m]
            v = v[m]
            if k.size <= 1:
                return float(v[0])
            k_span = float(np.max(k) - np.min(k))
            if k_span <= 0:
                return float(np.mean(v))
            x = (k - np.min(k)) / k_span
            return float(np.trapz(v, x))

        grouped = []
        for state_idx, g in eval_df_auc.groupby("state_index"):
            g0 = g.iloc[0]
            grouped.append({
                "state_index": int(state_idx),
                "episode_id": g0.get("episode_id", f"ep{int(state_idx):06d}"),
                "t_in_episode": int(g0.get("t_in_episode", -1)),
                "num_points": int(g0.get("num_points", -1)),
                "k_independent_score": _vdr_auc_over_k(g),
            })
        best_df = pd.DataFrame(grouped).sort_values("k_independent_score", ascending=False)
    else:
        best_df.sort_values([sort_col], ascending=False, inplace=True)

    metrics_csv = os.path.join(args.outdir, "metrics_per_state.csv")
    best_csv = os.path.join(args.outdir, "top_states.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    best_df.head(args.top_n).to_csv(best_csv, index=False)
    print(f"\nSaved:\n- {os.path.abspath(metrics_csv)}\n- {os.path.abspath(best_csv)}")

    if args.debug_wvar:
        suspicious = metrics_df[
            (metrics_df["best_k"] == False) &  # noqa: E712
            (metrics_df["variance_drop_ratio"] >= float(args.debug_vdr_threshold))
        ].copy()
        suspicious.sort_values("variance_drop_ratio", ascending=False, inplace=True)
        suspicious_csv = os.path.join(args.outdir, "suspicious_variance_drop_rows.csv")
        suspicious.to_csv(suspicious_csv, index=False)
        print(
            f"\nSuspicious rows (best_k=False, variance_drop_ratio >= {args.debug_vdr_threshold}): "
            f"{len(suspicious)}"
        )
        if len(suspicious) > 0:
            cols = [
                "state_index", "k", "num_points", "total_variance",
                "weighted_incluster_variance", "variance_drop_ratio"
            ]
            print(suspicious[cols].head(50).to_string(index=False))
        print(f"Saved suspicious rows: {os.path.abspath(suspicious_csv)}")

    if args.k_selection == "auc":
        print("\nTop states by k-independent variance_drop_ratio AUC:")
        cols_to_show = [
            "state_index", "episode_id", "t_in_episode", "num_points", "k_independent_score"
        ]
    else:
        print(f"\nTop states (best-k) by {sort_col}:")
        cols_to_show = [
            "state_index", "episode_id", "t_in_episode", "k", "num_points",
            "total_variance", "weighted_incluster_variance",
            "variance_drop_ratio", "r2", "calinski_harabasz", "silhouette"
        ]
    print(best_df[cols_to_show].head(args.top_n).to_string(index=False))

    # ---------- plots ----------
    if "weighted_incluster_variance" in best_df.columns:
        hist_path = os.path.join(args.outdir, "variance_hist.png")
        plt.figure(figsize=(7, 5))
        plt.hist(best_df["weighted_incluster_variance"].values, bins=40)
        plt.xlabel("Weighted in-cluster variance (best k per state)")
        plt.ylabel("Count of states")
        plt.title("Histogram of weighted in-cluster variance (best-k)")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"\nSaved histogram: {os.path.abspath(hist_path)}")
    else:
        print("\nSkipping best-k weighted-variance histogram in --k_selection=auc mode.")

    # per-k histos for weighted_incluster_variance + violin for variance_drop_ratio
    eval_df = metrics_df[metrics_df["best_k"] == False].copy()
    ks = sorted(eval_df["k"].unique().tolist())
    all_vals = eval_df["weighted_incluster_variance"].values
    if all_vals.size:
        bins = np.histogram_bin_edges(all_vals, bins=40)

        overlay_path = os.path.join(args.outdir, "variance_hist_by_k_overlay.png")
        plt.figure(figsize=(8, 6))
        for k in ks:
            vals = eval_df.loc[eval_df["k"] == k, "weighted_incluster_variance"].values
            if vals.size == 0:
                continue
            plt.hist(vals, bins=bins, histtype="step", linewidth=1.5, label=f"k={k}")
        plt.xlabel("Weighted in-cluster variance (per specific k)")
        plt.ylabel("Count of states")
        plt.title("Per-k histogram (overlay)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150)
        plt.close()
        print(f"Saved per-k overlay histogram: {os.path.abspath(overlay_path)}")

        grid_path = os.path.join(args.outdir, "variance_hist_by_k_grid.png")
        n = len(ks)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                                 squeeze=False, sharex=True, sharey=True)
        for idx, k in enumerate(ks):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            vals = eval_df.loc[eval_df["k"] == k, "weighted_incluster_variance"].values
            ax.hist(vals, bins=bins)
            ax.set_title(f"k={k}")
            ax.set_xlabel("weighted in-cluster var")
            ax.set_ylabel("count")
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")
        fig.suptitle("Per-k histograms (weighted in-cluster variance)", y=0.995)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        print(f"Saved per-k grid histogram: {os.path.abspath(grid_path)}")

        # Violin plot of variance-drop ratio by k
        violin_path = os.path.join(args.outdir, "variance_drop_ratio_by_k_violin.png")

        data_v = []
        positions = []
        xticklabels = []
        pos_counter = 1
        for k in ks:
            vals = eval_df.loc[eval_df["k"] == k, "variance_drop_ratio"].values
            if vals.size == 0:
                continue
            data_v.append(vals)
            positions.append(pos_counter)
            xticklabels.append(f"k={k}")
            pos_counter += 1

        if data_v:
            plt.figure(figsize=(max(6, 1.5 * len(positions)), 5))
            plt.violinplot(
                data_v,
                positions=positions,
                showmeans=True,
                showextrema=False,
                widths=0.8,
                points=30,
                bw_method=0.2,
            )
            plt.xticks(positions, xticklabels)
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Variance drop ratio ((TV - WVAR) / TV)")
            plt.title("Variance-drop ratio distribution by k")
            plt.tight_layout()
            plt.savefig(violin_path, dpi=150)
            plt.close()
            print(f"Saved variance-drop ratio violin plot: {os.path.abspath(violin_path)}")

    # ========================== Top-N visualizations (chunk-aware) ==========================
    top_plot_dir = os.path.join(args.outdir, "top_plots")
    os.makedirs(top_plot_dir, exist_ok=True)

    if args.k_selection == "auc":
        eval_df_for_plot = metrics_df[metrics_df["best_k"] == False].copy()  # noqa: E712
        idxmax = eval_df_for_plot.groupby("state_index")["variance_drop_ratio"].idxmax()
        rep_k_df = eval_df_for_plot.loc[idxmax, ["state_index", "k", "variance_drop_ratio"]].copy()
        top_rows = (
            best_df.head(args.plot_top_n)
            .merge(rep_k_df, on="state_index", how="left", suffixes=("", "_repk"))
            .reset_index(drop=True)
        )
        top_score_col = "k_independent_score"
        print(f"\nRendering cluster plots for top {len(top_rows)} states (by k-independent AUC)...")
    else:
        top_rows = best_df.head(args.plot_top_n).reset_index(drop=True)
        top_score_col = sort_col
        print(f"\nRendering cluster plots for top {len(top_rows)} states (by {sort_col})...")

    for rank, row in enumerate(top_rows.itertuples()):
        state_idx = int(row.state_index)
        ep_id = str(row.episode_id)
        t_in_ep = int(row.t_in_episode)
        if not np.isfinite(getattr(row, "k", np.nan)):
            sys.stderr.write(f"[warn] Skipping state {state_idx}: no representative k available.\n")
            continue
        best_k = int(row.k)

        acts_raw = actions_arr[state_idx]
        arr = np.asarray(acts_raw)

        # NEW: build integrated per-chunk trajectories and per-point plot matrix
        try:
            trajectories, _, traj_matrix, chunk_ids, time_ids = _prepare_trajectories_and_features_from_actions(
                arr,
                dt=worker_args["dt"],
                vel_dims=worker_args["vel_dims"],
                include_gripper=worker_args["include_gripper"],
            )
        except Exception as e:
            sys.stderr.write(f"[warn] Skipping state {state_idx}: {e}\n")
            continue

        # inferred chunk/step counts for later mapping
        if arr.ndim == 3:
            Kc, Tc, _A = arr.shape
        else:
            Kc = arr.shape[0]
            Tc = 1

        cache = best_labels_cache.get(state_idx, None)

        if cache is not None and cache.get("labels") is not None:
            chunk_labels_cached = np.asarray(cache["labels"])
            if cache.get("idx") is not None and len(cache["idx"]) > 0:
                used_idx_for_plot = np.asarray(cache["idx"], dtype=int)
                chunk_labels = chunk_labels_cached
            else:
                used_idx_for_plot = None
                chunk_labels = chunk_labels_cached
        else:
            if args.method == "minkowski":
                D_mink = compute_minkowski_distance_matrix(trajectories)
            else:
                D_mink = compute_dtw_matrix(
                    trajectories,
                    window=args.dtw_window,
                    parallelize=args.parallel_dtw,
                    max_workers=args.max_workers,
                )
            pos = D_mink[D_mink > 0]
            sigma_used = float(np.median(pos)) if (args.sigma is None and pos.size) else (args.sigma or 1.0)
            A_plot = np.exp(-D_mink ** 2 / (2.0 * sigma_used ** 2))
            np.fill_diagonal(A_plot, 1.0)

            cl = SpectralClustering(
                n_clusters=best_k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=args.random_state,
            )
            chunk_labels = cl.fit_predict(A_plot)
            used_idx_for_plot = None

        # map chunk-level labels -> per-point labels
        if used_idx_for_plot is None:
            labels_per_point = chunk_labels[chunk_ids]
            traj_matrix_plot = traj_matrix
            chunk_ids_plot = chunk_ids
            time_ids_plot = time_ids
        else:
            chunk_label_for_orig = np.full(Kc, -1, dtype=int)
            for local_idx, orig_idx in enumerate(used_idx_for_plot):
                if 0 <= orig_idx < Kc:
                    chunk_label_for_orig[orig_idx] = chunk_labels[local_idx]

            mask = chunk_label_for_orig[chunk_ids] >= 0
            if not np.any(mask):
                sys.stderr.write(f"[warn] No points to plot for state {state_idx} after applying used_idx.\n")
                continue

            traj_matrix_plot = traj_matrix[mask]
            chunk_ids_plot = chunk_ids[mask]
            time_ids_plot = time_ids[mask]
            labels_per_point = chunk_label_for_orig[chunk_ids_plot]

        xyz = to_xyz(traj_matrix_plot, mode=args.ee_mode)
        index_rows = [
            (state_idx, int(c), int(t))
            for c, t in zip(chunk_ids_plot, time_ids_plot)
        ]

        sub = os.path.join(top_plot_dir, f"rank{rank + 1:02d}_{ep_id}_state_{t_in_ep:06d}")
        os.makedirs(sub, exist_ok=True)

        title = (f"ep={ep_id} state={t_in_ep} | best-k={best_k} | "
                 f"score={getattr(row, top_score_col):.4f} | integrated(dt={worker_args['dt']:.4f})")
        overview_png = os.path.join(sub, "overview.png")
        percluster_png = os.path.join(sub, "percluster.png")

        plot_actions_xyz(
            xyz, labels_per_point, index_rows,
            png_path=overview_png, title=title,
            point_size=1, line_width=1.0, line_alpha=0.4,
            line_color_mode="by_dominant_cluster",
        )

        n_cls = len(np.unique(labels_per_point))
        plot_per_cluster_panels(
            xyz, labels_per_point, index_rows, n_clusters=n_cls,
            png_path=percluster_png, title_prefix=f"ep={ep_id} state={t_in_ep}",
            point_size=1, line_width=1.2, line_alpha=0.55,
        )

        if used_idx_for_plot is None:
            idx_to_save = np.array([], dtype=int)
        else:
            idx_to_save = used_idx_for_plot.astype(int, copy=False)

        np.savez_compressed(
            os.path.join(args.outdir, f"best_labels_state_{state_idx:06d}.npz"),
            labels=np.asarray(chunk_labels, dtype=int),
            idx=idx_to_save,
            k=best_k,
        )

    print(f"Top-N plots written under: {os.path.abspath(top_plot_dir)}")


if __name__ == "__main__":
    main()
