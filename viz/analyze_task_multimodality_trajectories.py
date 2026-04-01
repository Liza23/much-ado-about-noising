#!/usr/bin/env python3
"""
analyze_task_multimodality_trajectories.py

UPDATED PLOTTING:
- State anchors are joint positions q(t) loaded from LeRobot parquet (if provided).
- Ghost trajectories integrate joint-velocity actions into q(t) trajectories at 15Hz (configurable).
- One GLOBAL PCA(3) is fit across all plotted points (anchors + ghosts) and used everywhere.

If --lerobot_root is NOT provided, we fall back to the old behavior:
- anchors/ghosts in action space using first3 or per-episode PCA.
"""

import argparse
import os
import sys
import re
import glob
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ----------------------------------------------------------------------
# LeRobot parquet discovery/loading (same logic as before)
# ----------------------------------------------------------------------

def _discover_lerobot_parquets(data_root: str):
    patterns = [
        os.path.join(data_root, "data", "chunk-*", "file-*.parquet"),
        os.path.join(data_root, "data", "chunk-*", "*.parquet"),
        os.path.join(data_root, "data", "*.parquet"),
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            yield path

    if seen:
        return

    for path in sorted(glob.glob(os.path.join(data_root, "**", "*.parquet"), recursive=True)):
        if path in seen:
            continue
        seen.add(path)
        yield path


def _parse_episode_id_to_index(ep_id: str) -> int:
    if ep_id is None:
        raise ValueError("episode_id is None")
    m = re.match(r"ep(\d+)", str(ep_id).strip())
    if not m:
        raise ValueError(f"Bad episode_id format: {ep_id} (expected ep######)")
    return int(m.group(1))


def _pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_lerobot_states_for_keys(
    lerobot_root: str,
    needed_keys,                # set of (episode_index:int, frame_index:int)
    state_col_candidates=None,  # list of column names to try
    max_rows_per_parquet=None,
    frame_tolerance: int = 0,   # allow +/- this many frames as fallback
):
    def _coerce_to_int(arr):
        out = []
        for v in arr:
            if pd.isna(v):
                out.append(None)
                continue
            try:
                out.append(int(v))
                continue
            except Exception:
                pass
            m = re.search(r"(\d+)", str(v))
            out.append(int(m.group(1)) if m else None)
        return np.array([(-1 if v is None else int(v)) for v in out], dtype=int)

    if state_col_candidates is None:
        state_col_candidates = [
            "observation.state",
            "observation/robot_state",
            "robot_state",
            "state",
            "observation.qpos",
            "qpos",
        ]

    needed_keys = set((int(e), int(t)) for (e, t) in needed_keys)
    needed_eps = set(e for (e, _) in needed_keys)
    needed_by_ep = {}
    for (e, t) in needed_keys:
        needed_by_ep.setdefault(int(e), set()).add(int(t))

    state_map = {}
    if not needed_keys:
        return state_map

    parquets = list(_discover_lerobot_parquets(lerobot_root))
    if not parquets:
        sys.stderr.write(f"[lerobot][warn] no parquet files found under {lerobot_root}\n")

    for parquet_path in parquets:
        df = pd.read_parquet(parquet_path)
        if max_rows_per_parquet is not None and len(df) > max_rows_per_parquet:
            df = df.iloc[:max_rows_per_parquet].copy()

        ep_col = _pick_first_existing_col(df, ["episode_index", "episode_id", "episode", "episode_uid"])
        fr_col = _pick_first_existing_col(df, ["frame_index", "frame_id", "frame", "timestep", "step_index", "step_id"])
        if ep_col is None or fr_col is None:
            continue

        eps = pd.Series(_coerce_to_int(df[ep_col].to_numpy()), index=df.index)
        keep_ep = eps.isin(list(needed_eps))
        if not keep_ep.any():
            continue
        df = df.loc[keep_ep].copy()

        st_col = _pick_first_existing_col(df, state_col_candidates)
        if st_col is None:
            continue

        ep_idx = pd.Series(_coerce_to_int(df[ep_col].to_numpy()), index=df.index).to_numpy()
        fr_idx = pd.Series(_coerce_to_int(df[fr_col].to_numpy()), index=df.index).to_numpy()
        st_vals = df[st_col].to_numpy()

        for i in range(len(df)):
            key = (int(ep_idx[i]), int(fr_idx[i]))
            ep = key[0]
            fr = key[1]

            if key in needed_keys and key not in state_map:
                v = st_vals[i]
                arr = np.asarray(v, dtype=float).reshape(-1)
                state_map[key] = arr
                continue

            if frame_tolerance > 0 and ep in needed_by_ep:
                targets = needed_by_ep[ep]
                near = [t for t in targets if abs(fr - t) <= frame_tolerance]
                if not near:
                    continue
                target = sorted(near, key=lambda t: abs(fr - t))[0]
                target_key = (ep, target)
                if target_key in state_map:
                    continue
                v = st_vals[i]
                arr = np.asarray(v, dtype=float).reshape(-1)
                state_map[target_key] = arr

        if len(state_map) >= len(needed_keys):
            break

    return state_map


# ----------------------------------------------------------------------
# Integration helpers (NEW)
# ----------------------------------------------------------------------

def integrate_vel_chunk_to_q(chunk_vel, q0, dt, q_dim=7):
    """
    chunk_vel: (T, A) where first q_dim are joint velocities
    q0: (q_dim,)
    returns q_path: (T+1, q_dim) including q0
    """
    chunk_vel = np.asarray(chunk_vel, dtype=float)
    if chunk_vel.ndim != 2:
        raise ValueError(f"chunk must be 2D (T,A), got {chunk_vel.shape}")

    T, A = chunk_vel.shape
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    if q0.shape[0] < q_dim:
        raise ValueError(f"q0 dim {q0.shape[0]} < q_dim={q_dim}")

    dq = chunk_vel[:, :q_dim]
    q_path = np.zeros((T + 1, q_dim), dtype=float)
    q_path[0] = q0[:q_dim]
    for t in range(T):
        q_path[t + 1] = q_path[t] + dq[t] * dt
    return q_path


# ----------------------------------------------------------------------
# Embedding helpers (UPDATED)
# ----------------------------------------------------------------------

def embed_points_3d(X, mode="first3", pca_model=None):
    """
    X: (N, D) -> (N, 3)
    mode:
      - first3: first 3 dims (pad if needed)
      - pca: use provided pca_model (must be fit); if None, fit per-call (not recommended)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    if mode == "first3":
        if X.shape[1] < 3:
            pad = np.zeros((X.shape[0], 3 - X.shape[1]), dtype=float)
            return np.hstack([X, pad])[:, :3]
        return X[:, :3]

    # pca
    if pca_model is None:
        pca_model = PCA(n_components=3, random_state=0).fit(X)
    return pca_model.transform(X)


# ----------------------------------------------------------------------
# Plotly helpers (same colormap logic)
# ----------------------------------------------------------------------

viridis = matplotlib.colormaps.get_cmap("viridis")

def ratio_to_rgba(ratio, r_min, r_max, alpha=1.0):
    if r_max > r_min:
        norm = (ratio - r_min) / (r_max - r_min)
    else:
        norm = 0.5
    norm = float(np.clip(norm, 0.0, 1.0))
    r, g, b, _ = viridis(norm)
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"


def plot_episode_trajectory_with_ghosts(
    episode_id,
    band,
    task_label,
    ep_df,
    actions_arr,
    global_r_min,
    global_r_max,
    out_html,
    max_chunks_per_state=5,
    embed_mode="pca",          # "first3" or "pca"
    pca_model=None,            # GLOBAL PCA model (fit once)
    # NEW (for integrated plotting)
    use_integrated_q=False,
    q0_map=None,               # dict: (episode_id_str, t_in_episode_int) -> q0 (q_dim,)
    dt=1/15.0,
    q_dim=7,
):
    """
    If use_integrated_q:
      - anchors = q(t) from q0_map
      - ghost chunk lines = integrate joint velocities from q(t)
    Else:
      - old behavior: anchors derived from action chunk start/end
      - ghost lines in raw action space
    """
    ep_df = ep_df.sort_values("t_in_episode").reset_index(drop=True)

    state_indices = ep_df["state_index"].astype(int).to_numpy()
    t_steps = ep_df["t_in_episode"].astype(int).to_numpy()
    ratios = ep_df["variance_drop_ratio"].to_numpy()

    fig = go.Figure()

    # -------------------------
    # Anchors
    # -------------------------
    if use_integrated_q:
        if q0_map is None:
            raise ValueError("use_integrated_q=True requires q0_map")

        anchors_q = []
        anchors_keep = []
        for t in t_steps:
            key = (str(episode_id), int(t))
            q = q0_map.get(key, None)
            if q is None:
                continue
            anchors_q.append(np.asarray(q, float)[:q_dim])
            anchors_keep.append(True)

        anchors_q = np.vstack(anchors_q)
        # also filter state_indices/t_steps/ratios to those kept (so everything aligns)
        xyz_states = embed_points_3d(anchors_q, mode=embed_mode, pca_model=pca_model)

    else:
        # old anchor logic: start of first, end of previous
        def state_anchor_from_chunks(arr, which="start"):
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 3:
                if which == "start":
                    pts = arr[:, 0, :]
                else:
                    pts = arr[:, -1, :]
                return pts.mean(axis=0)
            elif arr.ndim == 2:
                return arr[0, :] if which == "start" else arr[-1, :]
            else:
                raise ValueError(f"Unsupported action shape: {arr.shape}")

        anchors = []
        for i, s in enumerate(state_indices):
            if i == 0:
                anchors.append(state_anchor_from_chunks(actions_arr[s], which="start"))
            else:
                s_prev = state_indices[i - 1]
                anchors.append(state_anchor_from_chunks(actions_arr[s_prev], which="end"))

        anchors = np.vstack(anchors)
        xyz_states = embed_points_3d(anchors, mode=embed_mode, pca_model=pca_model)

    # trajectory line through anchors
    fig.add_trace(go.Scatter3d(
        x=xyz_states[:, 0], y=xyz_states[:, 1], z=xyz_states[:, 2],
        mode="lines",
        line=dict(width=3),
        opacity=0.45,
        showlegend=False,
        name="trajectory",
    ))

    # anchor markers colored by ratio
    fig.add_trace(go.Scatter3d(
        x=xyz_states[:, 0], y=xyz_states[:, 1], z=xyz_states[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=ratios,
            colorscale="Viridis",
            cmin=global_r_min,
            cmax=global_r_max,
            colorbar=dict(title="variance_drop_ratio"),
        ),
        text=[f"task={task_label}<br>t={t}<br>ratio={r:.4f}" for t, r in zip(t_steps, ratios)],
        hoverinfo="text",
        name="states",
    ))

    # -------------------------
    # Ghost trajectories
    # -------------------------
    ghost_line_width = 2.5
    ghost_opacity = 0.35
    ghost_visible = "legendonly"  # toggle per-state via legend for clarity
    ghost_legend_shown_by_state = {}
    rng0 = np.random.RandomState(0)

    for idx, s in enumerate(state_indices):
        arr = np.asarray(actions_arr[s], dtype=float)
        ratio_s = float(ratios[idx])
        color_s = ratio_to_rgba(ratio_s, global_r_min, global_r_max, alpha=1.0)

        if use_integrated_q:
            # anchor q(t) for this state
            key = (str(episode_id), int(t_steps[idx]))
            q_anchor = np.asarray(q0_map.get(key, np.zeros(q_dim)), dtype=float)[:q_dim]
            ghost_group = f"ghost-{int(s)}"
            ghost_label = f"ghosts state={int(s)}"

            if arr.ndim == 3:
                Kc = arr.shape[0]
                chunk_indices = np.arange(Kc)
                if Kc > max_chunks_per_state:
                    rng = np.random.RandomState(1337 + int(s))
                    chunk_indices = rng.choice(Kc, size=max_chunks_per_state, replace=False)

                for ck in chunk_indices:
                    q_path = integrate_vel_chunk_to_q(arr[ck], q_anchor, dt=dt, q_dim=q_dim)  # (T+1,q_dim)
                    chunk_xyz = embed_points_3d(q_path, mode=embed_mode, pca_model=pca_model)
                    fig.add_trace(go.Scatter3d(
                        x=chunk_xyz[:, 0], y=chunk_xyz[:, 1], z=chunk_xyz[:, 2],
                        mode="lines",
                        line=dict(color=color_s, width=ghost_line_width),
                        opacity=ghost_opacity,
                        name=ghost_label,
                        legendgroup=ghost_group,
                        showlegend=not ghost_legend_shown_by_state.get(int(s), False),
                        visible=ghost_visible,
                        hoverinfo="none",
                    ))
                    ghost_legend_shown_by_state[int(s)] = True

            elif arr.ndim == 2:
                q_path = integrate_vel_chunk_to_q(arr, q_anchor, dt=dt, q_dim=q_dim)
                chunk_xyz = embed_points_3d(q_path, mode=embed_mode, pca_model=pca_model)
                fig.add_trace(go.Scatter3d(
                    x=chunk_xyz[:, 0], y=chunk_xyz[:, 1], z=chunk_xyz[:, 2],
                    mode="lines",
                    line=dict(color=color_s, width=ghost_line_width),
                    opacity=ghost_opacity,
                    name=ghost_label,
                    legendgroup=ghost_group,
                    showlegend=not ghost_legend_shown_by_state.get(int(s), False),
                    visible=ghost_visible,
                    hoverinfo="none",
                ))
                ghost_legend_shown_by_state[int(s)] = True
            else:
                sys.stderr.write(f"[warn] episode {episode_id}: state {s} unsupported action shape {arr.shape}\n")

        else:
            # old ghost in action space, embed using first3/pca
            def _embed_chunk_action(chunk):
                chunk = np.asarray(chunk, dtype=float)
                if chunk.ndim != 2:
                    raise ValueError("chunk must be (T,A)")
                return embed_points_3d(chunk, mode=embed_mode, pca_model=pca_model)

            ghost_group = f"ghost-{int(s)}"
            ghost_label = f"ghosts state={int(s)}"

            if arr.ndim == 3:
                Kc = arr.shape[0]
                chunk_indices = np.arange(Kc)
                if Kc > max_chunks_per_state:
                    chunk_indices = rng0.choice(Kc, size=max_chunks_per_state, replace=False)
                for ck in chunk_indices:
                    chunk_xyz = _embed_chunk_action(arr[ck])
                    fig.add_trace(go.Scatter3d(
                        x=chunk_xyz[:, 0], y=chunk_xyz[:, 1], z=chunk_xyz[:, 2],
                        mode="lines",
                        line=dict(color=color_s, width=ghost_line_width),
                        opacity=ghost_opacity,
                        name=ghost_label,
                        legendgroup=ghost_group,
                        showlegend=not ghost_legend_shown_by_state.get(int(s), False),
                        visible=ghost_visible,
                        hoverinfo="none",
                    ))
                    ghost_legend_shown_by_state[int(s)] = True
            elif arr.ndim == 2:
                chunk_xyz = _embed_chunk_action(arr)
                fig.add_trace(go.Scatter3d(
                    x=chunk_xyz[:, 0], y=chunk_xyz[:, 1], z=chunk_xyz[:, 2],
                    mode="lines",
                    line=dict(color=color_s, width=ghost_line_width),
                    opacity=ghost_opacity,
                    name=ghost_label,
                    legendgroup=ghost_group,
                    showlegend=not ghost_legend_shown_by_state.get(int(s), False),
                    visible=ghost_visible,
                    hoverinfo="none",
                ))
                ghost_legend_shown_by_state[int(s)] = True
            else:
                sys.stderr.write(f"[warn] episode {episode_id}: state {s} unsupported action shape {arr.shape}\n")

    fig.update_layout(
        title=f"Task={task_label} | Episode {episode_id} ({band}) | states colored by variance_drop_ratio",
        scene=dict(xaxis_title="PC1" if embed_mode == "pca" else "X",
                   yaxis_title="PC2" if embed_mode == "pca" else "Y",
                   zaxis_title="PC3" if embed_mode == "pca" else "Z"),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(groupclick="togglegroup"),
    )

    fig.write_html(out_html)
    print(f"  wrote {out_html}")


# ----------------------------------------------------------------------
# Task labeling + distribution plotting (UNCHANGED from your script)
# ----------------------------------------------------------------------

def normalize_tasks(tasks_csv):
    return [t.strip().lower() for t in tasks_csv.split(",") if t.strip()]


def task_match_label(row, tasks, source="both"):
    tn = str(row.get("task_name", ""))
    instr = str(row.get("instruction", ""))

    if source == "task_name":
        hay = tn
    elif source == "instruction":
        hay = instr
    else:
        hay = f"{tn} {instr}"

    h = hay.lower()
    for t in tasks:
        if t in h:
            return t
    return None


def compute_traj_stat(values, stat):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    if stat == "mean":
        return float(np.mean(x))
    if stat == "max":
        return float(np.max(x))
    if stat == "median":
        return float(np.median(x))
    if stat == "p80":
        return float(np.quantile(x, 0.80))
    if stat == "p90":
        return float(np.quantile(x, 0.90))
    if stat == "min":
        return float(np.min(x))
    raise ValueError(f"Unknown traj_stat: {stat}")


def violin_with_jitter(ax, groups, labels, title, ylabel):
    positions = np.arange(1, len(groups) + 1)
    ax.violinplot(groups, positions=positions, showmedians=True, showextrema=True)

    rng = np.random.RandomState(0)
    for i, vals in enumerate(groups, start=1):
        vals = np.asarray(vals)
        if vals.size == 0:
            continue
        xj = i + 0.06 * rng.randn(vals.size)
        ax.scatter(xj, vals, s=12, alpha=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Task")


def plot_over_time_median_iqr(out_png, time_stats_df, tasks, k_fixed):
    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    for task in tasks:
        sub = time_stats_df[time_stats_df["task"] == task].sort_values("t_norm")
        if sub.empty:
            continue

        t = sub["t_norm"].to_numpy()
        med = sub["median"].to_numpy()
        q25 = sub["q25"].to_numpy()
        q75 = sub["q75"].to_numpy()

        use_markers = (len(t) <= 25)
        marker = "." if use_markers else None
        ms = 3 if use_markers else None

        line = ax.plot(t, med, linewidth=2, marker=marker, markersize=ms, label=task)
        color = line[0].get_color()
        ax.fill_between(t, q25, q75, alpha=0.18, color=color)

    ax.set_title(f"State multimodality over time (median ± IQR), k={k_fixed}")
    ax.set_xlabel("normalized time (0 → start, 1 → end)")
    ax.set_ylabel("variance_drop_ratio")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_over_time_variance(out_png, time_var_df, tasks, k_fixed):
    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    for task in tasks:
        sub = time_var_df[time_var_df["task"] == task].sort_values("t_norm")
        if sub.empty:
            continue

        t = sub["t_norm"].to_numpy()
        v = sub["var"].to_numpy()

        use_markers = (len(t) <= 25)
        marker = "." if use_markers else None
        ms = 3 if use_markers else None

        ax.plot(t, v, linewidth=2, marker=marker, markersize=ms, label=task)

    ax.set_title(f"Variance of multimodality over time, k={k_fixed}")
    ax.set_xlabel("normalized time (0 → start, 1 → end)")
    ax.set_ylabel("Var(variance_drop_ratio)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, required=True)
    ap.add_argument("--actions_npz", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--tasks", type=str, default="pour")
    ap.add_argument("--task_source", type=str, default="both", choices=["both", "task_name", "instruction"])

    ap.add_argument("--traj_stat", type=str, default="max",
                    choices=["mean", "max", "median", "p80", "p90", "min"])
    ap.add_argument("--episodes_per_band", type=int, default=3)

    # embedding: now "pca" is recommended (global PCA)
    ap.add_argument("--embedding_mode", type=str, default="pca", choices=["first3", "pca"])

    ap.add_argument("--max_t", type=int, default=None)
    ap.add_argument("--k", type=int, default=None,
                    help="Single k value to analyze (deprecated; use --ks for multiple).")
    ap.add_argument("--ks", type=str, default=None,
                    help="Comma-separated list of k values to analyze, e.g. '3,5,7'.")

    # NEW: integration + LeRobot state access
    ap.add_argument("--lerobot_root", type=str, default=None,
                    help="If set, use q(t) anchors from parquet + integrate actions into q(t).")
    ap.add_argument("--lerobot_state_col", type=str, default=None,
                    help="Override parquet state column name (e.g. observation.state).")
    ap.add_argument("--lerobot_max_rows_per_parquet", type=int, default=None)
    ap.add_argument("--lerobot_frame_tolerance", type=int, default=0)

    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--q_dim", type=int, default=7)

    ap.add_argument("--pca_max_points", type=int, default=200000,
                    help="Cap points used to fit global PCA (speed/memory).")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plot_dir_root = os.path.join(args.outdir, "plots")
    os.makedirs(plot_dir_root, exist_ok=True)
    episode_plot_root = os.path.join(args.outdir, "episode_plots")
    os.makedirs(episode_plot_root, exist_ok=True)

    dt = 1.0 / float(args.hz)
    q_dim = int(args.q_dim)

    tasks = normalize_tasks(args.tasks)
    if not tasks:
        raise ValueError("--tasks parsed to empty list")
    print(f"Tasks: {tasks} (matching source={args.task_source})")

    metrics_df = pd.read_csv(args.metrics_csv)
    actions_data = np.load(args.actions_npz, allow_pickle=True)
    actions_arr = actions_data["actions"]

    required_cols = {"state_index", "episode_id", "t_in_episode", "task_name", "instruction",
                     "total_variance", "variance_drop", "k"}
    missing = required_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_csv missing required columns: {missing}")

    tv = metrics_df["total_variance"].to_numpy()
    vd = metrics_df["variance_drop"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics_df["variance_drop_ratio"] = np.where(tv > 0, vd / tv, np.nan)

    if args.ks is not None:
        k_list = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    elif args.k is not None:
        k_list = [int(args.k)]
    else:
        raise ValueError("Provide --k or --ks")

    if not k_list:
        raise ValueError("--ks parsed to empty list")

    for k_fixed in k_list:
        metrics_k = metrics_df[metrics_df["k"] == k_fixed].copy()
        if metrics_k.empty:
            print(f"[warn] no rows found with k={k_fixed} in metrics_csv; skipping.")
            continue
        print(f"Analyzing multimodality at k={k_fixed}")

        metrics_k["task_label"] = metrics_k.apply(
            lambda r: task_match_label(r, tasks, source=args.task_source),
            axis=1
        )

        df = metrics_k[metrics_k["task_label"].notna()].copy()
        if df.empty:
            print("No rows matched any tasks. Nothing to do for this k.")
            continue

        df["t_in_episode"] = pd.to_numeric(df["t_in_episode"], errors="coerce")
        df = df[df["t_in_episode"].notna()].copy()
        df["t_in_episode"] = df["t_in_episode"].astype(int)

        if args.max_t is not None:
            df = df[df["t_in_episode"] <= int(args.max_t)].copy()

        # normalized time bins
        ep_tmax = df.groupby("episode_id")["t_in_episode"].transform("max").astype(float)
        df["t_norm"] = np.where(ep_tmax > 0, df["t_in_episode"] / ep_tmax, 0.0)

        N_TIME_BINS = 50
        df["t_bin"] = np.clip(np.round(df["t_norm"] * (N_TIME_BINS - 1)).astype(int), 0, N_TIME_BINS - 1)
        df["t_norm_bin"] = df["t_bin"] / float(N_TIME_BINS - 1)

        valid_all = df["variance_drop_ratio"].to_numpy()
        valid_all = valid_all[~np.isnan(valid_all)]
        if valid_all.size == 0:
            print("No valid variance_drop_ratio values after filtering.")
            continue
        global_r_min = float(np.min(valid_all))
        global_r_max = float(np.max(valid_all))
        print(f"Global ratio range across selected tasks: [{global_r_min:.4f}, {global_r_max:.4f}]")

        plot_dir = os.path.join(plot_dir_root, f"k_{k_fixed:02d}")
        os.makedirs(plot_dir, exist_ok=True)
        episode_plot_root_k = os.path.join(episode_plot_root, f"k_{k_fixed:02d}")
        os.makedirs(episode_plot_root_k, exist_ok=True)

        # ------------------------------------------------------------------
        # 1) Per-trajectory summary distribution
        # ------------------------------------------------------------------
        traj_rows = []
        for (task, ep_id), g in df.groupby(["task_label", "episode_id"]):
            vals = g["variance_drop_ratio"].to_numpy()
            stat_val = compute_traj_stat(vals, args.traj_stat)
            traj_rows.append({
                "task": task,
                "episode_id": ep_id,
                "n_states": len(g),
                f"{args.traj_stat}_ratio": stat_val,
                "max_ratio": compute_traj_stat(vals, "max"),
                "mean_ratio": compute_traj_stat(vals, "mean"),
                "median_ratio": compute_traj_stat(vals, "median"),
            })

        traj_df = pd.DataFrame(traj_rows)
        task_summary_csv = os.path.join(plot_dir, "task_episode_summary.csv")
        traj_df.to_csv(task_summary_csv, index=False)
        print(f"Wrote: {task_summary_csv}")

        pairs = []
        for t in tasks:
            sub_df = traj_df[traj_df["task"] == t].copy()
            vals = sub_df[f"{args.traj_stat}_ratio"].dropna().to_numpy()
            if vals.size == 0:
                continue
            n_traj = sub_df["episode_id"].nunique()
            pairs.append((f"{t} (n={n_traj})", vals))

        if pairs:
            labels, groups = zip(*pairs)
            out_violin = os.path.join(plot_dir, f"traj_summary_violin_{args.traj_stat}.png")
            plt.figure(figsize=(max(7, 1.2 * len(groups)), 5))
            ax = plt.gca()
            violin_with_jitter(
                ax, groups=list(groups), labels=list(labels),
                title=f"Per-trajectory multimodality summary ({args.traj_stat}) at k={k_fixed}",
                ylabel=f"{args.traj_stat}(variance_drop_ratio over states in trajectory)",
            )
            plt.tight_layout()
            plt.savefig(out_violin, dpi=150)
            plt.close()
            print(f"Wrote: {out_violin}")
        else:
            print("[warn] No data for violin plot.")

        # ------------------------------------------------------------------
        # 2) Over time median + IQR
        # ------------------------------------------------------------------
        time_stats = []
        for (task, tb), g in df.groupby(["task_label", "t_bin"]):
            vals = g["variance_drop_ratio"].to_numpy()
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            t_norm = float(g["t_norm_bin"].iloc[0])
            time_stats.append({
                "task": task, "t_norm": t_norm, "n": int(vals.size),
                "median": float(np.median(vals)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
            })
        time_stats_df = pd.DataFrame(time_stats)
        if not time_stats_df.empty:
            out_time = os.path.join(plot_dir, "ratio_over_time_median_iqr.png")
            plot_over_time_median_iqr(out_time, time_stats_df, tasks=tasks, k_fixed=k_fixed)
            print(f"Wrote: {out_time}")

        # ------------------------------------------------------------------
        # 3) Over time variance
        # ------------------------------------------------------------------
        time_var = []
        for (task, tb), g in df.groupby(["task_label", "t_bin"]):
            vals = g["variance_drop_ratio"].to_numpy()
            vals = vals[~np.isnan(vals)]
            if vals.size < 2:
                continue
            t_norm = float(g["t_norm_bin"].iloc[0])
            time_var.append({"task": task, "t_norm": t_norm, "n": int(vals.size), "var": float(np.var(vals, ddof=1))})
        time_var_df = pd.DataFrame(time_var)
        if not time_var_df.empty:
            out_var = os.path.join(plot_dir, "ratio_variance_over_time.png")
            plot_over_time_variance(out_var, time_var_df, tasks=tasks, k_fixed=k_fixed)
            print(f"Wrote: {out_var}")

        # ------------------------------------------------------------------
        # Episode 3D plots: select episodes across low/mid/high bands
        # ------------------------------------------------------------------
        episodes_to_plot = []   # list of (task, band, episode_id)
        selection_meta = []     # for debugging

        for task in tasks:
            sub_traj = traj_df[traj_df["task"] == task].copy()
            if sub_traj.empty:
                continue
            col = f"{args.traj_stat}_ratio"
            sub_traj = sub_traj[sub_traj[col].notna()].copy()
            if sub_traj.empty:
                continue

            q_low = sub_traj[col].quantile(0.33)
            q_high = sub_traj[col].quantile(0.66)

            def band_fn(x):
                if x <= q_low:
                    return "low"
                if x >= q_high:
                    return "high"
                return "mid"

            sub_traj["band"] = sub_traj[col].apply(band_fn)

            for band in ["high", "mid", "low"]:
                band_df = sub_traj[sub_traj["band"] == band].copy()
                if band_df.empty:
                    continue

                if band == "high":
                    band_df.sort_values(col, ascending=False, inplace=True)
                elif band == "low":
                    band_df.sort_values(col, ascending=True, inplace=True)
                else:
                    m = band_df[col].median()
                    band_df["dist"] = (band_df[col] - m).abs()
                    band_df.sort_values("dist", ascending=True, inplace=True)

                selected = band_df.head(args.episodes_per_band)
                for r in selected.itertuples():
                    episodes_to_plot.append((task, band, r.episode_id))
                    selection_meta.append((task, band, r.episode_id, float(getattr(r, col))))

        # ------------------------------------------------------------------
        # NEW: If using lerobot_root and embedding_mode=pca, fit ONE global PCA
        # on points from all selected episodes (anchors + some integrated ghosts).
        # ------------------------------------------------------------------
        use_integrated_q = (args.lerobot_root is not None)
        pca_model = None
        q0_map = None  # (episode_id_str, t_int) -> qpos[:q_dim]

        if use_integrated_q:
            # gather keys needed
            needed = set()
            for (task, band, ep_id) in episodes_to_plot:
                ep_states = df[(df["task_label"] == task) & (df["episode_id"] == ep_id)].copy()
                for t in ep_states["t_in_episode"].astype(int).tolist():
                    needed.add((_parse_episode_id_to_index(ep_id), int(t)))

            state_col_candidates = [args.lerobot_state_col] if args.lerobot_state_col else None
            state_map = load_lerobot_states_for_keys(
                args.lerobot_root,
                needed_keys=needed,
                state_col_candidates=state_col_candidates,
                max_rows_per_parquet=args.lerobot_max_rows_per_parquet,
                frame_tolerance=max(0, int(args.lerobot_frame_tolerance or 0)),
            )
            print(f"[lerobot] loaded {len(state_map)}/{len(needed)} anchor states")

            # build q0_map keyed by (episode_id_str, t)
            q0_map = {}
            for (task, band, ep_id) in episodes_to_plot:
                ep_i = _parse_episode_id_to_index(ep_id)
                ep_states = df[(df["task_label"] == task) & (df["episode_id"] == ep_id)].copy()
                for t in ep_states["t_in_episode"].astype(int).tolist():
                    key = (ep_i, int(t))
                    if key in state_map and state_map[key].shape[0] >= q_dim:
                        q0_map[(str(ep_id), int(t))] = state_map[key][:q_dim].astype(float, copy=False)

            # fit PCA if requested
            if args.embedding_mode == "pca":
                pts = []
                kept = 0
                cap = int(args.pca_max_points)

                # anchors + a few integrated ghost points
                for (task, band, ep_id) in episodes_to_plot:
                    ep_states = df[(df["task_label"] == task) & (df["episode_id"] == ep_id)].copy()
                    ep_states = ep_states.sort_values("t_in_episode")

                    # anchors
                    for t in ep_states["t_in_episode"].astype(int).tolist():
                        q = q0_map.get((str(ep_id), int(t)), None)
                        if q is None:
                            continue
                        pts.append(q.reshape(1, -1))
                        kept += 1
                        if kept >= cap:
                            break
                    if kept >= cap:
                        break

                    # ghosts (subsample chunks)
                    for r in ep_states.itertuples():
                        if kept >= cap:
                            break
                        s = int(r.state_index)
                        t = int(r.t_in_episode)
                        q_anchor = q0_map.get((str(ep_id), t), None)
                        if q_anchor is None:
                            continue
                        arr = np.asarray(actions_arr[s], dtype=float)
                        if arr.ndim != 3:
                            continue
                        Kc = arr.shape[0]
                        take = min(2, Kc)  # tiny, just for PCA support
                        rng = np.random.RandomState(9000 + s)
                        idxs = rng.choice(Kc, size=take, replace=False) if Kc > take else np.arange(Kc)
                        for ck in idxs:
                            q_path = integrate_vel_chunk_to_q(arr[ck], q_anchor, dt=dt, q_dim=q_dim)  # (T+1,q_dim)
                            # subsample points along the path to reduce
                            step = max(1, q_path.shape[0] // 5)
                            pts.append(q_path[::step])
                            kept += q_path[::step].shape[0]
                            if kept >= cap:
                                break

                Xfit = np.vstack(pts) if pts else None
                if Xfit is None or Xfit.shape[0] < 10:
                    print("[warn] Not enough points to fit global PCA; falling back to first3.")
                    args.embedding_mode = "first3"
                else:
                    pca_model = PCA(n_components=3, random_state=0).fit(Xfit)
                    print(f"[viz] fit GLOBAL PCA on {Xfit.shape[0]} points (q_dim={q_dim})")

        # ------------------------------------------------------------------
        # Plot selected episodes
        # ------------------------------------------------------------------
        for task in tasks:
            task_episode_plot_dir = os.path.join(episode_plot_root_k, task)
            os.makedirs(task_episode_plot_dir, exist_ok=True)

            for band in ["high", "mid", "low"]:
                chosen = [ep_id for (t, b, ep_id) in episodes_to_plot if t == task and b == band]
                if not chosen:
                    continue

                print(f"\nTask '{task}' band '{band}': plotting {len(chosen)} episode(s)")
                for ep_id in chosen:
                    ep_states = df[(df["task_label"] == task) & (df["episode_id"] == ep_id)].copy()
                    if ep_states.empty:
                        continue

                    out_html = os.path.join(task_episode_plot_dir, f"episode_{str(ep_id)}_band_{band}.html")

                    plot_episode_trajectory_with_ghosts(
                        episode_id=ep_id,
                        band=band,
                        task_label=task,
                        ep_df=ep_states,
                        actions_arr=actions_arr,
                        global_r_min=global_r_min,
                        global_r_max=global_r_max,
                        out_html=out_html,
                        max_chunks_per_state=5,
                        embed_mode=args.embedding_mode,
                        pca_model=pca_model,
                        use_integrated_q=use_integrated_q,
                        q0_map=q0_map,
                        dt=dt,
                        q_dim=q_dim,
                    )

        print("\nDone.")


if __name__ == "__main__":
    main()
