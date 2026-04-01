#!/usr/bin/env python3
"""Shared helpers for DROID clustering and Gaussian-threshold postprocessing."""

import bisect
import glob
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist


def npz_actions_len(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "actions" not in data:
        return 0
    return int(len(data["actions"]))


class LazyActionsFromNpzRanges:
    """
    Memory-light indexable actions dataset over many checkpoint NPZ files.
    Supports len() and __getitem__(int).
    """

    def __init__(self, npz_paths):
        self._npz_paths = list(npz_paths)
        self._ends = []
        total = 0
        for p in self._npz_paths:
            n = npz_actions_len(p)
            total += int(n)
            self._ends.append(total)
        self._len = total
        self._cache_path = None
        self._cache_actions = None

    def __len__(self):
        return self._len

    def _load_actions_for_path(self, p):
        if self._cache_path == p and self._cache_actions is not None:
            return self._cache_actions
        data = np.load(p, allow_pickle=True)
        arr = data["actions"] if "actions" in data else np.array([], dtype=object)
        self._cache_path = p
        self._cache_actions = arr
        return arr

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)
        file_i = bisect.bisect_right(self._ends, idx)
        start = 0 if file_i == 0 else self._ends[file_i - 1]
        local_i = idx - start
        p = self._npz_paths[file_i]
        arr = self._load_actions_for_path(p)
        return arr[local_i]


def load_actions_with_fallback(actions_npz_path):
    """
    Primary: load merged actions NPZ.
    Fallback: if missing, lazily index checkpoint NPZs from:
      - episodes/*/actions.npz
      - shards/*_actions.npz
    """
    if os.path.exists(actions_npz_path):
        data = np.load(actions_npz_path, allow_pickle=True)
        return data["actions"]

    root = os.path.dirname(os.path.abspath(actions_npz_path))
    ep_paths = sorted(glob.glob(os.path.join(root, "episodes", "*", "actions.npz")))
    shard_paths = sorted(glob.glob(os.path.join(root, "shards", "*_actions.npz")))

    src_paths = ep_paths if ep_paths else shard_paths
    src_kind = "episodes" if ep_paths else "shards"
    if not src_paths:
        raise FileNotFoundError(
            f"actions_npz not found at '{actions_npz_path}', and no fallback checkpoints under "
            f"'{root}/episodes/*/actions.npz' or '{root}/shards/*_actions.npz'."
        )

    valid_paths = []
    total_states = 0
    for p in src_paths:
        try:
            n = npz_actions_len(p)
        except Exception as e:
            sys.stderr.write(f"[warn] failed reading checkpoint actions npz {p}: {e}\n")
            continue
        if n <= 0:
            continue
        valid_paths.append(p)
        total_states += int(n)

    if not valid_paths or total_states <= 0:
        raise FileNotFoundError(
            f"Found {len(src_paths)} {src_kind} checkpoint npz files but none had readable 'actions'."
        )

    sys.stderr.write(
        f"[warn] '{actions_npz_path}' not found; using lazy fallback from "
        f"{len(valid_paths)} {src_kind} checkpoint files ({total_states} states).\n"
    )
    return LazyActionsFromNpzRanges(valid_paths)


def integrate_joint_velocity_chunk(chunk, dt, vel_dims=7, include_gripper=True, anchor_start=True,
                                   abs_action=False):
    """
    chunk: (T, A) where first vel_dims are joint velocities (or absolute positions if abs_action=True).
    Returns trajectory:
      abs_action=False: cumsum of velocities * dt (original behavior)
      abs_action=True:  action positions used directly, anchored to first timestep
    """
    arr = np.asarray(chunk, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"chunk must be 2D (T,A), got {arr.shape}")
    t_len, a_dim = arr.shape
    if a_dim < vel_dims:
        raise ValueError(f"chunk has {a_dim} dims but vel_dims={vel_dims}")

    v = arr[:, :vel_dims]
    if abs_action:
        dq = v - v[:1]  # anchor to first timestep; no cumsum needed
    else:
        dq = np.cumsum(v * float(dt), axis=0)
        if anchor_start and t_len > 1:
            dq = dq - dq[:1]

    if include_gripper and a_dim >= vel_dims + 1:
        g = arr[:, vel_dims:vel_dims + 1]
        return np.hstack([dq, g])
    return dq


def prepare_integrated_trajectories_from_actions(arr, dt, vel_dims=7, include_gripper=True,
                                                  abs_action=False):
    """
    Convert raw actions into trajectories.
    Supports:
      - arr: (K, T, A) -> list length K
      - arr: (T, A)    -> list length 1
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 3:
        n_chunks = arr.shape[0]
        return [
            integrate_joint_velocity_chunk(
                arr[k], dt=dt, vel_dims=vel_dims, include_gripper=include_gripper,
                anchor_start=True, abs_action=abs_action,
            )
            for k in range(n_chunks)
        ]
    if arr.ndim == 2:
        return [
            integrate_joint_velocity_chunk(
                arr, dt=dt, vel_dims=vel_dims, include_gripper=include_gripper,
                anchor_start=True, abs_action=abs_action,
            )
        ]
    raise ValueError(f"Unsupported action shape: {arr.shape}")


def prepare_trajectories_and_features_from_actions(arr, dt, vel_dims=7, include_gripper=True,
                                                    abs_action=False):
    """
    Converts raw actions (K,T,A) into:
      trajectories: list of (T_i, D) arrays
      x_feat:       (N_chunks, flat_dim) for silhouette/CH
      plot_matrix:  per-point matrix for plotting (sum(T_i), D)
      chunk_ids, time_ids: per-point indexing (len = sum(T_i))
    """
    if arr.ndim == 3:
        n_chunks, t_len, _ = arr.shape
        trajs = []
        for k in range(n_chunks):
            trajs.append(
                integrate_joint_velocity_chunk(
                    arr[k], dt=dt, vel_dims=vel_dims, include_gripper=include_gripper,
                    anchor_start=True, abs_action=abs_action,
                )
            )
        x_feat = np.stack([tr.reshape(-1) for tr in trajs], axis=0).astype(float)
        plot_matrix = np.vstack(trajs).astype(float)
        chunk_ids = np.repeat(np.arange(n_chunks), t_len)
        time_ids = np.tile(np.arange(t_len), n_chunks)
        return trajs, x_feat, plot_matrix, chunk_ids, time_ids

    raise ValueError(f"Unsupported actions shape: {arr.shape}")


def as_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(float, copy=False)
    if x.ndim >= 3:
        x = x.reshape(x.shape[0], -1)
    return x.astype(float, copy=False)


def symmetrized_l2_minkowski_traj(x_traj, y_traj):
    """
    x_traj: (T_x, D)  y_traj: (T_y, D)
    d1^2 = sum_i min_j ||x_i - y_j||^2
    d2^2 = sum_j min_i ||y_j - x_i||^2
    d = sqrt((d1^2 + d2^2) / 2)
    """
    x_traj = as_2d(x_traj)
    y_traj = as_2d(y_traj)
    if x_traj.shape[0] == 0 and y_traj.shape[0] == 0:
        return 0.0
    if x_traj.shape[0] == 0 or y_traj.shape[0] == 0:
        return np.inf

    dists_sq = cdist(x_traj, y_traj, metric="sqeuclidean")
    d1_sq = np.sum(np.min(dists_sq, axis=1))
    d2_sq = np.sum(np.min(dists_sq, axis=0))
    return float(np.sqrt(0.5 * (d1_sq + d2_sq)))


def compute_minkowski_distance_matrix(trajectories):
    """
    trajectories: list of arrays, each (T_i, D).
    Returns D[i,j] = symmetrized_l2_minkowski_traj(traj_i, traj_j).
    """
    n_items = len(trajectories)
    dist_mat = np.zeros((n_items, n_items), dtype=float)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            d = symmetrized_l2_minkowski_traj(trajectories[i], trajectories[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat


def _minkowski_variance_from_D(D):
    """
    Simplified variance proxy:
      Var := 0.5 * E[d^2]
    """
    D = np.asarray(D, float)
    if D.size == 0:
        return 0.0
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got {D.shape}")

    n = D.shape[0]
    if n <= 1:
        return 0.0

    offdiag = ~np.eye(n, dtype=bool)
    term1 = float(np.mean((D[offdiag]) ** 2))
    return 0.5 * term1


def minkowski_variance_from_D(D):
    return _minkowski_variance_from_D(D)


def total_variance_minkowski(D):
    return _minkowski_variance_from_D(D)


def weighted_incluster_variance_minkowski(D, labels, debug=False):
    labels = np.asarray(labels)
    n = D.shape[0]
    if n == 0:
        return 0.0

    wvar = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        Dc = D[np.ix_(idx, idx)]
        var_c = _minkowski_variance_from_D(Dc)
        wvar += (idx.size / n) * var_c

    if debug:
        return float(wvar)
    return float(wvar)
