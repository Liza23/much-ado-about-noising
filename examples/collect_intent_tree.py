"""Collect intent-tree rollouts to visualise when and how multimodality emerges.

Algorithm:
  1. Reset env; run forward with standard ODE intent sampling.
  2. At every policy step: probe intent diversity (N samples).
  3. Low variance  → execute ODE action, continue on current branch.
  4. High variance → freeze sim state, K-means cluster intents into K groups,
                     restore K times and run one act_steps chunk per centroid,
                     then recurse from each child independently.

The result is a tree of EEF trajectories where branching marks genuine
decision points in the policy's intent distribution.

Usage:
    python examples/collect_intent_tree.py \\
        --run "flow_intent:/ckpt.pt:task=lift_mh_state_flow_intent:+network.arch_variant=flow_intent" \\
        --n-rollouts 5 \\
        --diversity-threshold 0.005 \\
        --k-branches 3 \\
        --max-depth 2 \\
        --device cuda \\
        --out rollouts/lift_mh_state_intent_tree.pkl
"""

import argparse
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from sklearn.decomposition import PCA
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

from collect_diversity_rollouts import (
    load_config,
    parse_run_spec,
    setup_config_for_env,
    load_model,
)
from collect_intent_outcomes import (
    make_vec_env,
    make_dataset,
    get_sim_state,
    restore_env_obs,
    _get_inner_step_env,
    update_obs_buf,
    get_eef_from_envs,
    render_frame,
    undo_action,
    to_fi_obs,
    decode_fixed_intent,
    _unnormalize_intent_xyz,
)
from sklearn.cluster import KMeans


# ──────────────────────────────────────────────────────────────────────────────
# Tree data structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BranchNode:
    """One contiguous trajectory segment in the intent tree.

    A node runs from a branch point (or episode start) until it either:
      - branches again  → branch_point_t is set, children is populated
      - the episode ends / max_steps reached  → leaf node
    """
    branch_id: tuple          # () = root trunk, (0,) = first child, (0,2) = grandchild, …
    depth: int                # number of branching events above this node
    start_t: int              # episode timestep when this segment begins
    entry_intent: np.ndarray  # fixed intent used for the first chunk (None for trunk)
    entry_intent_xyz_world: np.ndarray  # unnormalized EEF pos from entry_intent (None for trunk)
    entry_score: float        # centroid separation score at the parent branch point

    eef_trajectory: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    frames: list = field(default_factory=list)
    n_steps: int = 0
    done: bool = False
    success: bool = False

    # Populated when this node itself branches
    children: list = field(default_factory=list)
    branch_point_t: int = None       # timestep where THIS node branched
    branch_point_eef: np.ndarray = None
    branch_score: float = None       # centroid separation score that triggered the branch
    branch_probe_intents: np.ndarray = None
    branch_probe_intents_xyz_world: np.ndarray = None


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def _probe_multimodality(obs_buf, agent, config, dataset, device, n_probe, n_clusters, num_steps=9):
    """Sample n_probe intents, cluster into n_clusters, return a multimodality score and centroids.

    Score = mean pairwise Euclidean distance between cluster centroids in normalized intent
    space.  High score → genuinely multimodal; low score → unimodal spread that does not
    warrant branching.  Centroids are snapped to the nearest actual sampled intent so that
    representatives are always real, valid poses (no quaternion component averaging).

    Returns: (score: float, intents: ndarray (n_probe, D), centroids: ndarray (n_clusters, D))
    """
    from itertools import combinations

    fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
    intents = []
    with torch.no_grad():
        for _ in range(n_probe):
            _, iv = agent.sample(obs=fi_obs, use_ema=True,
                                  num_steps=num_steps, return_intent=True)
            intents.append(iv[0].cpu().numpy())
    intents = np.stack(intents)

    k = min(n_clusters, len(intents))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(intents)
    raw_centroids = km.cluster_centers_  # (k, D)

    # Snap each raw centroid to its nearest actual sampled intent.
    # This guarantees the representative is a real, sampled pose (valid unit quaternion, no
    # component averaging artifacts).
    centroids = np.empty_like(raw_centroids)
    for ki in range(k):
        dists = np.linalg.norm(intents - raw_centroids[ki], axis=1)
        centroids[ki] = intents[np.argmin(dists)]

    # Score = mean pairwise centroid distance.  Near-zero → unimodal; large → genuinely
    # multimodal intents that warrant branching.
    pairs = list(combinations(range(k), 2))
    score = float(np.mean([np.linalg.norm(centroids[i] - centroids[j])
                            for i, j in pairs])) if pairs else 0.0

    return score, intents, centroids


def _run_chunk_fixed_intent(envs, obs_buf, fixed_intent, agent, config, dataset,
                             device, capture_render, num_steps=9):
    """Execute one act_steps chunk with a fixed intent centroid.

    Returns: (obs_buf, eef_positions, frames, done, info, total_reward)
    """
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    action_un = decode_fixed_intent(obs_buf, fixed_intent, agent, config, dataset, device)

    eef_positions, frames = [], []
    done = False
    info = {}
    total_reward = 0.0
    for a_i in range(act_steps):
        act = undo_action(action_un[0, s + a_i], config, dataset)
        _, reward, done, info = inner.step(act)
        done = bool(done)
        eef_positions.append(get_eef_from_envs(envs, config))
        if capture_render:
            f = render_frame(envs, config)
            if f is not None:
                frames.append(f)
        total_reward += float(reward)
        if done:
            break

    obs_buf = update_obs_buf(obs_buf, envs, config)
    return obs_buf, eef_positions, frames, done, info, total_reward


def _run_chunk_ode(envs, obs_buf, agent, config, dataset, device,
                   capture_render, num_steps=9):
    """Execute one act_steps chunk with ODE-sampled intent (normal policy inference).

    Returns: (obs_buf, eef_positions, frames, done, info, total_reward)
    """
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)

    fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
    with torch.no_grad():
        act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
    act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())

    eef_positions, frames = [], []
    done = False
    info = {}
    total_reward = 0.0
    for a_i in range(act_steps):
        act = undo_action(act_un[0, s + a_i], config, dataset)
        _, reward, done, info = inner.step(act)
        done = bool(done)
        eef_positions.append(get_eef_from_envs(envs, config))
        if capture_render:
            f = render_frame(envs, config)
            if f is not None:
                frames.append(f)
        total_reward += float(reward)
        if done:
            break

    obs_buf = update_obs_buf(obs_buf, envs, config)
    return obs_buf, eef_positions, frames, done, info, total_reward


# ──────────────────────────────────────────────────────────────────────────────
# Core tree growth
# ──────────────────────────────────────────────────────────────────────────────

def _copy_obs_buf(obs_buf, obs_type):
    if obs_type == "state":
        return obs_buf.copy()
    return {k: v.copy() for k, v in obs_buf.items()}


def grow_branch(
    sim_state,
    obs_buf,
    start_t,
    agent,
    config,
    dataset,
    envs,
    device,
    args,
    depth,
    branch_id,
    entry_intent=None,
    entry_score=0.0,
    num_steps=9,
):
    """Grow one branch of the intent tree from (sim_state, obs_buf) at time start_t.

    - Restores env to sim_state.
    - If entry_intent is set: executes one fixed-intent chunk first (this branch
      diverges from the parent at the branch point via its specific centroid).
    - Then loops: probe multimodality → ODE step (low score) or branch (high score).
    - Returns a populated BranchNode.
    """
    node = BranchNode(
        branch_id=branch_id,
        depth=depth,
        start_t=start_t,
        entry_intent=entry_intent,
        entry_intent_xyz_world=_unnormalize_intent_xyz(dataset, entry_intent)
                               if entry_intent is not None else None,
        entry_score=entry_score,
    )

    # Restore env to the frozen state for this branch.
    # Call for simulator-state side-effect only; do NOT overwrite obs_buf.
    # The passed obs_buf already contains the correct obs_steps stacked history at the
    # branch point (e.g. [obs[t-1], obs[t]] for obs_steps=2).  Replacing it with
    # restore_env_obs's return value would bootstrap from [obs[t], obs[t]], losing history.
    restore_env_obs(envs, sim_state, config)

    eef_traj = [get_eef_from_envs(envs, config)]
    all_frames = []
    total_reward = 0.0
    info = {}
    done = False
    t = start_t

    # ── Execute fixed-intent chunk (only for non-root branches) ───────────────
    if entry_intent is not None:
        obs_buf, eef_pos_list, frames, done, info, r = _run_chunk_fixed_intent(
            envs, obs_buf, entry_intent, agent, config, dataset, device,
            capture_render=args.render, num_steps=num_steps,
        )
        eef_traj.extend(eef_pos_list)
        all_frames.extend(frames)
        total_reward += r
        t += len(eef_pos_list)
        if done:
            node.eef_trajectory = np.array(eef_traj)
            node.frames = all_frames
            node.n_steps = t - start_t
            node.done = True
            node.success = bool(info.get("success", False)) or total_reward > 0
            return node

    # ── Main loop: probe then ODE-step or branch ───────────────────────────────
    max_steps = config.task.max_episode_steps

    while not done and t < max_steps:
        score, intents, centroids = _probe_multimodality(
            obs_buf, agent, config, dataset, device,
            n_probe=args.n_probe, n_clusters=args.k_branches, num_steps=num_steps,
        )

        if score >= args.diversity_threshold and depth < args.max_depth:
            # ── Branch: centroids already computed & snapped in _probe_variance ─
            n_clusters = len(centroids)

            frozen_sim = get_sim_state(envs, config)
            frozen_obs = _copy_obs_buf(obs_buf, config.task.obs_type)

            node.branch_point_t = t
            node.branch_point_eef = get_eef_from_envs(envs, config).copy()
            node.branch_score = score
            node.branch_probe_intents = intents.copy()
            node.branch_probe_intents_xyz_world = np.stack([
                _unnormalize_intent_xyz(dataset, intent) for intent in intents
            ])

            print(f"  {'  ' * depth}Branch  branch_id={branch_id}  t={t}  "
                  f"score={score:.5f}  → {n_clusters} children")

            for ki in range(n_clusters):
                child = grow_branch(
                    sim_state=frozen_sim,
                    obs_buf=frozen_obs,
                    start_t=t,
                    agent=agent,
                    config=config,
                    dataset=dataset,
                    envs=envs,
                    device=device,
                    args=args,
                    depth=depth + 1,
                    branch_id=branch_id + (ki,),
                    entry_intent=centroids[ki],
                    entry_score=score,
                    num_steps=num_steps,
                )
                node.children.append(child)
            break  # this node's segment ends at the branch point

        else:
            # ── Continue: ODE step ─────────────────────────────────────────────
            obs_buf, eef_pos_list, frames, done, info, r = _run_chunk_ode(
                envs, obs_buf, agent, config, dataset, device,
                capture_render=args.render, num_steps=num_steps,
            )
            eef_traj.extend(eef_pos_list)
            all_frames.extend(frames)
            total_reward += r
            t += len(eef_pos_list)

    node.eef_trajectory = np.array(eef_traj)
    node.frames = all_frames
    node.n_steps = t - start_t
    node.done = done
    node.success = bool(info.get("success", False)) or total_reward > 0
    return node


def collect_tree_rollouts(config, agent, dataset, envs, args, device):
    """Run n_rollouts episodes, each producing an intent tree."""
    trees = []
    num_steps = 9
    n_done = 0

    while n_done < args.n_rollouts:
        print(f"\nEpisode {n_done + 1}/{args.n_rollouts}")
        obs, _ = envs.reset()
        sim_state = get_sim_state(envs, config)

        root = grow_branch(
            sim_state=sim_state,
            obs_buf=obs,
            start_t=0,
            agent=agent,
            config=config,
            dataset=dataset,
            envs=envs,
            device=device,
            args=args,
            depth=0,
            branch_id=(),
            entry_intent=None,
            entry_score=0.0,
            num_steps=num_steps,
        )

        trees.append({"episode": n_done, "root": root})
        n_done += config.task.num_envs
        _print_tree_stats(root, indent=0)

    return trees


def _print_tree_stats(node, indent=0):
    bp = (f" → branches at t={node.branch_point_t} score={node.branch_score:.5f}"
          if node.branch_point_t is not None else " (leaf)")
    print(f"{'  ' * indent}[{node.branch_id or 'root'}] "
          f"depth={node.depth}  steps={node.n_steps}  success={node.success}{bp}")
    for child in node.children:
        _print_tree_stats(child, indent + 1)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

_TRUNK_COLOR = np.array([0.25, 0.25, 0.25, 1.0])
_BRANCH_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))
_DEPTH_LINESTYLES = ["-", "--", ":", "-."]


def _branch_color(branch_id):
    """Top-level branch color; trunk is dark gray."""
    if not branch_id:
        return _TRUNK_COLOR
    return _BRANCH_PALETTE[branch_id[0] % 10]


def _collect_all_xyz(node):
    """Recursively collect all EEF points for axis-limit computation."""
    pts = []
    if node.eef_trajectory.shape[0] > 0:
        pts.append(node.eef_trajectory)
    if node.branch_point_eef is not None:
        pts.append(node.branch_point_eef.reshape(1, 3))
    if node.branch_probe_intents_xyz_world is not None:
        pts.append(node.branch_probe_intents_xyz_world)
    for child in node.children:
        if child.entry_intent_xyz_world is not None:
            pts.append(child.entry_intent_xyz_world.reshape(1, 3))
        pts.extend(_collect_all_xyz(child))
    return pts


def _draw_node(ax, node):
    """Recursively draw a node's trajectory segment and its children."""
    traj = node.eef_trajectory
    c = _branch_color(node.branch_id)
    ls = _DEPTH_LINESTYLES[min(node.depth, len(_DEPTH_LINESTYLES) - 1)]
    lw = max(2.2 - 0.5 * node.depth, 0.8)
    alpha = max(0.95 - 0.1 * node.depth, 0.5)

    if traj.shape[0] > 0:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=c, lw=lw, ls=ls, alpha=alpha)

        # Episode start marker
        if not node.branch_id:
            ax.scatter(*traj[0], c="black", s=80, marker="o",
                       depthshade=False, zorder=10)

        # Leaf end marker
        if not node.children:
            ax.scatter(*traj[-1], color=c, s=60, marker="*",
                       depthshade=False, zorder=5)

    # Branch point diamond
    if node.branch_point_eef is not None:
        ax.scatter(*node.branch_point_eef, color=c, s=90, marker="D",
                   depthshade=False, edgecolors="black", linewidths=0.6, zorder=8)

        # Intent markers for each child at the branch point
        for child in node.children:
            if child.entry_intent_xyz_world is not None:
                cc = _branch_color(child.branch_id)
                ax.scatter(*child.entry_intent_xyz_world, color=cc, s=65,
                           marker="^", depthshade=False, edgecolors="black",
                           linewidths=0.4, zorder=7, alpha=0.85)

    for child in node.children:
        _draw_node(ax, child)


def _draw_node_raw_intents(ax, node):
    """Recursively draw only branch points and raw intent centroid positions."""
    c = _branch_color(node.branch_id)

    if node.branch_point_eef is not None:
        ax.scatter(*node.branch_point_eef, color=c, s=90, marker="D",
                   depthshade=False, edgecolors="black", linewidths=0.6, zorder=8)

        for child in node.children:
            if child.entry_intent_xyz_world is not None:
                cc = _branch_color(child.branch_id)
                intent_xyz = child.entry_intent_xyz_world
                ax.plot(
                    [node.branch_point_eef[0], intent_xyz[0]],
                    [node.branch_point_eef[1], intent_xyz[1]],
                    [node.branch_point_eef[2], intent_xyz[2]],
                    color=cc, lw=1.2, ls=":", alpha=0.9,
                )
                ax.scatter(*intent_xyz, color=cc, s=65, marker="^",
                           depthshade=False, edgecolors="black",
                           linewidths=0.4, zorder=7, alpha=0.9)

    for child in node.children:
        _draw_node_raw_intents(ax, child)


def _draw_node_probe_intents(ax, node):
    """Recursively draw full probe-intent samples at each branch point."""
    c = _branch_color(node.branch_id)

    if node.branch_point_eef is not None:
        ax.scatter(*node.branch_point_eef, color=c, s=90, marker="D",
                   depthshade=False, edgecolors="black", linewidths=0.6, zorder=8)

        if node.branch_probe_intents_xyz_world is not None:
            pts = node.branch_probe_intents_xyz_world
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                color=c, s=20, marker=".", alpha=0.4,
                depthshade=False, zorder=5,
            )
            for pt in pts:
                ax.plot(
                    [node.branch_point_eef[0], pt[0]],
                    [node.branch_point_eef[1], pt[1]],
                    [node.branch_point_eef[2], pt[2]],
                    color=c, lw=0.7, ls=":", alpha=0.15,
                )

        for child in node.children:
            if child.entry_intent_xyz_world is not None:
                cc = _branch_color(child.branch_id)
                intent_xyz = child.entry_intent_xyz_world
                ax.scatter(*intent_xyz, color=cc, s=65, marker="^",
                           depthshade=False, edgecolors="black",
                           linewidths=0.4, zorder=7, alpha=0.95)

    for child in node.children:
        _draw_node_probe_intents(ax, child)


def _collect_all_raw_intent_vectors(node):
    """Recursively collect all raw intent vectors stored in a tree."""
    vecs = []
    if node.entry_intent is not None:
        vecs.append(np.asarray(node.entry_intent, dtype=np.float64).reshape(1, -1))
    if node.branch_probe_intents is not None:
        vecs.append(np.asarray(node.branch_probe_intents, dtype=np.float64))
    for child in node.children:
        vecs.extend(_collect_all_raw_intent_vectors(child))
    return vecs


def _draw_node_raw_intents_pca(ax, node, projector):
    """Recursively draw raw full intent vectors in PCA space."""
    c = _branch_color(node.branch_id)

    if node.branch_probe_intents is not None:
        pts = projector(np.asarray(node.branch_probe_intents, dtype=np.float64))
        if pts.shape[1] < 3:
            pts = np.pad(pts, ((0, 0), (0, 3 - pts.shape[1])))
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            color=c, s=22, marker=".", alpha=0.35,
            depthshade=False, zorder=5,
        )

    for child in node.children:
        if child.entry_intent is not None:
            cc = _branch_color(child.branch_id)
            pt = projector(np.asarray(child.entry_intent, dtype=np.float64).reshape(1, -1))
            if pt.shape[1] < 3:
                pt = np.pad(pt, ((0, 0), (0, 3 - pt.shape[1])))
            ax.scatter(
                pt[:, 0], pt[:, 1], pt[:, 2],
                color=cc, s=65, marker="^",
                depthshade=False, edgecolors="black",
                linewidths=0.4, zorder=7, alpha=0.95,
            )

    for child in node.children:
        _draw_node_raw_intents_pca(ax, child, projector)


def fig_intent_tree(trees, out_dir, task_name, args):
    """3D plot — one panel per episode showing the full intent tree."""
    n = len(trees)
    if n == 0:
        return

    fig = plt.figure(figsize=(6.5 * n, 6.5))

    for ep_i, tree in enumerate(trees):
        root = tree["root"]
        ax = fig.add_subplot(1, n, ep_i + 1, projection="3d")

        # Axis limits from all EEF points in the tree
        all_pts = _collect_all_xyz(root)
        if not all_pts:
            continue
        xyz = np.vstack(all_pts)
        center = 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))
        max_range = float(np.max(xyz.max(axis=0) - xyz.min(axis=0)))
        half = max(0.6 * max_range, 1e-3)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

        _draw_node(ax, root)

        ax.set_xlabel("X (m)", fontsize=7, labelpad=2)
        ax.set_ylabel("Y (m)", fontsize=7, labelpad=2)
        ax.set_zlabel("Z (m)", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-55)

        # Legend
        legend_elems = [
            Line2D([0], [0], color=_TRUNK_COLOR, lw=2, label="trunk (ODE)"),
            Line2D([0], [0], color="black", lw=0, marker="o", ms=7,
                   label="episode start"),
            Line2D([0], [0], color="black", lw=0, marker="D", ms=6,
                   markeredgecolor="black", label="branch point"),
            Line2D([0], [0], color="black", lw=0, marker="^", ms=6,
                   markeredgecolor="black", label="intent EEF pos (world)"),
            Line2D([0], [0], color="black", lw=0, marker="*", ms=8,
                   label="leaf end"),
        ]
        for ki in range(args.k_branches):
            legend_elems.append(
                Line2D([0], [0], color=_BRANCH_PALETTE[ki % 10], lw=2,
                       label=f"branch {ki + 1}")
            )
        ax.legend(handles=legend_elems, fontsize=6.5, loc="upper left")
        ax.set_title(f"Episode {ep_i}", fontsize=9)

    fig.suptitle(
        f"[{task_name}]  Intent tree  (branch when centroid separation ≥ {args.diversity_threshold})\n"
        f"K={args.k_branches}  max_depth={args.max_depth}  "
        f"◆=branch point  ^=intent EEF pos  ★=leaf end",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "intent_tree.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_raw_intent_tree(trees, out_dir, task_name, args):
    """3D plot showing only branch points and intent XYZ positions."""
    n = len(trees)
    if n == 0:
        return

    fig = plt.figure(figsize=(6.5 * n, 6.5))

    for ep_i, tree in enumerate(trees):
        root = tree["root"]
        ax = fig.add_subplot(1, n, ep_i + 1, projection="3d")

        all_pts = _collect_all_xyz(root)
        if not all_pts:
            continue
        xyz = np.vstack(all_pts)
        center = 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))
        max_range = float(np.max(xyz.max(axis=0) - xyz.min(axis=0)))
        half = max(0.6 * max_range, 1e-3)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

        _draw_node_raw_intents(ax, root)

        ax.set_xlabel("X (m)", fontsize=7, labelpad=2)
        ax.set_ylabel("Y (m)", fontsize=7, labelpad=2)
        ax.set_zlabel("Z (m)", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-55)

        legend_elems = [
            Line2D([0], [0], color="black", lw=0, marker="D", ms=6,
                   markeredgecolor="black", label="branch point"),
            Line2D([0], [0], color="black", lw=1.2, ls=":",
                   label="intent connector"),
            Line2D([0], [0], color="black", lw=0, marker="^", ms=6,
                   markeredgecolor="black", label="intent EEF pos (world)"),
        ]
        for ki in range(args.k_branches):
            legend_elems.append(
                Line2D([0], [0], color=_BRANCH_PALETTE[ki % 10], lw=2,
                       label=f"branch {ki + 1}")
            )
        ax.legend(handles=legend_elems, fontsize=6.5, loc="upper left")
        ax.set_title(f"Episode {ep_i}", fontsize=9)

    fig.suptitle(
        f"[{task_name}]  Raw intents (XYZ slice only)\n"
        f"K={args.k_branches}  max_depth={args.max_depth}  "
        f"◆=branch point  ^=intent EEF pos",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "intent_tree_raw_intents.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_raw_intent_tree_pca(trees, out_dir, task_name, args):
    """3D PCA plot of the full raw intent vectors at all branch points."""
    n = len(trees)
    if n == 0:
        return

    all_raw = []
    for tree in trees:
        all_raw.extend(_collect_all_raw_intent_vectors(tree["root"]))
    if not all_raw:
        return

    X = np.vstack(all_raw)
    n_comp = min(3, X.shape[0], X.shape[1])
    if n_comp <= 0:
        return
    pca = PCA(n_components=n_comp, random_state=0).fit(X)

    fig = plt.figure(figsize=(6.5 * n, 6.5))

    for ep_i, tree in enumerate(trees):
        root = tree["root"]
        ax = fig.add_subplot(1, n, ep_i + 1, projection="3d")

        episode_raw = _collect_all_raw_intent_vectors(root)
        if not episode_raw:
            continue
        xyz = pca.transform(np.vstack(episode_raw))
        if xyz.shape[1] < 3:
            xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])))
        center = 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))
        max_range = float(np.max(xyz.max(axis=0) - xyz.min(axis=0)))
        half = max(0.6 * max_range, 1e-3)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

        _draw_node_raw_intents_pca(ax, root, pca.transform)

        ax.set_xlabel("PC1", fontsize=7, labelpad=2)
        ax.set_ylabel("PC2", fontsize=7, labelpad=2)
        ax.set_zlabel("PC3", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-55)

        legend_elems = [
            Line2D([0], [0], color="black", lw=0, marker=".", ms=8,
                   label="probe intent sample"),
            Line2D([0], [0], color="black", lw=0, marker="^", ms=6,
                   markeredgecolor="black", label="chosen centroid"),
        ]
        for ki in range(args.k_branches):
            legend_elems.append(
                Line2D([0], [0], color=_BRANCH_PALETTE[ki % 10], lw=2,
                       label=f"branch {ki + 1}")
            )
        ax.legend(handles=legend_elems, fontsize=6.5, loc="upper left")
        ax.set_title(f"Episode {ep_i}", fontsize=9)

    fig.suptitle(
        f"[{task_name}]  Raw intents (full vector PCA)\n"
        f"K={args.k_branches}  max_depth={args.max_depth}  "
        f".=probe intent  ^=chosen centroid",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "intent_tree_raw_intents_pca.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_probe_intent_tree(trees, out_dir, task_name, args):
    """3D plot showing all sampled probe intents at each branch point."""
    n = len(trees)
    if n == 0:
        return

    fig = plt.figure(figsize=(6.5 * n, 6.5))

    for ep_i, tree in enumerate(trees):
        root = tree["root"]
        ax = fig.add_subplot(1, n, ep_i + 1, projection="3d")

        all_pts = _collect_all_xyz(root)
        if not all_pts:
            continue
        xyz = np.vstack(all_pts)
        center = 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))
        max_range = float(np.max(xyz.max(axis=0) - xyz.min(axis=0)))
        half = max(0.6 * max_range, 1e-3)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

        _draw_node_probe_intents(ax, root)

        ax.set_xlabel("X (m)", fontsize=7, labelpad=2)
        ax.set_ylabel("Y (m)", fontsize=7, labelpad=2)
        ax.set_zlabel("Z (m)", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-55)

        legend_elems = [
            Line2D([0], [0], color="black", lw=0, marker="D", ms=6,
                   markeredgecolor="black", label="branch point"),
            Line2D([0], [0], color="black", lw=1.0, ls=":",
                   label="probe-intent connector"),
            Line2D([0], [0], color="black", lw=0, marker=".", ms=8,
                   label="probe intent sample"),
            Line2D([0], [0], color="black", lw=0, marker="^", ms=6,
                   markeredgecolor="black", label="chosen centroid"),
        ]
        for ki in range(args.k_branches):
            legend_elems.append(
                Line2D([0], [0], color=_BRANCH_PALETTE[ki % 10], lw=2,
                       label=f"branch {ki + 1}")
            )
        ax.legend(handles=legend_elems, fontsize=6.5, loc="upper left")
        ax.set_title(f"Episode {ep_i}", fontsize=9)

    fig.suptitle(
        f"[{task_name}]  All probe intents at branch points\n"
        f"n_probe={args.n_probe}  K={args.k_branches}  max_depth={args.max_depth}  "
        f"◆=branch point  .=probe intent  ^=chosen centroid",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "intent_tree_probe_intents.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grow intent trees: branch when the flow model's intent is diverse."
    )
    parser.add_argument("--run", required=True,
                        help="'label:ckpt_path:task=X[:overrides]'")
    parser.add_argument("--n-rollouts", type=int, default=5,
                        help="Number of root episodes to grow trees from")
    parser.add_argument("--diversity-threshold", type=float, default=0.005,
                        help="Mean pairwise centroid separation threshold above which to branch. "
                             "Low score → unimodal; high score → multimodal. "
                             "Tune: too low → branches everywhere, too high → never branches.")
    parser.add_argument("--k-branches", type=int, default=3,
                        help="Number of K-means clusters at each branch point")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Maximum branching depth (K^max_depth leaves at most)")
    parser.add_argument("--n-probe", type=int, default=20,
                        help="Intent samples per policy step for multimodality probing")
    parser.add_argument("--render", action="store_true",
                        help="Capture rendered RGB frames along trajectories")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="Output .pkl path")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for figures (default: <out_stem>_figs/)")
    parser.add_argument("--plot-mode", choices=["tree", "raw-intents", "probe-intents", "both"],
                        default="both",
                        help="Which figure variant(s) to save")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label, ckpt_path, overrides = parse_run_spec(args.run)
    print(f"[collect_intent_tree] variant={label}  ckpt={ckpt_path}")
    print(f"  threshold={args.diversity_threshold}  K={args.k_branches}  "
          f"max_depth={args.max_depth}  n_probe={args.n_probe}")

    config = load_config(overrides)
    config.optimization.device = args.device
    config.task.num_envs = 1
    config.task.save_video = bool(args.render)

    envs = make_vec_env(config, seed=args.seed)
    setup_config_for_env(config, envs)
    dataset = make_dataset(config)

    agent, _ = load_model(ckpt_path, config, dataset, args.device)
    agent.eval()

    print(f"Task: {config.task.env_name}  obs_type: {config.task.obs_type}  "
          f"intent_dim: {getattr(config.task, 'intent_dim', '?')}")

    trees = collect_tree_rollouts(config, agent, dataset, envs, args, args.device)

    # Save pkl
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "task": config.task.env_name,
            "obs_type": config.task.obs_type,
            "intent_dim": getattr(config.task, "intent_dim", 7),
            "diversity_threshold": args.diversity_threshold,
            "k_branches": args.k_branches,
            "max_depth": args.max_depth,
            "trees": trees,
        }, f)
    print(f"\nSaved pkl: {out_path}")

    # Figures
    out_dir = (Path(args.out_dir) if args.out_dir
               else out_path.parent / (out_path.stem + "_figs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_mode in ("tree", "both"):
        fig_intent_tree(trees, out_dir, f"{config.task.env_name} [{label}]", args)
    if args.plot_mode in ("raw-intents", "both"):
        fig_raw_intent_tree(trees, out_dir, f"{config.task.env_name} [{label}]", args)
        fig_raw_intent_tree_pca(trees, out_dir, f"{config.task.env_name} [{label}]", args)
    if args.plot_mode in ("probe-intents", "both"):
        fig_probe_intent_tree(trees, out_dir, f"{config.task.env_name} [{label}]", args)
    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
