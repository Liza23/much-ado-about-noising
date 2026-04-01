"""Conflict Audit for Push-T dataset: Mining Multimodal Hubs.

Finds "decision points" where the block state is nearly identical but
expert actions diverge significantly — evidence of multimodality.

Usage:
    python examples/conflict_audit_pusht.py
    python examples/conflict_audit_pusht.py --eps 0.05 --delta 0.15 --obs block
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import zarr
from sklearn.neighbors import NearestNeighbors

DATASET_PATH = Path.home() / (
    ".cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset"
    "/snapshots/71de6c7b6d83e4edad3cfbd10bfe9f8942f9d792"
    "/pusht/pusht_cchi_v7_replay.zarr"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize each column to [0, 1]."""
    lo = arr.min(0, keepdims=True)
    hi = arr.max(0, keepdims=True)
    return (arr - lo) / (hi - lo + 1e-8)


def wrap_angle_col(arr: np.ndarray, col: int) -> np.ndarray:
    """Wrap a single angle column to [-π, π] for correct distance."""
    out = arr.copy()
    out[:, col] = np.arctan2(np.sin(arr[:, col]), np.cos(arr[:, col]))
    return out


def load_data(dataset_path: Path, obs_mode: str):
    """Load raw state and action arrays from zarr.

    obs_mode:
      'block'  → [block_x, block_y, block_angle]  (what matters for strategy)
      'full'   → [agent_x, agent_y, block_x, block_y, block_angle]
    """
    z = zarr.open(str(dataset_path))
    state  = z["data/state"][:]   # (N, 5)
    action = z["data/action"][:]  # (N, 2)

    if obs_mode == "block":
        obs = state[:, 2:5]   # block_x, block_y, block_angle
    else:
        obs = state[:, :5]    # full state

    return obs, action, state


# ── core analysis ─────────────────────────────────────────────────────────────

def find_conflict_pairs(
    obs_norm: np.ndarray,
    action_norm: np.ndarray,
    eps: float,
    delta: float,
    k: int = 20,
) -> list[tuple[int, int]]:
    """Return index pairs (i, j) with |obs_i - obs_j| < eps and |act_i - act_j| > delta."""
    print(f"Running KNN (k={k}) on {len(obs_norm):,} timesteps …")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(obs_norm)
    distances, indices = nn.kneighbors(obs_norm)

    pairs = []
    for i, (dists, nbrs) in enumerate(zip(distances, indices)):
        for dist, j in zip(dists[1:], nbrs[1:]):   # skip self (index 0)
            if dist < eps:
                act_dist = np.linalg.norm(action_norm[i] - action_norm[j])
                if act_dist > delta:
                    pairs.append((i, j))

    # Deduplicate unordered pairs
    seen = set()
    unique_pairs = []
    for a, b in pairs:
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((a, b))

    print(f"Found {len(unique_pairs):,} conflict pairs.")
    return unique_pairs


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_conflict_audit(
    pairs: list[tuple[int, int]],
    state_raw: np.ndarray,
    action_raw: np.ndarray,
    obs_mode: str,
    eps: float,
    delta: float,
    out_path: str = "conflict_audit_pusht.png",
):
    block_xy  = state_raw[:, 2:4]   # raw pixel coords
    agent_xy  = state_raw[:, :2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Push-T Conflict Audit  |  obs={obs_mode}  |  ε={eps}  δ={delta}\n"
        f"{len(pairs):,} conflict pairs found",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Where are the conflict block positions? ──────────────────────
    ax = axes[0]
    ax.set_title("Conflict block positions (raw pixel space)")

    # Background: all block positions in grey
    ax.scatter(state_raw[:, 2], state_raw[:, 3], s=1, c="lightgrey", alpha=0.3, label="all")

    idx_a = np.array([p[0] for p in pairs])
    idx_b = np.array([p[1] for p in pairs])
    conf_pts = np.concatenate([block_xy[idx_a], block_xy[idx_b]])
    ax.scatter(conf_pts[:, 0], conf_pts[:, 1], s=4, c="crimson", alpha=0.5, label="conflict")

    ax.set_xlabel("block_x (px)")
    ax.set_ylabel("block_y (px)")
    ax.set_aspect("equal")
    ax.invert_yaxis()   # screen coords: y down
    ax.legend(markerscale=4, fontsize=8)

    # ── Panel 2: Action arrow fan for a sample of conflict pairs ─────────────
    ax = axes[1]
    ax.set_title("Conflicting action pairs (block position + action arrows)")

    sample_n = min(300, len(pairs))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(pairs), size=sample_n, replace=False)

    arrow_kw = dict(head_width=3, head_length=3, length_includes_head=True, linewidth=0.5)

    for k in sample_idx:
        i, j = pairs[k]
        # Use midpoint of block positions as origin
        bx = (block_xy[i, 0] + block_xy[j, 0]) / 2
        by = (block_xy[i, 1] + block_xy[j, 1]) / 2

        ai = action_raw[i] - np.array([bx, by])
        aj = action_raw[j] - np.array([bx, by])
        # Normalise arrow length for visibility
        scale = 25.0
        ai = ai / (np.linalg.norm(ai) + 1e-8) * scale
        aj = aj / (np.linalg.norm(aj) + 1e-8) * scale

        ax.arrow(bx, by, ai[0], ai[1], color="royalblue", alpha=0.4, **arrow_kw)
        ax.arrow(bx, by, aj[0], aj[1], color="tomato",    alpha=0.4, **arrow_kw)

    ax.set_xlabel("block_x (px)")
    ax.set_ylabel("block_y (px)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    patch_a = mpatches.Patch(color="royalblue", label="action_i")
    patch_b = mpatches.Patch(color="tomato",    label="action_j")
    ax.legend(handles=[patch_a, patch_b], fontsize=8)

    # ── Panel 3: Action divergence distribution ────────────────────────────────
    ax = axes[2]
    ax.set_title("Distribution of action divergence in conflict pairs")

    idx_a = np.array([p[0] for p in pairs])
    idx_b = np.array([p[1] for p in pairs])
    action_dists = np.linalg.norm(action_raw[idx_a] - action_raw[idx_b], axis=1)
    obs_dists    = np.linalg.norm(
        (state_raw[idx_a, 2:5] - state_raw[idx_b, 2:5]),
        axis=1,
    )

    ax.scatter(obs_dists, action_dists, s=3, alpha=0.4, c="steelblue")
    ax.axhline(delta * (action_raw.max() - action_raw.min()), color="red",
               linestyle="--", linewidth=1, label=f"δ threshold (normalised)")
    ax.set_xlabel("||obs_i - obs_j||  (raw units)")
    ax.set_ylabel("||act_i - act_j||  (raw units)")
    ax.set_title("Obs similarity vs Action divergence")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


def plot_hub_closeup(
    pairs: list[tuple[int, int]],
    state_raw: np.ndarray,
    action_raw: np.ndarray,
    obs_mode: str,
    top_n_hubs: int = 6,
    out_path: str = "conflict_hubs_closeup.png",
):
    """Find the densest conflict hubs and plot action fans for each."""
    from collections import Counter

    idx_a = np.array([p[0] for p in pairs])
    idx_b = np.array([p[1] for p in pairs])
    all_conflict_idx = np.concatenate([idx_a, idx_b])

    counts = Counter(all_conflict_idx.tolist())
    hub_indices = [idx for idx, _ in counts.most_common(top_n_hubs)]

    ncols = 3
    nrows = (top_n_hubs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()
    fig.suptitle(
        f"Top-{top_n_hubs} Multimodal Hubs  |  obs={obs_mode}",
        fontsize=13, fontweight="bold",
    )

    arrow_kw = dict(head_width=5, head_length=5, length_includes_head=True, linewidth=1.0)

    for panel, hub_idx in enumerate(hub_indices):
        ax = axes[panel]
        hub_block = state_raw[hub_idx, 2:4]

        # All partners of this hub
        partner_mask_a = idx_a == hub_idx
        partner_mask_b = idx_b == hub_idx
        partners = np.concatenate([idx_b[partner_mask_a], idx_a[partner_mask_b]])

        # Background: nearby block positions
        block_xy = state_raw[:, 2:4]
        ax.scatter(block_xy[:, 0], block_xy[:, 1], s=1, c="lightgrey", alpha=0.2)

        # Hub position
        ax.scatter(*hub_block, s=80, c="gold", edgecolors="black", zorder=5, label="hub")

        # Hub's own action
        ha = action_raw[hub_idx]
        direction = ha - hub_block
        direction = direction / (np.linalg.norm(direction) + 1e-8) * 30
        ax.arrow(*hub_block, *direction, color="royalblue", **arrow_kw)

        # Partner actions
        for p in partners:
            pa = action_raw[p]
            d = pa - hub_block
            d = d / (np.linalg.norm(d) + 1e-8) * 30
            ax.arrow(*hub_block, *d, color="tomato", alpha=0.6, **arrow_kw)

        ax.set_title(
            f"Hub #{panel+1}  (block=[{hub_block[0]:.0f},{hub_block[1]:.0f}])\n"
            f"{len(partners)} conflict partners",
            fontsize=9,
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("block_x")
        ax.set_ylabel("block_y")

    # Hide unused panels
    for i in range(top_n_hubs, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument(
        "--obs", default="block", choices=["block", "full"],
        help="Observation space for KNN: 'block'=[block_x,block_y,block_angle], 'full'=all 5 dims",
    )
    parser.add_argument("--eps",   type=float, default=0.06,
                        help="Max normalised obs distance to be 'similar'")
    parser.add_argument("--delta", type=float, default=0.20,
                        help="Min normalised action distance to be 'conflicting'")
    parser.add_argument("--k",     type=int,   default=20,
                        help="Number of KNN neighbours to search")
    parser.add_argument("--top-hubs", type=int, default=6,
                        help="Number of top hubs to plot in closeup figure")
    args = parser.parse_args()

    obs_raw, action_raw, state_raw = load_data(Path(args.dataset), args.obs)

    # Normalise for distance computation
    # For angle: wrap first, then normalise
    obs_for_knn = obs_raw.copy().astype(np.float64)
    if args.obs == "block":
        obs_for_knn = wrap_angle_col(obs_for_knn, 2)   # col 2 = block_angle
    elif args.obs == "full":
        obs_for_knn = wrap_angle_col(obs_for_knn, 4)   # col 4 = block_angle
    obs_norm    = normalize(obs_for_knn)
    action_norm = normalize(action_raw.astype(np.float64))

    print(f"Dataset: {args.dataset}")
    print(f"Obs mode: {args.obs}  |  shape: {obs_norm.shape}")
    print(f"ε={args.eps}  δ={args.delta}  k={args.k}")

    pairs = find_conflict_pairs(obs_norm, action_norm, args.eps, args.delta, args.k)

    if not pairs:
        print("No conflict pairs found. Try increasing ε or decreasing δ.")
        return

    plot_conflict_audit(
        pairs, state_raw, action_raw,
        obs_mode=args.obs, eps=args.eps, delta=args.delta,
        out_path="conflict_audit_pusht.png",
    )
    plot_hub_closeup(
        pairs, state_raw, action_raw,
        obs_mode=args.obs, top_n_hubs=args.top_hubs,
        out_path="conflict_hubs_closeup.png",
    )

    # ── Print statistics ───────────────────────────────────────────────────────
    idx_a = np.array([p[0] for p in pairs])
    idx_b = np.array([p[1] for p in pairs])
    act_dists = np.linalg.norm(action_raw[idx_a] - action_raw[idx_b], axis=1)
    obs_dists = np.linalg.norm(obs_norm[idx_a] - obs_norm[idx_b], axis=1)

    print("\n── Conflict Pair Statistics ────────────────────────────────")
    print(f"  Total pairs:           {len(pairs):>8,}")
    print(f"  Unique conflict nodes: {len(set(idx_a) | set(idx_b)):>8,}")
    print(f"  Action dist  mean/max: {act_dists.mean():.3f} / {act_dists.max():.3f}")
    print(f"  Obs dist     mean/max: {obs_dists.mean():.4f} / {obs_dists.max():.4f}")
    print("────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
