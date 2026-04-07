"""Build a multi-row LIBERO intent-outcome comparison figure.

Each row uses the highest-variance critical state from one saved
collect_intent_outcomes_image.py run and shows:
  1. 3D EEF trajectories for the fixed-intent clusters
  2. Final rendered frame for cluster 1
  3. Final rendered frame for cluster 2
  4. Final rendered frame for cluster 3
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from collect_intent_outcomes_image import _normalize_render_frame


def parse_row_spec(spec: str):
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"--row must be 'label:path', got: {spec}")
    return parts[0], parts[1]


def endpoint_spread(outcome: dict):
    cluster_outcomes = outcome.get("cluster_outcomes", [])
    if len(cluster_outcomes) < 2:
        return 0.0
    endpoints = []
    for co in cluster_outcomes:
        traj = np.asarray(co["eef_trajectory"], dtype=np.float32)
        if len(traj) == 0:
            continue
        endpoints.append(traj[-1])
    if len(endpoints) < 2:
        return 0.0
    endpoints = np.stack(endpoints, axis=0)
    d = endpoints[:, None, :] - endpoints[None, :, :]
    return float(np.mean(np.linalg.norm(d, axis=-1)))


def best_outcome(result: dict, metric: str = "variance"):
    outcomes = result.get("outcomes", [])
    if not outcomes:
        raise ValueError("No outcomes found in result file.")
    if metric == "endpoint_spread":
        return max(outcomes, key=endpoint_spread)
    return max(outcomes, key=lambda o: float(o.get("variance", 0.0)))


def format_frame(frame, mode="full"):
    frame = _normalize_render_frame(frame)
    if frame is None:
        return None
    if mode == "full":
        return frame
    if mode != "tabletop":
        raise ValueError(f"Unsupported frame mode: {mode}")
    h, w = frame.shape[:2]
    y0 = int(0.28 * h)
    y1 = int(0.98 * h)
    x0 = int(0.04 * w)
    x1 = int(0.96 * w)
    return frame[y0:y1, x0:x1]


def choose_projection(cluster_outcomes):
    pairs = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]
    best = pairs[0]
    best_score = -np.inf
    for pair in pairs:
        _, _, i, j = pair
        endpoints = np.array([np.asarray(co["eef_trajectory"])[-1, [i, j]] for co in cluster_outcomes])
        if len(endpoints) < 2:
            score = 0.0
        else:
            d = endpoints[:, None, :] - endpoints[None, :, :]
            score = float(np.mean(np.linalg.norm(d, axis=-1)))
        if score > best_score:
            best = pair
            best_score = score
    return best


def plot_row_v1(fig, n_rows, row_idx, row_label, outcome, n_clusters=3, frame_mode="full"):
    cluster_outcomes = outcome["cluster_outcomes"][:n_clusters]
    k = len(cluster_outcomes)
    colors = plt.cm.tab10(np.arange(max(k, 1)) / max(k - 1, 1))
    ts = outcome["timestep"]
    var = float(outcome["variance"])
    spread = endpoint_spread(outcome)
    start_eef = np.asarray(outcome["eef_pos"], dtype=np.float32)

    ax = fig.add_subplot(n_rows, 5, row_idx * 5 + 1, projection="3d")
    ax.scatter(
        start_eef[0],
        start_eef[1],
        start_eef[2],
        c="black",
        s=60,
        marker="o",
        zorder=10,
        label="start" if row_idx == 0 else None,
    )
    mins = [start_eef]
    maxs = [start_eef]
    for ki, co in enumerate(cluster_outcomes):
        traj = np.asarray(co["eef_trajectory"], dtype=np.float32)
        mins.append(traj.min(axis=0))
        maxs.append(traj.max(axis=0))
        c = colors[ki]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=c, lw=2.0, alpha=0.9)
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            traj[-1, 2],
            color=c,
            s=90,
            marker="*",
            zorder=6,
            label=f"Cluster {ki + 1}" if row_idx == 0 else None,
        )

    mins = np.min(np.stack(mins, axis=0), axis=0)
    maxs = np.max(np.stack(maxs, axis=0), axis=0)
    center = 0.5 * (mins + maxs)
    max_range = float(np.max(maxs - mins))
    if max_range < 1e-6:
        max_range = 1e-3
    half = 0.6 * max_range
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{row_label}\nt={ts} var={var:.4f}\nspread={spread:.4f}", fontsize=10)
    if row_idx == 0:
        ax.legend(loc="upper left")

    start_ax = fig.add_subplot(n_rows, 5, row_idx * 5 + 2)
    start_frames = cluster_outcomes[0].get("frames", []) if k > 0 else []
    start_im = format_frame(start_frames[0], mode=frame_mode) if start_frames else None
    if start_im is not None:
        start_ax.imshow(start_im)
    else:
        start_ax.text(0.5, 0.5, "No frame", ha="center", va="center", fontsize=8)
    start_ax.set_title("Start", fontsize=9)
    start_ax.axis("off")

    for ki in range(n_clusters):
        ax = fig.add_subplot(n_rows, 5, row_idx * 5 + 3 + ki)
        if ki < k:
            frames = cluster_outcomes[ki].get("frames", [])
            end_im = format_frame(frames[-1], mode=frame_mode) if frames else None
            if end_im is not None:
                ax.imshow(end_im)
            else:
                ax.text(0.5, 0.5, "No frame", ha="center", va="center", fontsize=8)
            ax.set_title(f"C{ki + 1} end", fontsize=9)
        else:
            ax.axis("off")
        ax.axis("off")


def plot_row(fig, n_rows, row_idx, row_label, outcome, n_clusters=3, frame_mode="full"):
    cluster_outcomes = outcome["cluster_outcomes"][:n_clusters]
    k = len(cluster_outcomes)
    colors = plt.cm.tab10(np.arange(max(k, 1)) / max(k - 1, 1))
    ts = outcome["timestep"]
    var = float(outcome["variance"])
    start_eef = np.asarray(outcome["eef_pos"], dtype=np.float32)

    # Column 1: best-separating 2D physical projection.
    ax = fig.add_subplot(n_rows, 4, row_idx * 4 + 1)
    a_name, b_name, ai, bi = choose_projection(cluster_outcomes)
    proj_points = [start_eef[[ai, bi]]]
    ax.scatter(
        start_eef[ai],
        start_eef[bi],
        c="black",
        s=60,
        marker="o",
        zorder=10,
        label="start" if row_idx == 0 else None,
    )
    for ki, co in enumerate(cluster_outcomes):
        traj = np.asarray(co["eef_trajectory"], dtype=np.float32)
        proj = traj[:, [ai, bi]]
        proj_points.append(proj)
        c = colors[ki]
        tail = proj[-min(len(proj), 18):]
        ax.plot(proj[:, 0], proj[:, 1], color=c, alpha=0.35, lw=1.2)
        ax.plot(tail[:, 0], tail[:, 1], color=c, alpha=0.95, lw=2.2)
        ax.scatter(
            proj[-1, 0],
            proj[-1, 1],
            color=c,
            s=75,
            marker="*",
            zorder=6,
        )
        ax.text(
            proj[-1, 0],
            proj[-1, 1],
            f" C{ki + 1}",
            color=c,
            fontsize=8,
            weight="bold",
            va="center",
        )

    proj = np.vstack(proj_points)
    proj_min = proj.min(axis=0)
    proj_max = proj.max(axis=0)
    center = 0.5 * (proj_min + proj_max)
    max_range = float(np.max(proj_max - proj_min))
    if max_range < 1e-6:
        max_range = 1e-3
    half = 0.62 * max_range
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(f"{a_name} (m)", fontsize=8)
    ax.set_ylabel(f"{b_name} (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{row_label}\nt={ts}  var={var:.4f}  view={a_name}{b_name}", fontsize=10, weight="bold")

    # Inset start frame crop so the perturbation object is visible.
    inset = inset_axes(ax, width="42%", height="42%", loc="upper right", borderpad=0.9)
    start_frames = cluster_outcomes[0].get("frames", [])
    start_im = format_frame(start_frames[0], mode=frame_mode) if start_frames else None
    if start_im is not None:
        inset.imshow(start_im)
        inset.set_title("start scene", fontsize=7)
    else:
        inset.text(0.5, 0.5, "no start", ha="center", va="center", fontsize=7)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)

    # Columns 2-4: final rendered frames.
    for ki in range(n_clusters):
        ax = fig.add_subplot(n_rows, 4, row_idx * 4 + 2 + ki)
        if ki < k:
            frames = cluster_outcomes[ki].get("frames", [])
            end_im = format_frame(frames[-1], mode=frame_mode) if frames else None
            if end_im is not None:
                ax.imshow(end_im)
            else:
                ax.text(0.5, 0.5, "No frame", ha="center", va="center", fontsize=8)
            ax.set_title(f"C{ki + 1} end", fontsize=9, color=colors[ki], weight="bold")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.2)
                spine.set_edgecolor(colors[ki])
        else:
            ax.axis("off")
        ax.axis("off")


def main():
    parser = argparse.ArgumentParser(
        description="Create a stacked LIBERO intent-outcome comparison figure."
    )
    parser.add_argument(
        "--row",
        action="append",
        required=True,
        metavar="label:path",
        help="Row label and intent_outcomes.pkl path. Repeat for multiple rows.",
    )
    parser.add_argument(
        "--style",
        choices=("v1", "v2"),
        default="v2",
        help="Figure layout style. Use v1 for the original 3D-left layout.",
    )
    parser.add_argument(
        "--select-by",
        choices=("variance", "endpoint_spread"),
        default="variance",
        help="How to choose one critical state per row.",
    )
    parser.add_argument(
        "--frame-crop",
        choices=("full", "tabletop"),
        default="full",
        help="How to display rendered frames inside the combined grid.",
    )
    parser.add_argument("--out", required=True, help="Output PNG path.")
    args = parser.parse_args()

    rows = []
    for spec in args.row:
        label, path = parse_row_spec(spec)
        with open(path, "rb") as f:
            result = pickle.load(f)
        rows.append((label, best_outcome(result, metric=args.select_by)))

    n_rows = len(rows)
    fig = plt.figure(figsize=((18 if args.style == "v1" else 15), 3.5 * n_rows))
    for row_idx, (label, outcome) in enumerate(rows):
        if args.style == "v1":
            plot_row_v1(fig, n_rows, row_idx, label, outcome, frame_mode=args.frame_crop)
        else:
            plot_row(fig, n_rows, row_idx, label, outcome, frame_mode=args.frame_crop)

    if args.style == "v1":
        fig.suptitle(
            f"LIBERO Intent-Outcome Comparison\nleft: selected EEF trajectory, then start frame, then cluster end frames ({args.frame_crop})",
            fontsize=14,
            fontweight="bold",
        )
    else:
        fig.suptitle(
            f"LIBERO Intent-Outcome Comparison\nleft: best-separating EEF view + start scene, right: rendered end states ({args.frame_crop})",
            fontsize=14,
            fontweight="bold",
        )
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
