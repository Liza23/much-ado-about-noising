"""Analyze BC or flow-intent reset-state distribution for lift-mh-state.

Rolls out a policy and a random-action baseline from N independently seeded
resets. Fits a Gaussian KDE on training-demo initial object positions and uses
-log(KDE density) as an OOD score for each eval initial state.

Supports parallel collection via --seed-start / --seed-end (saves a shard pkl),
then merging shards with --merge-only to produce figures.

Output (--out-dir):
    rollouts.pkl             — raw rollout data (reuse with --skip-collect)
    reset_dist.png           — 2D object (x,y) scatter, colored by BC success
    ood_histogram.png        — OOD score distributions split by outcome
    pareto_front.png         — Coverage-vs-success Pareto curves (BC & random)
    counterfactual_pairs.png — Nearest success/failure pairs with EEF trajectories
    random_control.png       — Grouped bar chart by OOD tier (low/mid/high)

Usage (single job):
    python examples/analyze_bc_reset_distribution.py \\
        --ckpt checkpoints/lift_mh_state_flow_mlp_512_h10_seed0_success100.pt \\
        --n-rollouts 200 --device cuda --out-dir rollouts/bc_reset_dist_lift_mh_state

Usage (parallel, 3 shards then merge):
    python examples/analyze_bc_reset_distribution.py --variant flow_intent \\
        --ckpt <ckpt> --seed-start 0   --seed-end 67  --out-dir rollouts/fi_reset_dist
    python examples/analyze_bc_reset_distribution.py --variant flow_intent \\
        --ckpt <ckpt> --seed-start 67  --seed-end 134 --out-dir rollouts/fi_reset_dist
    python examples/analyze_bc_reset_distribution.py --variant flow_intent \\
        --ckpt <ckpt> --seed-start 134 --seed-end 200 --out-dir rollouts/fi_reset_dist
    python examples/analyze_bc_reset_distribution.py --merge-only \\
        --out-dir rollouts/fi_reset_dist
"""

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from mip.agent import TrainingAgent
from mip.flow_intent_agent import FlowIntentAgent
from mip.datasets.robomimic_dataset import make_dataset as make_dataset_robomimic
from mip.datasets.pusht_dataset import make_dataset as make_dataset_pusht
from mip.envs.robomimic.robomimic_env import make_vec_env as make_vec_env_robomimic
from mip.envs.pusht.pusht_env_wrapper import make_vec_env as make_vec_env_pusht
from mip.samplers import get_default_step_list
from mip.torch_utils import set_seed

ABS_ACTION_ENVS = {"can", "lift", "square", "tool_hang", "transport"}

# ── Per-task configuration ────────────────────────────────────────────────────
TASK_CONFIGS = {
    "lift_mh_state": {
        # obs layout: [object(10D), eef_pos(3D), eef_quat(4D), gripper(2D)]
        "obj_pos_slice": slice(0, 2),    # cube XY from raw obs[0:3][:2]
        "obj_pos_full_slice": slice(0, 3),  # cube XYZ (used to extract then drop Z)
        "traj_pos_slice": slice(10, 13), # eef XYZ
        "success_from_terminated": False,  # use ep_reward > 0
        "base_task_cfg": "lift_mh_state",
        "base_network_cfg": "mlp",
        "fi_task_cfg": "lift_mh_state_flow_intent",
        "fi_network_cfg": "mlp_flow_intent",
        "xlabel": "Object x (world frame)",
        "ylabel": "Object y (world frame)",
        "title_env": "lift-mh-state",
        "traj_xlabel": "EEF x",
        "traj_ylabel": "EEF y",
    },
    "pusht_state": {
        # obs layout: [agent_x, agent_y, block_x, block_y, block_angle]
        "obj_pos_slice": slice(2, 4),    # block XY (already 2D)
        "obj_pos_full_slice": slice(2, 4),
        "traj_pos_slice": slice(0, 2),   # agent XY
        "success_from_terminated": True,  # terminated=True means coverage>95%
        "base_task_cfg": "pusht_state",
        "base_network_cfg": "mlp",
        "fi_task_cfg": "pusht_state_flow_intent",
        "fi_network_cfg": "mlp_flow_intent",
        "xlabel": "Block x (pixels)",
        "ylabel": "Block y (pixels)",
        "title_env": "pusht-state",
        "traj_xlabel": "Agent x",
        "traj_ylabel": "Agent y",
    },
}


# ── Config / model loading ────────────────────────────────────────────────────

def load_config(overrides: list[str]):
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/configs"), version_base=None
    ):
        cfg = compose("main", overrides=overrides)
    return cfg


def _patch_config_from_checkpoint(checkpoint_path: str, config):
    """Infer horizon / obs_steps from checkpoint weights and patch config."""
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # FlowIntentAgent checkpoints have a different key structure — skip patching
    if "intent_flow_map" in sd:
        return
    fm = sd.get("flow_map", {})
    out_w = fm.get("net.main_output.weight")
    in_w = fm.get("net.input_proj.weight")
    if out_w is None or in_w is None:
        return
    act_dim = config.task.act_dim
    Ta = out_w.shape[0] // act_dim
    input_dim = in_w.shape[1]
    freq = fm.get("net.frequencies")
    if freq is not None:
        time_contribution = freq.shape[0] * 4
    else:
        time_contribution = 2
    obs_flat = input_dim - act_dim * Ta - time_contribution
    emb_dim = in_w.shape[0]
    if obs_flat % emb_dim == 0:
        config.task.obs_steps = obs_flat // emb_dim
    config.task.horizon = Ta


def _make_vec_env(task_config, seed=0):
    if getattr(task_config, "env_name", "") == "pusht":
        return make_vec_env_pusht(task_config, seed=seed)
    return make_vec_env_robomimic(task_config, seed=seed)


def _make_dataset(task_config):
    if getattr(task_config, "env_name", "") == "pusht":
        return make_dataset_pusht(task_config)
    return make_dataset_robomimic(task_config)


def setup_config_and_env(overrides: list[str]):
    """Load config, build envs, infer obs_dim at runtime. Returns (config, envs, init_obs)."""
    config = load_config(overrides)
    envs = _make_vec_env(config.task, seed=0)

    # Mirror train_robomimic.py: set obs_dim from actual env
    obs, _ = envs.reset()
    config.task.obs_dim = obs.shape[-1]
    return config, envs, obs


def load_bc_agent(ckpt_path: str, config, device: str) -> TrainingAgent:
    """Load a plain TrainingAgent (BC w/o intent) from checkpoint."""
    _patch_config_from_checkpoint(ckpt_path, config)
    agent = TrainingAgent(config)
    agent.load(ckpt_path, load_optimizer=False)
    agent.eval()
    return agent


def load_flow_intent_agent(ckpt_path: str, config, device: str) -> FlowIntentAgent:
    """Load a FlowIntentAgent from a flow-intent checkpoint."""
    agent = FlowIntentAgent(config)
    agent.load(ckpt_path, load_optimizer=False)
    agent.eval()
    return agent


# ── Single-episode rollout ────────────────────────────────────────────────────

def _add_intent(obs_t: torch.Tensor, config) -> torch.Tensor:
    """Append CV-proxy intent to obs tensor for flow-intent variant.

    Intent = current eef pos+quat (indices 10:17 in normalized obs), replicated
    across all obs_steps. This matches the eval-time proxy used during training.
    """
    # eef pos+quat occupies indices [10:17] in the normalized state obs
    intent_start, intent_end = 10, 17
    eef_now = obs_t[:, -1, intent_start:intent_end]          # (B, 7)
    intent_exp = eef_now.unsqueeze(1).expand(-1, obs_t.shape[1], -1)  # (B, obs_steps, 7)
    return torch.cat([obs_t, intent_exp], dim=-1)


def _run_episode_bc(agent, envs, init_obs_raw, dataset, config, device: str,
                    num_steps: int, task_info: dict,
                    flow_intent: bool = False) -> tuple[bool, np.ndarray]:
    """Run one BC episode starting from init_obs_raw. Returns (success, traj)."""
    obs_raw = init_obs_raw
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    max_t = config.task.max_episode_steps
    traj_sl = task_info["traj_pos_slice"]
    success_from_terminated = task_info["success_from_terminated"]

    success = False
    ep_reward = 0.0
    traj = [obs_raw[0, -1, traj_sl].copy()]

    t = 0
    while t < max_t:
        obs_f = obs_raw.astype(np.float32)
        obs_norm = dataset.normalizer["obs"]["state"].normalize(obs_f)
        obs_t = torch.tensor(obs_norm, device=device, dtype=torch.float32)

        with torch.no_grad():
            if flow_intent:
                act_normed = agent.sample(obs=obs_t, num_steps=num_steps, use_ema=True)
            else:
                act_0 = torch.randn(
                    (1, config.task.horizon, config.task.act_dim), device=device
                )
                act_normed = agent.sample(
                    act_0=act_0, obs={"state": obs_t}, num_steps=num_steps, use_ema=True
                )

        act_un = dataset.normalizer["action"].unnormalize(act_normed.cpu().numpy())
        act_chunk = act_un[:, s:s + act_steps]

        if getattr(config.task, "abs_action", False) and config.task.env_name in ABS_ACTION_ENVS:
            act_env = dataset.undo_transform_action(act_chunk)
        else:
            act_env = act_chunk

        obs_raw, reward, terminated, truncated, info = envs.step(act_env)
        ep_reward += float(np.asarray(reward).sum())
        traj.append(obs_raw[0, -1, traj_sl].copy())

        if not success_from_terminated:
            if "_final_info" in info and "final_info" in info:
                for i in range(config.task.num_envs):
                    if info["_final_info"][i]:
                        fi = info["final_info"][i]
                        if fi and "success" in fi:
                            success = success or bool(np.asarray(fi["success"]).any())
            if "success" in info:
                sv = info["success"]
                for i in range(config.task.num_envs):
                    v = sv[i] if hasattr(sv, "__len__") else sv
                    success = success or bool(np.asarray(v).any())

        t += act_steps
        done = bool(np.asarray(terminated).any())
        if done:
            if success_from_terminated:
                success = True
            break

    if not success_from_terminated:
        success = success or (ep_reward > 0)
    return success, np.array(traj)


def _run_episode_random(envs, init_obs_raw, config, env_act_dim: int,
                        task_info: dict) -> tuple[bool, np.ndarray]:
    """Run one random-action episode starting from init_obs_raw."""
    obs_raw = init_obs_raw
    act_steps = config.task.act_steps
    max_t = config.task.max_episode_steps
    traj_sl = task_info["traj_pos_slice"]
    success_from_terminated = task_info["success_from_terminated"]

    success = False
    ep_reward = 0.0
    traj = [obs_raw[0, -1, traj_sl].copy()]

    t = 0
    while t < max_t:
        act_env = np.random.uniform(-1, 1, (1, act_steps, env_act_dim))
        obs_raw, reward, terminated, truncated, info = envs.step(act_env)
        ep_reward += float(np.asarray(reward).sum())
        traj.append(obs_raw[0, -1, traj_sl].copy())

        if not success_from_terminated:
            if "_final_info" in info and "final_info" in info:
                for i in range(config.task.num_envs):
                    if info["_final_info"][i]:
                        fi = info["final_info"][i]
                        if fi and "success" in fi:
                            success = success or bool(np.asarray(fi["success"]).any())
            if "success" in info:
                sv = info["success"]
                for i in range(config.task.num_envs):
                    v = sv[i] if hasattr(sv, "__len__") else sv
                    success = success or bool(np.asarray(v).any())

        t += act_steps
        done = bool(np.asarray(terminated).any())
        if done:
            if success_from_terminated:
                success = True
            break

    if not success_from_terminated:
        success = success or (ep_reward > 0)
    return success, np.array(traj)


# ── Training distribution ─────────────────────────────────────────────────────

def load_train_obj_positions(task_config, task_info: dict, dataset=None) -> np.ndarray:
    """Load initial object positions from training demos. Returns (M, D) where D=2 or 3."""
    if getattr(task_config, "env_name", "") == "pusht":
        # Zarr dataset: initial state = first timestep of each episode
        rb = dataset.replay_buffer
        ep_starts = np.concatenate([[0], rb.episode_ends[:-1]])
        init_states = rb["state"][ep_starts]                     # (M, 5)
        sl = task_info["obj_pos_full_slice"]
        return init_states[:, sl]                                # (M, 2) block XY

    # HDF5 (lift-mh-state and similar Robomimic tasks)
    if hasattr(task_config, "dataset_repo") and hasattr(task_config, "dataset_filename"):
        dataset_path = hf_hub_download(
            repo_id=task_config.dataset_repo,
            filename=task_config.dataset_filename,
            repo_type="dataset",
        )
    else:
        dataset_path = os.path.expanduser(task_config.dataset_path)

    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        # data/demo_i/obs/object has shape (T, 10); [0, :3] = initial cube world XYZ
        train_obj_pos = np.stack(
            [f[f"data/{dk}/obs/object"][0, :3] for dk in demo_keys]
        )
    return train_obj_pos  # (M, 3)


# ── Main collection ───────────────────────────────────────────────────────────

def collect_rollouts(agent, envs, dataset, config, device: str,
                     n_rollouts: int, num_steps: int, task_info: dict,
                     seed_start: int = 0, flow_intent: bool = False) -> dict:
    """Roll out policy and random from n_rollouts seeds. Returns dict of arrays."""
    env_act_dim = envs.action_space.shape[-1]
    obj_sl = task_info["obj_pos_full_slice"]   # slice into raw obs (last step)
    xy_sl = task_info["obj_pos_slice"]          # which dims to keep as XY (may be 2D already)
    obj_pos_list = []
    bc_success_list = []
    rand_success_list = []
    bc_traj_list = []
    rand_traj_list = []

    for i in range(n_rollouts):
        seed = seed_start + i
        if i % 20 == 0:
            print(f"  Seed {seed} ({i}/{n_rollouts}) ...", flush=True)

        # ── BC episode ──
        obs_bc, _ = envs.reset(seed=[seed])  # (1, obs_steps, obs_dim) raw
        initial_obj = obs_bc[0, -1, obj_sl].copy()  # (2,) or (3,)

        bc_ok, bc_traj = _run_episode_bc(
            agent, envs, obs_bc, dataset, config, device, num_steps,
            task_info=task_info, flow_intent=flow_intent,
        )

        # ── Random episode from same initial state ──
        obs_rand, _ = envs.reset(seed=[seed])
        rand_ok, rand_traj = _run_episode_random(envs, obs_rand, config, env_act_dim,
                                                  task_info=task_info)

        obj_pos_list.append(initial_obj[:2])  # keep (x, y)
        bc_success_list.append(bc_ok)
        rand_success_list.append(rand_ok)
        bc_traj_list.append(bc_traj)
        rand_traj_list.append(rand_traj)

    return {
        "obj_pos": np.array(obj_pos_list),         # (N, 2) raw x, y
        "bc_success": np.array(bc_success_list),   # (N,) bool
        "rand_success": np.array(rand_success_list),
        "bc_traj_eef": bc_traj_list,               # list of (T_i, D) arrays
        "rand_traj_eef": rand_traj_list,
    }


# ── OOD scoring ───────────────────────────────────────────────────────────────

def compute_ood_scores(eval_obj_pos_xy: np.ndarray,
                       train_obj_pos_xy: np.ndarray) -> np.ndarray:
    """Return OOD scores via -log(KDE density). Higher = more OOD. Returns (N,)."""
    kde = gaussian_kde(train_obj_pos_xy.T, bw_method="scott")
    log_density = kde.logpdf(eval_obj_pos_xy.T)
    ood = -log_density
    # Normalize to [0, 1]
    ood_min, ood_max = ood.min(), ood.max()
    if ood_max > ood_min:
        ood = (ood - ood_min) / (ood_max - ood_min)
    return ood, kde


# ── Figures ───────────────────────────────────────────────────────────────────

def _setup_fig(figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_reset_distribution(data: dict, train_obj_pos_xy: np.ndarray,
                             kde, out_path: Path, task_info: dict):
    """Fig 1: 2D scatter of object (x,y) colored by BC success, KDE contour overlay."""
    obj = data["obj_pos"]  # (N, 2)
    bc_ok = data["bc_success"]

    fig, ax = _setup_fig((7, 6))

    # KDE contour of training distribution
    xmin, xmax = train_obj_pos_xy[:, 0].min(), train_obj_pos_xy[:, 0].max()
    ymin, ymax = train_obj_pos_xy[:, 1].min(), train_obj_pos_xy[:, 1].max()
    pad_x = (xmax - xmin) * 0.5
    pad_y = (ymax - ymin) * 0.5
    xx, yy = np.mgrid[xmin - pad_x:xmax + pad_x:80j,
                       ymin - pad_y:ymax + pad_y:80j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    z = np.exp(kde.logpdf(grid)).reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=6, cmap="Greys", alpha=0.35)
    ax.contour(xx, yy, z, levels=4, colors="gray", linewidths=0.8, alpha=0.6)

    # Eval scatter
    ax.scatter(obj[~bc_ok, 0], obj[~bc_ok, 1],
               c="#e74c3c", s=30, alpha=0.7, zorder=3, label="BC failure")
    ax.scatter(obj[bc_ok, 0], obj[bc_ok, 1],
               c="#2ecc71", s=30, alpha=0.7, zorder=4, label="BC success")

    ax.set_xlabel(task_info["xlabel"])
    ax.set_ylabel(task_info["ylabel"])
    ax.set_title(f"BC Reset Distribution — {task_info['title_env']}\n(gray = training demo density)")
    ax.legend(loc="upper right", fontsize=9)

    sr = bc_ok.mean()
    ax.text(0.02, 0.98, f"BC success: {sr:.1%}  (N={len(bc_ok)})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_ood_histogram(data: dict, out_path: Path):
    """Fig 2: OOD score distributions split by BC success/failure."""
    ood = data["ood_score"]
    bc_ok = data["bc_success"]

    fig, ax = _setup_fig((6, 4))
    bins = np.linspace(0, 1, 25)
    ax.hist(ood[bc_ok],  bins=bins, alpha=0.65, color="#2ecc71", label="BC success")
    ax.hist(ood[~bc_ok], bins=bins, alpha=0.65, color="#e74c3c", label="BC failure")
    ax.set_xlabel("OOD score (normalised, higher = more OOD)")
    ax.set_ylabel("Count")
    ax.set_title("OOD Score Distribution by BC Outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_pareto_front(data: dict, out_path: Path):
    """Fig 3: Coverage–performance Pareto curve as OOD threshold varies.

    x-axis: OOD threshold τ (states with ood_score ≤ τ are included)
    y-axis: BC / random success rate on included states
    Shaded region: 'BC advantage' over random baseline.
    """
    ood = data["ood_score"]
    bc_ok = data["bc_success"].astype(float)
    rand_ok = data["rand_success"].astype(float)
    N = len(ood)

    thresholds = np.linspace(0, 1, 60)
    bc_sr, rand_sr, coverage = [], [], []

    for tau in thresholds:
        mask = ood <= tau
        n = mask.sum()
        if n < 3:
            bc_sr.append(np.nan)
            rand_sr.append(np.nan)
            coverage.append(n / N)
            continue
        bc_sr.append(bc_ok[mask].mean())
        rand_sr.append(rand_ok[mask].mean())
        coverage.append(n / N)

    bc_sr = np.array(bc_sr)
    rand_sr = np.array(rand_sr)
    coverage = np.array(coverage)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: success rate vs OOD threshold
    ax = axes[0]
    ax.plot(thresholds, bc_sr, color="#2980b9", lw=2, label="BC")
    ax.plot(thresholds, rand_sr, color="#e67e22", lw=2, linestyle="--", label="Random")
    ax.fill_between(thresholds, rand_sr, bc_sr,
                    where=(bc_sr > rand_sr), alpha=0.2, color="#2980b9",
                    label="BC advantage")
    ax.set_xlabel("OOD threshold τ")
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate vs OOD level")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)

    # Right: Pareto front (coverage vs success rate) — BC only
    ax2 = axes[1]
    valid = ~np.isnan(bc_sr)
    ax2.plot(coverage[valid], bc_sr[valid], color="#2980b9", lw=2, label="BC")
    ax2.plot(coverage[valid], rand_sr[valid], color="#e67e22", lw=2,
             linestyle="--", label="Random")
    ax2.fill_between(coverage[valid], rand_sr[valid], bc_sr[valid],
                     where=(bc_sr[valid] > rand_sr[valid]),
                     alpha=0.2, color="#2980b9", label="BC advantage")
    ax2.set_xlabel("Coverage fraction (fraction of eval states included)")
    ax2.set_ylabel("Success rate")
    ax2.set_title("Coverage–Performance Pareto Front")
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_counterfactual_pairs(data: dict, out_path: Path, task_info: dict, n_pairs: int = 5):
    """Fig 4: Nearest success/failure pairs with 2D agent/EEF trajectories (top-down)."""
    obj = data["obj_pos"]        # (N, 2)
    bc_ok = data["bc_success"]
    bc_trajs = data["bc_traj_eef"]  # list of (T, D) arrays

    fail_idx = np.where(~bc_ok)[0]
    succ_idx = np.where(bc_ok)[0]

    if len(fail_idx) == 0 or len(succ_idx) == 0:
        print("  [skip] not enough successes or failures for counterfactual plot")
        return

    # Pairwise distances in XY between failures and successes
    dists = cdist(obj[fail_idx], obj[succ_idx])
    nearest = dists.argmin(axis=1)
    pair_dists = dists[np.arange(len(fail_idx)), nearest]

    # Take the n_pairs closest pairs
    top_k = np.argsort(pair_dists)[:n_pairs]
    pairs = [(fail_idx[i], succ_idx[nearest[i]], pair_dists[i]) for i in top_k]

    n_actual = min(n_pairs, len(pairs))
    fig, axes = plt.subplots(1, n_actual, figsize=(4 * n_actual, 4))
    if n_actual == 1:
        axes = [axes]

    obj_label = task_info["xlabel"].split(" ")[0].lower()  # e.g. "Object" or "Block"
    for ax, (fi, si, dist) in zip(axes, pairs):
        t_fail = bc_trajs[fi]   # (T, D)
        t_succ = bc_trajs[si]

        ax.plot(t_fail[:, 0], t_fail[:, 1], color="#e74c3c", lw=1.5,
                label=f"Failure\n{obj_label}=({obj[fi, 0]:.1f},{obj[fi, 1]:.1f})")
        ax.plot(t_succ[:, 0], t_succ[:, 1], color="#2ecc71", lw=1.5,
                label=f"Success\n{obj_label}=({obj[si, 0]:.1f},{obj[si, 1]:.1f})")

        # Mark initial object positions
        ax.scatter(*obj[fi], marker="x", c="#e74c3c", s=80, zorder=5)
        ax.scatter(*obj[si], marker="*", c="#2ecc71", s=80, zorder=5)

        ax.set_xlabel(task_info["traj_xlabel"])
        ax.set_ylabel(task_info["traj_ylabel"])
        ax.set_title(f"Δ{obj_label}={dist:.2f}")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Counterfactual Pairs — Nearest Success/Failure in {task_info['title_env']}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_random_control(data: dict, out_path: Path):
    """Fig 5: Grouped bar chart — BC vs random success rate per OOD tier."""
    ood = data["ood_score"]
    bc_ok = data["bc_success"].astype(float)
    rand_ok = data["rand_success"].astype(float)

    # 3 equal-width OOD tiers
    tier_labels = ["Low OOD\n(0–33%)", "Mid OOD\n(33–66%)", "High OOD\n(66–100%)"]
    tiers = [ood < 0.33, (ood >= 0.33) & (ood < 0.66), ood >= 0.66]

    bc_means, rand_means, ns = [], [], []
    for mask in tiers:
        n = mask.sum()
        ns.append(n)
        bc_means.append(bc_ok[mask].mean() if n > 0 else 0.0)
        rand_means.append(rand_ok[mask].mean() if n > 0 else 0.0)

    x = np.arange(3)
    w = 0.35
    fig, ax = _setup_fig((7, 4))
    bars_bc = ax.bar(x - w / 2, bc_means, w, label="BC", color="#2980b9", alpha=0.85)
    bars_rn = ax.bar(x + w / 2, rand_means, w, label="Random", color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(tier_labels, ns)])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("BC vs Random Policy — Success by OOD Tier")
    ax.legend()

    for bar in bars_bc:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_rn:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _merge_shards(out_dir: Path) -> dict:
    """Merge all shard pkls (rollouts_*.pkl) in out_dir into a single dict."""
    shards = sorted(out_dir.glob("rollouts_*.pkl"))
    if not shards:
        raise FileNotFoundError(f"No shard pkl files found in {out_dir}")
    parts = [pickle.load(open(p, "rb")) for p in shards]
    merged = {}
    for key in parts[0]:
        vals = [p[key] for p in parts]
        if isinstance(vals[0], np.ndarray):
            merged[key] = np.concatenate(vals, axis=0)
        else:
            merged[key] = sum(vals, [])
    print(f"  Merged {len(shards)} shards → {len(merged['bc_success'])} rollouts")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Analyze policy reset distribution (lift-mh-state or pusht-state)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", choices=list(TASK_CONFIGS.keys()),
                        default="lift_mh_state",
                        help="Which task to analyze")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Policy checkpoint path (required unless --merge-only or --skip-collect)",
    )
    parser.add_argument("--variant", choices=["bc", "flow_intent"], default="bc",
                        help="bc: no intent; flow_intent: flow-intent agent")
    parser.add_argument("--n-rollouts", type=int, default=200,
                        help="Total number of seeded eval episodes")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First env seed (for parallel sharding)")
    parser.add_argument("--seed-end", type=int, default=None,
                        help="Exclusive end seed (default: seed-start + n-rollouts)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for pkl and figures (default: rollouts/bc_reset_dist_<task>)")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip rollout collection; load existing rollouts.pkl")
    parser.add_argument("--merge-only", action="store_true",
                        help="Merge shard pkls and generate figures; skip collection")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    task_info = TASK_CONFIGS[args.task]
    flow_intent = args.variant == "flow_intent"

    if args.out_dir is None:
        args.out_dir = f"rollouts/bc_reset_dist_{args.task}{'_fi' if flow_intent else ''}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_start = args.seed_start
    seed_end = args.seed_end if args.seed_end is not None else seed_start + args.n_rollouts
    n_shard = seed_end - seed_start
    is_shard = (seed_start != 0 or seed_end != args.n_rollouts)
    shard_pkl = out_dir / f"rollouts_{seed_start}_{seed_end}.pkl"
    full_pkl = out_dir / "rollouts.pkl"

    # ── Variant-specific Hydra overrides ─────────────────────────────────────
    if flow_intent:
        task_cfg = task_info["fi_task_cfg"]
        network_cfg = task_info["fi_network_cfg"]
    else:
        task_cfg = task_info["base_task_cfg"]
        network_cfg = task_info["base_network_cfg"]

    base_overrides = [
        f"task={task_cfg}",
        f"network={network_cfg}",
        f"optimization.device={args.device}",
        "task.num_envs=1",
        f"optimization.seed={args.seed}",
    ]

    # ── Phase 0: training distribution ───────────────────────────────────────
    print(f"\n=== Phase 0: Loading training distribution ({args.task}) ===")
    # Use base (non-flow-intent) config to load dataset for training positions
    base_cfg = load_config([
        f"task={task_info['base_task_cfg']}",
        f"network={task_info['base_network_cfg']}",
        f"optimization.device={args.device}",
        "task.num_envs=1",
    ])
    # For pusht we need the dataset object; for lift we only need task_config
    if getattr(base_cfg.task, "env_name", "") == "pusht":
        train_dataset = _make_dataset(base_cfg.task)
    else:
        train_dataset = None
    train_obj_pos = load_train_obj_positions(base_cfg.task, task_info, train_dataset)
    # Always use first 2 dims as XY
    train_obj_pos_xy = train_obj_pos[:, :2]
    print(f"  Training demos: {len(train_obj_pos_xy)} initial states")
    print(f"  Training XY range: x=[{train_obj_pos_xy[:,0].min():.3f}, "
          f"{train_obj_pos_xy[:,0].max():.3f}] "
          f"y=[{train_obj_pos_xy[:,1].min():.3f}, {train_obj_pos_xy[:,1].max():.3f}]")

    # ── Phase 1: collect rollouts ─────────────────────────────────────────────
    if args.merge_only:
        print(f"\n=== Phase 1: Merging shards from {out_dir} ===")
        data = _merge_shards(out_dir)
        with open(full_pkl, "wb") as f:
            pickle.dump(data, f)
    elif args.skip_collect and full_pkl.exists():
        print(f"\n=== Phase 1: Loading cached rollouts from {full_pkl} ===")
        with open(full_pkl, "rb") as f:
            data = pickle.load(f)
    else:
        print(f"\n=== Phase 1: Collecting seeds {seed_start}–{seed_end} "
              f"(task={args.task}, variant={args.variant}) ===")
        if args.ckpt is None:
            raise ValueError("--ckpt is required for rollout collection")
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        config, envs, _ = setup_config_and_env(base_overrides)
        dataset = _make_dataset(config.task)
        if flow_intent:
            agent = load_flow_intent_agent(str(ckpt_path), config, args.device)
        else:
            agent = load_bc_agent(str(ckpt_path), config, args.device)

        num_steps = int(get_default_step_list(config.optimization.loss_type)[0])
        print(f"  ODE steps: {num_steps}  |  act_steps: {config.task.act_steps}  "
              f"|  horizon: {config.task.horizon}")

        data = collect_rollouts(
            agent, envs, dataset, config, args.device,
            n_rollouts=n_shard,
            num_steps=num_steps,
            task_info=task_info,
            seed_start=seed_start,
            flow_intent=flow_intent,
        )
        envs.close()

        bc_sr = data["bc_success"].mean()
        rand_sr = data["rand_success"].mean()
        print(f"  Policy success: {bc_sr:.1%}  |  Random success: {rand_sr:.1%}")

        out_pkl = shard_pkl if is_shard else full_pkl
        with open(out_pkl, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved rollouts to {out_pkl}")

        if is_shard:
            print("  Shard saved. Run --merge-only when all shards are done.")
            return

    # ── Phase 2: OOD scoring ──────────────────────────────────────────────────
    print("\n=== Phase 2: Computing OOD scores ===")
    ood, kde = compute_ood_scores(data["obj_pos"], train_obj_pos_xy)
    data["ood_score"] = ood
    print(f"  OOD range: [{ood.min():.3f}, {ood.max():.3f}]")
    bc_succ = np.array(data["bc_success"]) if not isinstance(data["bc_success"], np.ndarray) else data["bc_success"]
    print(f"  OOD by outcome:  Policy success={ood[bc_succ].mean():.3f}  "
          f"Policy failure={ood[~bc_succ].mean():.3f}")

    # ── Phase 3: figures ──────────────────────────────────────────────────────
    print("\n=== Phase 3: Generating figures ===")

    plot_reset_distribution(data, train_obj_pos_xy, kde,
                            out_dir / "reset_dist.png", task_info)
    plot_ood_histogram(data, out_dir / "ood_histogram.png")
    plot_pareto_front(data, out_dir / "pareto_front.png")
    plot_counterfactual_pairs(data, out_dir / "counterfactual_pairs.png", task_info)
    plot_random_control(data, out_dir / "random_control.png")

    print(f"\nAll outputs written to: {out_dir}/")


if __name__ == "__main__":
    main()
