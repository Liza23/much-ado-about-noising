"""Comprehensive visualization + wandb logging for intent-conditioned policy ablations.

Collects rollout trajectories, intent predictions, and action distributions
across methods, then generates and logs all visualizations to wandb.

Usage:
    python examples/eval_visualize.py \
        --run "baseline:checkpoints/lift_ph_state_flow_mlp_512_seed0_success98.pt:task=lift_ph_state" \
        --run "intent_seq:checkpoints/lift_ph_state_flow_mlp_512_seed0_intent_sequence_success98.pt:task=lift_ph_state_intent_sequence" \
        --n-rollouts 20 \
        --project intent-conditioning-comparison \
        --run-name viz-lift-2026-03-13 \
        --device cuda
"""

import argparse
import io
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch
import wandb

# ── Bootstrap: add repo root to path ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.intent_predictor import IntentPredictor
from mip.network_utils import get_network, get_encoder

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config(overrides: list[str]):
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/configs"), version_base=None
    ):
        cfg = compose("train_robomimic", overrides=overrides)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, config, device: str):
    """Load agent from checkpoint. Returns (agent, intent_predictor | None)."""
    # Build agent (same as train_robomimic.py)
    if getattr(config.task, "intent_conditioning", False):
        intent_dim = getattr(config.task, "intent_dim", 7)
        config.task.obs_dim = config.task.obs_dim + intent_dim

    network = get_network(config)
    encoder = get_encoder(config)
    agent = TrainingAgent(config, network, encoder)
    agent.load(checkpoint_path, load_optimizer=False)
    agent.eval()

    intent_predictor = None
    hl_path = Path(checkpoint_path).with_suffix("").with_name(
        Path(checkpoint_path).stem.split("_success")[0] + "_hl_policy.pt"
    )
    if hl_path.exists() and getattr(config.task, "use_learned_predictor", False):
        base_obs_dim = config.task.obs_dim - getattr(config.task, "intent_dim", 7)
        intent_predictor = IntentPredictor(
            obs_steps=config.task.obs_steps,
            base_obs_dim=base_obs_dim,
            intent_dim=getattr(config.task, "intent_dim", 7),
        ).to(device)
        intent_predictor.load_state_dict(
            torch.load(hl_path, map_location=device, weights_only=True)
        )
        intent_predictor.eval()

    return agent, intent_predictor


# ──────────────────────────────────────────────────────────────────────────────
# Rollout collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_rollouts(config, agent, dataset, envs, intent_predictor, n_rollouts: int, device: str):
    """Run n_rollouts and return trajectory data dict.

    Returns dict with keys:
        eef_pos:      list of (T, 3) arrays  — world-frame, unnormalized
        eef_quat:     list of (T, 4) arrays
        actions:      list of (T, act_dim) arrays — unnormalized
        intent_pred:  list of (T, intent_dim) arrays | None
        success:      list of bool
        ep_len:       list of int
    """
    from mip.envs.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

    data = defaultdict(list)
    intent_start = getattr(dataset, "intent_start", None)
    intent_end   = getattr(dataset, "intent_end",   None)
    has_intent   = getattr(config.task, "intent_conditioning", False)
    intent_type  = getattr(config.task, "intent_type", "mean")
    intent_dim   = getattr(config.task, "intent_dim", 7)
    intent_hz    = getattr(config.task, "intent_horizon", config.task.act_steps)

    n_done = 0
    while n_done < n_rollouts:
        obs, _ = envs.reset()
        ep_eef, ep_quat, ep_act, ep_intent = [], [], [], []
        success = False
        t = 0

        while t < config.task.max_episode_steps:
            obs_f = obs.astype(np.float32)
            obs_norm = dataset.normalizer["obs"]["state"].normalize(obs_f)
            obs_t = torch.tensor(obs_norm, device=device, dtype=torch.float32)

            # record raw eef (unnormalized) from the last obs step
            obs_unnorm = dataset.normalizer["obs"]["state"].unnormalize(obs_norm)
            eef_pos  = obs_unnorm[0, -1, intent_start:intent_start+3]
            eef_quat = obs_unnorm[0, -1, intent_start+3:intent_start+7] if intent_start is not None else np.zeros(4)
            ep_eef.append(eef_pos)
            ep_quat.append(eef_quat)

            # intent conditioning
            if has_intent and intent_start is not None:
                if intent_predictor is not None:
                    with torch.no_grad():
                        ip = intent_predictor(obs_t)  # (1, intent_dim)
                else:
                    eef_now  = obs_t[0, -1, intent_start:intent_end]
                    eef_prev = obs_t[0, -2, intent_start:intent_end]
                    vel = eef_now - eef_prev
                    if intent_type == "sequence":
                        ks = torch.arange(1, intent_hz+1, dtype=torch.float32, device=device)
                        ip = (eef_now + ks.unsqueeze(1) * vel.unsqueeze(0)).reshape(1, -1)
                    else:
                        ip = (eef_now + (intent_hz+1)/2.0 * vel).unsqueeze(0)

                ep_intent.append(ip[0].cpu().numpy())
                ip_expanded = ip.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
                obs_t = torch.cat([obs_t, ip_expanded], dim=-1)

            obs_in = {"state": obs_t}
            act_0 = torch.randn(
                (config.task.num_envs, config.task.horizon, config.task.act_dim), device=device
            )
            with torch.no_grad():
                act_norm = agent.sample(act_0=act_0, obs=obs_in, num_steps=1, use_ema=True)

            act_norm_np = act_norm.detach().cpu().numpy()
            act_unnorm  = dataset.normalizer["action"].unnormalize(act_norm_np)
            start = config.task.obs_steps - 1
            end   = start + config.task.act_steps
            act_chunk = act_unnorm[0, start:end, :]
            ep_act.append(act_chunk)

            # step envs
            obs, reward, terminated, truncated, info = envs.step(act_unnorm[:, start:end, :])
            t += config.task.act_steps

            if "is_success" in info and info["is_success"].any():
                success = True
                break

        data["eef_pos"].append(np.array(ep_eef))
        data["eef_quat"].append(np.array(ep_quat))
        data["actions"].append(np.concatenate(ep_act, axis=0) if ep_act else np.zeros((0, config.task.act_dim)))
        data["intent_pred"].append(np.array(ep_intent) if ep_intent else None)
        data["success"].append(success)
        data["ep_len"].append(t)

        n_done += config.task.num_envs  # vectorised: num_envs episodes per reset

    return data


# ──────────────────────────────────────────────────────────────────────────────
# 1. Trajectory visualization
# ──────────────────────────────────────────────────────────────────────────────

def fig_trajectories(all_data: dict) -> plt.Figure:
    """3D trajectories + 3 2D projections, per method, colored by success."""
    methods = list(all_data.keys())
    n = len(methods)
    fig = plt.figure(figsize=(5 * n, 14))
    gs  = gridspec.GridSpec(4, n, figure=fig, hspace=0.35, wspace=0.3)

    proj_pairs = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]

    for col, method in enumerate(methods):
        data = all_data[method]
        trajs   = data["eef_pos"]
        success = data["success"]

        # 3D
        ax3 = fig.add_subplot(gs[0, col], projection="3d")
        for traj, suc in zip(trajs, success):
            T = len(traj)
            if T < 2:
                continue
            c_map = plt.cm.coolwarm(np.linspace(0, 1, T))
            for i in range(T - 1):
                alpha = 0.6 if suc else 0.2
                ax3.plot(traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2],
                         color=c_map[i], alpha=alpha, lw=1)
        ax3.set_title(method, fontsize=9)
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
        ax3.tick_params(labelsize=6)

        # 2D projections
        for row, (xl, yl, xi, yi) in enumerate(proj_pairs, start=1):
            ax = fig.add_subplot(gs[row, col])
            for traj, suc in zip(trajs, success):
                color = "steelblue" if suc else "tomato"
                alpha = 0.5 if suc else 0.2
                ax.plot(traj[:, xi], traj[:, yi], color=color, alpha=alpha, lw=0.8)
            ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
            ax.set_title(f"{xl}-{yl}", fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("EEF Trajectories (blue=success, red=failure)", fontweight="bold")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. UMAP / PCA of trajectories
# ──────────────────────────────────────────────────────────────────────────────

def fig_embedding(all_data: dict) -> plt.Figure:
    """UMAP (or PCA fallback) of flattened eef trajectories, colored by method and success."""
    method_colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

    # Pad/truncate all trajectories to same length
    max_T = max(max(len(t) for t in d["eef_pos"]) for d in all_data.values())
    rows, labels_method, labels_success = [], [], []
    for mi, (method, data) in enumerate(all_data.items()):
        for traj, suc in zip(data["eef_pos"], data["success"]):
            T = len(traj)
            if T == 0:
                continue
            padded = np.zeros((max_T, 3))
            padded[:T] = traj
            rows.append(padded.flatten())
            labels_method.append(mi)
            labels_success.append(int(suc))

    X = np.array(rows)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        method_name = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        emb = reducer.fit_transform(X)
        method_name = "PCA"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    method_list = list(all_data.keys())

    # Color by method
    for mi, method in enumerate(method_list):
        mask = np.array(labels_method) == mi
        axes[0].scatter(emb[mask, 0], emb[mask, 1],
                        color=method_colors[mi], label=method, alpha=0.6, s=20)
    axes[0].set_title(f"{method_name} — colored by method")
    axes[0].legend(fontsize=7)

    # Color by success
    suc_arr = np.array(labels_success)
    axes[1].scatter(emb[suc_arr == 0, 0], emb[suc_arr == 0, 1],
                    color="tomato", alpha=0.5, s=20, label="failure")
    axes[1].scatter(emb[suc_arr == 1, 0], emb[suc_arr == 1, 1],
                    color="steelblue", alpha=0.5, s=20, label="success")
    axes[1].set_title(f"{method_name} — colored by success")
    axes[1].legend(fontsize=7)

    fig.suptitle("Trajectory Embedding", fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Intent prediction accuracy (retrospective GT)
# ──────────────────────────────────────────────────────────────────────────────

def fig_intent_accuracy(all_data: dict, intent_horizon: int) -> plt.Figure:
    """Compare predicted intent vs retrospective ground truth (actual future eef mean)."""
    intent_methods = {k: v for k, v in all_data.items() if v["intent_pred"][0] is not None}
    if not intent_methods:
        return None

    n = len(intent_methods)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * max(n, 1), 5))
    if n == 1:
        axes = [axes]

    for ax, (method, data) in zip(axes, intent_methods.items()):
        errors = []
        for traj, intent_preds in zip(data["eef_pos"], data["intent_pred"]):
            if intent_preds is None or len(traj) < 2:
                continue
            T = len(traj)
            for t in range(T):
                fut_end = min(t + intent_horizon, T)
                gt_mean = traj[t:fut_end, :3].mean(axis=0)  # GT future mean eef pos (xyz only)
                pred_pos = intent_preds[t, :3]               # predicted (xyz)
                errors.append(np.linalg.norm(pred_pos - gt_mean))

        if errors:
            ax.hist(errors, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(errors), color="tomato", lw=2, label=f"mean={np.mean(errors):.3f}")
            ax.set_title(method)
            ax.set_xlabel("L2 error (predicted vs GT future mean eef)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    fig.suptitle("Intent Prediction Error (xyz only)", fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Action distribution
# ──────────────────────────────────────────────────────────────────────────────

def fig_action_distribution(all_data: dict) -> plt.Figure:
    """Action variance per dim and PCA of action chunks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-dim variance
    method_colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    act_dim_labels = ["dx", "dy", "dz", "r0", "r1", "r2", "r3", "r4", "r5", "grip"]
    x = np.arange(len(act_dim_labels))
    width = 0.8 / max(len(all_data), 1)

    for mi, (method, data) in enumerate(all_data.items()):
        all_acts = np.concatenate([a for a in data["actions"] if len(a) > 0], axis=0)
        variances = all_acts.var(axis=0)
        axes[0].bar(x + mi * width, variances, width, label=method,
                    color=method_colors[mi], alpha=0.8)

    axes[0].set_xticks(x + width * len(all_data) / 2)
    axes[0].set_xticklabels(act_dim_labels, rotation=30, fontsize=8)
    axes[0].set_ylabel("Variance across rollouts")
    axes[0].set_title("Action variance per dimension")
    axes[0].legend(fontsize=7)

    # PCA of action chunks
    from sklearn.decomposition import PCA
    all_chunks, chunk_labels = [], []
    method_list = list(all_data.keys())
    for mi, (method, data) in enumerate(all_data.items()):
        for ep_acts in data["actions"]:
            if len(ep_acts) == 0:
                continue
            all_chunks.append(ep_acts.flatten())
            chunk_labels.append(mi)

    if all_chunks:
        X = np.array(all_chunks)
        pca = PCA(n_components=2)
        emb = pca.fit_transform(X)
        for mi, method in enumerate(method_list):
            mask = np.array(chunk_labels) == mi
            axes[1].scatter(emb[mask, 0], emb[mask, 1],
                            color=method_colors[mi], label=method, alpha=0.5, s=15)
        axes[1].set_title(f"PCA of action chunks (var explained: "
                          f"{pca.explained_variance_ratio_.sum():.0%})")
        axes[1].legend(fontsize=7)

    fig.suptitle("Action Distribution", fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. Sensitivity / steerability
# ──────────────────────────────────────────────────────────────────────────────

def fig_sensitivity(all_data: dict, all_models: dict, all_configs: dict,
                    all_datasets: dict, device: str) -> plt.Figure:
    """How much does perturbing the intent vector change the output action?

    For each intent-conditioned method: take 5 rollout steps, perturb intent ±0.1
    per dimension, measure mean L2 change in the action chunk.
    """
    intent_methods = {k for k in all_data if all_data[k]["intent_pred"][0] is not None}
    if not intent_methods:
        return None

    fig, axes = plt.subplots(1, len(intent_methods), figsize=(6 * len(intent_methods), 5))
    if len(intent_methods) == 1:
        axes = [axes]

    delta = 0.1
    n_samples = 10  # number of rollout steps to probe

    for ax, method in zip(axes, intent_methods):
        agent, ip = all_models[method]
        config   = all_configs[method]
        dataset  = all_datasets[method]
        data     = all_data[method]

        intent_dim = getattr(config.task, "intent_dim", 7)
        sensitivities = np.zeros(intent_dim)

        for ep_idx in range(min(3, len(data["eef_pos"]))):
            traj    = data["eef_pos"][ep_idx]
            ip_preds = data["intent_pred"][ep_idx]
            if ip_preds is None or len(traj) < n_samples:
                continue

            # Use obs from dataset (normalized, same as during rollout)
            obs_norm = dataset.normalizer["obs"]["state"].normalize(
                traj[np.newaxis, :config.task.obs_steps, :]
            )
            obs_t = torch.tensor(obs_norm, device=device, dtype=torch.float32)

            for step_idx in range(min(n_samples, len(ip_preds))):
                base_intent = torch.tensor(
                    ip_preds[step_idx], device=device, dtype=torch.float32
                ).unsqueeze(0)  # (1, intent_dim)

                # Base action
                base_intent_expanded = base_intent.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
                base_obs = torch.cat([obs_t, base_intent_expanded], dim=-1)
                act_0 = torch.randn(1, config.task.horizon, config.task.act_dim, device=device)
                with torch.no_grad():
                    base_act = agent.sample(act_0=act_0.clone(), obs={"state": base_obs},
                                            num_steps=1, use_ema=True)

                # Perturb each intent dimension
                for dim in range(intent_dim):
                    pert = base_intent.clone()
                    pert[0, dim] += delta
                    pert_exp = pert.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
                    pert_obs = torch.cat([obs_t, pert_exp], dim=-1)
                    with torch.no_grad():
                        pert_act = agent.sample(act_0=act_0.clone(), obs={"state": pert_obs},
                                                num_steps=1, use_ema=True)
                    sensitivities[dim] += (
                        (pert_act - base_act).abs().mean().item() / delta
                    )

        sensitivities /= max(1, 3 * n_samples)
        intent_labels = [f"eef_{i}" for i in range(intent_dim)]
        ax.bar(range(intent_dim), sensitivities, color="steelblue", alpha=0.8)
        ax.set_xticks(range(intent_dim))
        ax.set_xticklabels(intent_labels[:intent_dim], rotation=30, fontsize=7)
        ax.set_title(method)
        ax.set_ylabel("∂action / ∂intent (mean abs sensitivity)")

    fig.suptitle("Intent Sensitivity Analysis — How much does intent change actions?",
                 fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. Success / failure breakdown
# ──────────────────────────────────────────────────────────────────────────────

def fig_success_analysis(all_data: dict) -> plt.Figure:
    """Success rate bars + episode length distribution by method."""
    methods = list(all_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    # Success rate
    rates = [np.mean(all_data[m]["success"]) for m in methods]
    bars = axes[0].bar(methods, rates, color=colors, alpha=0.85, edgecolor="white")
    for bar, rate in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{rate:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Success rate by method")
    axes[0].tick_params(axis="x", rotation=20)

    # Episode length histogram
    for mi, method in enumerate(methods):
        ep_lens = all_data[method]["ep_len"]
        axes[1].hist(ep_lens, bins=20, alpha=0.5, label=method,
                     color=colors[mi], edgecolor="white")
    axes[1].set_xlabel("Episode length (steps)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Episode length distribution")
    axes[1].legend(fontsize=7)

    fig.suptitle("Success / Failure Analysis", fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. Training dynamics (from log files)
# ──────────────────────────────────────────────────────────────────────────────

def fig_training_dynamics(log_paths: dict) -> plt.Figure:
    """Loss curves and eval success rate over training steps."""
    import re

    def parse(path):
        steps, losses, evals, intent_mse = [], [], [], []
        if not path or not os.path.exists(path):
            return steps, losses, evals, intent_mse
        with open(path) as f:
            for line in f:
                m = re.search(r"\[Step (\d+)\].*? loss: ([\d.e+\-]+)", line)
                if m and "mean_success" not in line:
                    steps.append(int(m.group(1)) + 1)
                    losses.append(float(m.group(2)))
                m2 = re.search(r"\[Step (\d+)\].*?mean_success_9: ([\d.e+\-]+)", line)
                if m2:
                    evals.append((int(m2.group(1)) + 1, float(m2.group(2))))
                m3 = re.search(r"intent_pred_mse: ([\d.e+\-]+)", line)
                if m3:
                    intent_mse.append((steps[-1] if steps else 0, float(m3.group(1))))
        return steps, losses, evals, intent_mse

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_paths)))

    for ci, (method, path) in enumerate(log_paths.items()):
        steps, losses, evals, intent_mse = parse(path)
        col = colors[ci]

        if steps:
            # Smooth loss
            w = min(50, len(losses) // 5 + 1)
            smooth = np.convolve(losses, np.ones(w)/w, mode="valid")
            axes[0].plot(steps[w-1:], smooth, color=col, lw=1.5, alpha=0.9, label=method)

        if evals:
            ex, ey = zip(*evals)
            axes[1].plot(ex, ey, "o-", color=col, lw=2, ms=6, label=method)

        if intent_mse:
            mx, my = zip(*intent_mse)
            axes[2].plot(mx, my, color=col, lw=1.5, alpha=0.9, label=method)

    axes[0].set_title("Training loss (smoothed)"); axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Flow matching loss"); axes[0].legend(fontsize=7); axes[0].set_yscale("log")
    axes[1].set_title("Eval success rate"); axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Success rate"); axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=7)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[2].set_title("Intent predictor MSE loss"); axes[2].set_xlabel("Step")
    axes[2].set_ylabel("MSE"); axes[2].legend(fontsize=7)

    fig.suptitle("Training Dynamics", fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Wandb logging helper
# ──────────────────────────────────────────────────────────────────────────────

def fig_to_wandb_image(fig):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return wandb.Image(Image.open(buf))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[],
                        metavar="LABEL:CKPT:task=<task>[,key=val,...]",
                        help="e.g. 'baseline:checkpoints/xyz.pt:task=lift_ph_state'")
    parser.add_argument("--log", action="append", default=[],
                        metavar="LABEL:path/to/run.err",
                        help="Log file for training dynamics (optional, matched to --run label)")
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--project", default="intent-conditioning-comparison")
    parser.add_argument("--run-name", default="viz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
        return

    # ── Parse run specs ────────────────────────────────────────────────────────
    all_data, all_models, all_configs, all_datasets = {}, {}, {}, {}
    log_paths = {lbl: path for entry in args.log for lbl, path in [entry.split(":", 1)]}

    for entry in args.run:
        parts = entry.split(":", 2)
        label, ckpt = parts[0], parts[1]
        overrides = parts[2].split(",") if len(parts) > 2 else []

        print(f"\n[{label}] Loading config: {overrides}")
        config = load_config(overrides)

        # obs_dim is set at runtime in train_robomimic from env; replicate that here
        # (intent conditioning will be added inside load_model)
        print(f"[{label}] Loading model from {ckpt}")
        agent, intent_predictor = load_model(ckpt, config, args.device)

        print(f"[{label}] Creating env + dataset")
        from mip.envs.robomimic_lowdim_wrapper import make_vec_env
        envs = make_vec_env(config, num_envs=1)
        obs, _ = envs.reset()
        config.task.obs_dim = obs.shape[-1]  # actual obs_dim from env

        dataset, normalizer = make_dataset(config.task)

        all_models[label]  = (agent, intent_predictor)
        all_configs[label] = config
        all_datasets[label] = dataset

        print(f"[{label}] Collecting {args.n_rollouts} rollouts...")
        data = collect_rollouts(
            config, agent, dataset, envs, intent_predictor,
            args.n_rollouts, args.device
        )
        sr = np.mean(data["success"])
        print(f"[{label}] Done. Success rate: {sr:.0%}")
        all_data[label] = data
        envs.close()

    # ── Generate figures ────────────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    figures = {}

    figures["trajectories"]     = fig_trajectories(all_data)
    figures["embedding"]        = fig_embedding(all_data)
    figures["action_dist"]      = fig_action_distribution(all_data)
    figures["success_analysis"] = fig_success_analysis(all_data)

    intent_horizon = max(
        getattr(c.task, "intent_horizon", 8) for c in all_configs.values()
    )
    f_intent = fig_intent_accuracy(all_data, intent_horizon)
    if f_intent:
        figures["intent_accuracy"] = f_intent

    f_sens = fig_sensitivity(all_data, all_models, all_configs, all_datasets, args.device)
    if f_sens:
        figures["sensitivity"] = f_sens

    if log_paths:
        figures["training_dynamics"] = fig_training_dynamics(log_paths)

    # ── Log to wandb ────────────────────────────────────────────────────────────
    print("\nLogging to wandb...")
    run = wandb.init(project=args.project, name=args.run_name, job_type="visualization")

    log_dict = {f"viz/{name}": fig_to_wandb_image(fig) for name, fig in figures.items()}

    # Also log scalar success rates
    for label, data in all_data.items():
        log_dict[f"eval/success_rate/{label}"] = float(np.mean(data["success"]))
        log_dict[f"eval/mean_ep_len/{label}"]  = float(np.mean(data["ep_len"]))

    wandb.log(log_dict)
    run.finish()
    print(f"\nDone. View at: {run.url}")


if __name__ == "__main__":
    main()
