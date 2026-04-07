"""Visualize how PushT intent changes when the block pose is perturbed.

This script compares paired intent samples from a PushT flow_intent checkpoint:
1. Original observation / block pose
2. The same observation with only the T-block pose shifted

For each panel, we render the scene and overlay:
- the current agent position
- sampled future agent-position intent targets
- the mean future agent-position target arrow (agent_pos -> mean target)

The comparison uses the same latent noise seeds in both conditions so that
differences are attributable to the changed block pose rather than sampling
noise.

Usage:
    python examples/plot_pusht_intent_shift.py \
        --run "flow_intent:/path/to/ckpt.pt:task=pusht_state_flow_intent:+network.arch_variant=flow_intent" \
        --env-seed 0 \
        --perturb-dx 40 \
        --perturb-dy 0 \
        --perturb-dtheta 0.0 \
        --n-intent-samples 32 \
        --out plots/pusht_intent_shift.png
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

from collect_diversity_rollouts import (  # noqa: E402
    load_config,
    parse_run_spec,
    setup_config_for_env,
    load_model,
    preprocess_obs,
)
from mip.datasets.pusht_dataset import make_dataset as make_dataset_pusht  # noqa: E402
from mip.envs.pusht import make_vec_env as make_vec_env_pusht  # noqa: E402
from mip.flow_intent_agent import FlowIntentAgent  # noqa: E402


WINDOW_SIZE = 512.0


def _get_pusht_base(envs):
    from mip.envs.pusht.pusht_env import PushTEnv

    env = envs.envs[0]
    while env is not None:
        if isinstance(env, PushTEnv):
            return env
        env = getattr(env, "env", None)
    raise RuntimeError("Could not find PushTEnv in wrapper chain")


def _capture_state(base) -> dict[str, np.ndarray | float]:
    return {
        "agent_pos": np.array(base.agent.position, dtype=np.float64).copy(),
        "agent_vel": np.array(base.agent.velocity, dtype=np.float64).copy(),
        "block_pos": np.array(base.block.position, dtype=np.float64).copy(),
        "block_angle": float(base.block.angle),
        "block_vel": np.array(base.block.velocity, dtype=np.float64).copy(),
        "block_angvel": float(base.block.angular_velocity),
    }


def _restore_state(base, state: dict[str, np.ndarray | float], obs_steps: int) -> np.ndarray:
    pos_state = np.array(
        list(np.asarray(state["agent_pos"]).reshape(-1))
        + list(np.asarray(state["block_pos"]).reshape(-1))
        + [float(state["block_angle"])],
        dtype=np.float64,
    )
    try:
        base.reset()
    except Exception:
        pass
    base._set_state(pos_state)
    base.agent.velocity = tuple(np.asarray(state["agent_vel"], dtype=np.float64).reshape(-1)[:2].tolist())
    base.block.velocity = tuple(np.asarray(state["block_vel"], dtype=np.float64).reshape(-1)[:2].tolist())
    base.block.angular_velocity = float(state["block_angvel"])
    base.latest_action = None
    obs = base._get_obs()
    return np.stack([obs] * obs_steps, axis=0)[np.newaxis]


def _shift_block_state(
    state: dict[str, np.ndarray | float],
    dx: float,
    dy: float,
    dtheta: float,
) -> dict[str, np.ndarray | float]:
    shifted = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in state.items()
    }
    shifted["block_pos"] = shifted["block_pos"] + np.array([dx, dy], dtype=np.float64)
    shifted["block_pos"] = np.clip(shifted["block_pos"], 5.0, WINDOW_SIZE - 5.0)
    shifted["block_angle"] = float(((shifted["block_angle"] + dtheta + np.pi) % (2 * np.pi)) - np.pi)
    return shifted


def _unnormalize_intents(intents_norm: np.ndarray, dataset) -> np.ndarray:
    state_norm = dataset.normalizer["obs"]["state"]
    idx = np.asarray(dataset.intent_indices, dtype=np.int64)
    mins = state_norm.min[idx]
    ranges = state_norm.range[idx]
    return ((intents_norm.astype(np.float32) + 1.0) / 2.0) * ranges + mins


def _sample_paired_intents(
    obs_original: np.ndarray,
    obs_shifted: np.ndarray,
    agent: FlowIntentAgent,
    config,
    dataset,
    device: str,
    n_samples: int,
    num_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_orig_t, _ = preprocess_obs(obs_original, config, dataset, device)
    obs_shift_t, _ = preprocess_obs(obs_shifted, config, dataset, device)

    gen = np.random.default_rng(seed)
    pair_seeds = gen.integers(0, 2**31 - 1, size=n_samples, dtype=np.int64)

    orig_intents = []
    shift_intents = []
    with torch.no_grad():
        for pair_seed in pair_seeds:
            torch.manual_seed(int(pair_seed))
            _, intent_orig = agent.sample(
                obs=obs_orig_t,
                use_ema=True,
                num_steps=num_steps,
                return_intent=True,
            )
            torch.manual_seed(int(pair_seed))
            _, intent_shift = agent.sample(
                obs=obs_shift_t,
                use_ema=True,
                num_steps=num_steps,
                return_intent=True,
            )
            orig_intents.append(intent_orig[0].cpu().numpy())
            shift_intents.append(intent_shift[0].cpu().numpy())

    return np.stack(orig_intents), np.stack(shift_intents)


def _rollout_to_current_obs(envs, agent, config, dataset, device: str, rollout_steps: int, seed: int):
    obs, _ = envs.reset()
    if rollout_steps <= 0:
        return obs

    torch.manual_seed(seed)
    t = 0
    while t < rollout_steps:
        obs_t, _ = preprocess_obs(obs, config, dataset, device)
        with torch.no_grad():
            act_norm = agent.sample(obs=obs_t, use_ema=True)
        act = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
        start = config.task.obs_steps - 1
        end = start + config.task.act_steps
        act = act[:, start:end, :]
        obs, _, terminated, truncated, _ = envs.step(act)
        if bool(terminated[0] or truncated[0]):
            break
        t += config.task.act_steps
    return obs


def _plot_panel(ax, image, state, intents_raw, title: str, color: str):
    agent_pos = np.asarray(state["agent_pos"], dtype=np.float32)
    block_pos = np.asarray(state["block_pos"], dtype=np.float32)
    mean_intent = intents_raw.mean(axis=0)

    ax.imshow(image, extent=[0, WINDOW_SIZE, WINDOW_SIZE, 0])
    ax.scatter(
        intents_raw[:, 0],
        intents_raw[:, 1],
        s=24,
        color=color,
        alpha=0.28,
        edgecolors="none",
        label="sampled future agent targets",
    )
    ax.scatter(
        [agent_pos[0]],
        [agent_pos[1]],
        s=80,
        color="royalblue",
        edgecolors="white",
        linewidths=1.0,
        label="agent",
        zorder=4,
    )
    ax.scatter(
        [block_pos[0]],
        [block_pos[1]],
        s=110,
        marker="X",
        color="gold",
        edgecolors="black",
        linewidths=1.0,
        label="block center",
        zorder=4,
    )
    ax.scatter(
        [mean_intent[0]],
        [mean_intent[1]],
        s=90,
        color=color,
        edgecolors="black",
        linewidths=0.8,
        label="mean future agent target",
        zorder=5,
    )
    ax.arrow(
        float(agent_pos[0]),
        float(agent_pos[1]),
        float(mean_intent[0] - agent_pos[0]),
        float(mean_intent[1] - agent_pos[1]),
        color=color,
        width=2.8,
        head_width=18.0,
        head_length=18.0,
        length_includes_head=True,
        alpha=0.95,
        zorder=4,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, WINDOW_SIZE)
    ax.set_ylim(WINDOW_SIZE, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="Run spec: label:ckpt_path:override1:override2:...",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--perturb-dx", type=float, default=40.0)
    parser.add_argument("--perturb-dy", type=float, default=0.0)
    parser.add_argument("--perturb-dtheta", type=float, default=0.0)
    parser.add_argument("--n-intent-samples", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=9)
    parser.add_argument(
        "--out",
        default="plots/pusht_intent_shift.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--out-data",
        default="plots/pusht_intent_shift.pkl",
        help="Output pickle with raw states and intent samples",
    )
    args = parser.parse_args()

    label, checkpoint_path, overrides = parse_run_spec(args.run)
    config = load_config(overrides)
    config.task.num_envs = 1
    config.task.save_video = False

    if getattr(config.task, "env_name", "") != "pusht":
        raise ValueError("This script only supports PushT")
    if getattr(config.network, "arch_variant", "flow_action") != "flow_intent":
        raise ValueError("This experiment expects a flow_intent checkpoint/config")

    envs = make_vec_env_pusht(config.task, seed=args.env_seed)
    dataset = make_dataset_pusht(config.task)
    _ = setup_config_for_env(config, envs)
    agent, _ = load_model(checkpoint_path, config, dataset, args.device)
    if not isinstance(agent, FlowIntentAgent):
        raise TypeError(f"Expected FlowIntentAgent, got {type(agent).__name__}")

    obs_original = _rollout_to_current_obs(
        envs=envs,
        agent=agent,
        config=config,
        dataset=dataset,
        device=args.device,
        rollout_steps=args.rollout_steps,
        seed=args.env_seed,
    )

    base = _get_pusht_base(envs)
    original_state = _capture_state(base)
    shifted_state = _shift_block_state(
        original_state,
        dx=args.perturb_dx,
        dy=args.perturb_dy,
        dtheta=args.perturb_dtheta,
    )

    obs_original = _restore_state(base, original_state, config.task.obs_steps)
    image_original = np.array(base.render(mode="rgb_array"), copy=True)
    obs_shifted = _restore_state(base, shifted_state, config.task.obs_steps)
    image_shifted = np.array(base.render(mode="rgb_array"), copy=True)
    _ = _restore_state(base, original_state, config.task.obs_steps)

    intents_orig_norm, intents_shift_norm = _sample_paired_intents(
        obs_original=obs_original,
        obs_shifted=obs_shifted,
        agent=agent,
        config=config,
        dataset=dataset,
        device=args.device,
        n_samples=args.n_intent_samples,
        num_steps=args.num_steps,
        seed=args.env_seed,
    )
    intents_orig_raw = _unnormalize_intents(intents_orig_norm, dataset)
    intents_shift_raw = _unnormalize_intents(intents_shift_norm, dataset)

    mean_delta = intents_shift_raw.mean(axis=0) - intents_orig_raw.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "PushT Future Agent-Target Shift Under Object Perturbation\n"
        f"{label} | samples={args.n_intent_samples} | "
        f"block shift=({args.perturb_dx:.1f}, {args.perturb_dy:.1f}, {args.perturb_dtheta:.2f} rad) | "
        f"mean target delta=({mean_delta[0]:.1f}, {mean_delta[1]:.1f})",
        fontsize=13,
        fontweight="bold",
    )

    _plot_panel(
        axes[0],
        image_original,
        original_state,
        intents_orig_raw,
        title="Original block pose",
        color="crimson",
    )
    _plot_panel(
        axes[1],
        image_shifted,
        shifted_state,
        intents_shift_raw,
        title="Perturbed block pose",
        color="darkorange",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    out_data = Path(args.out_data)
    out_data.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_overrides": overrides,
        "checkpoint_path": checkpoint_path,
        "env_seed": args.env_seed,
        "rollout_steps": args.rollout_steps,
        "perturb": {
            "dx": args.perturb_dx,
            "dy": args.perturb_dy,
            "dtheta": args.perturb_dtheta,
        },
        "original_state": original_state,
        "shifted_state": shifted_state,
        "intents_original_norm": intents_orig_norm,
        "intents_shifted_norm": intents_shift_norm,
        "intents_original_raw": intents_orig_raw,
        "intents_shifted_raw": intents_shift_raw,
        "mean_intent_delta_raw": mean_delta,
    }
    with out_data.open("wb") as f:
        pickle.dump(payload, f)

    envs.close()
    print(f"Saved figure to {out_path}")
    print(f"Saved raw data to {out_data}")


if __name__ == "__main__":
    main()
