"""Standardized multimodality visualization: same-state, multi-sample evaluation.

Collects data for visualizing policy multimodality by:
1. Running full rollouts to find "critical states" (high variance across repeated samples)
2. At each critical state, sampling N action chunks from each variant
3. Saving all data to a .pkl for downstream plotting

The key idea: freeze the observation at specific states, then draw many action
samples from each policy variant. This reveals whether the policy is multimodal
(produces diverse actions from the same state) and how intent conditioning
changes the action distribution.

Usage:
    python examples/collect_multimodal_viz.py \
        --task lift_ph_state \
        --run "baseline:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_success0.pt:task=lift_ph_state" \
        --run "flow_intent:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt:task=lift_ph_state_flow_intent:+network.arch_variant=flow_intent" \
        --run "hierarchical_emb:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_intent_learned_joint_emb_success98.pt:task=lift_ph_state_hierarchical_emb" \
        --n-rollouts 20 \
        --n-samples 50 \
        --n-critical 10 \
        --device cuda \
        --out rollouts/lift_ph_state_multimodal.pkl

Output .pkl structure:
    {
      "<task>": {
        "critical_states": {
            "obs_raw":       list of np.ndarray — raw obs at each critical state
            "eef_pos":       list of (3,) arrays — eef position at critical state
            "timestep":      list of int — timestep in the source episode
            "episode_idx":   list of int — which episode the state came from
            "variance_score": list of float — how multimodal this state is
        },
        "<variant_label>": {
            "rollout_chunks":    np.ndarray (N_total, flat_dim) — all rollout action chunks
            "rollout_success":   list[bool]
            "critical_samples":  np.ndarray (n_critical, n_samples, act_steps, act_dim)
            "critical_intents":  np.ndarray (n_critical, n_samples, intent_dim) or None
            "steer_actions":     np.ndarray (n_eps, K, flat_dim) or None  (flow_intent only)
            "steer_intents":     np.ndarray (n_eps, K, intent_dim) or None
        },
        "gt_demos": {"action_chunks": np.ndarray (M, flat_dim)},
      }
    }
"""

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from mip.agent import TrainingAgent
from mip.flow_intent_agent import FlowIntentAgent
from mip.intent_predictor import IntentPredictor
from mip.datasets.robomimic_dataset import make_dataset as make_dataset_robomimic
from mip.datasets.pusht_dataset import make_dataset as make_dataset_pusht
from mip.datasets.kitchen_dataset import make_dataset as make_dataset_kitchen
from mip.datasets.libero_dataset import make_dataset as make_dataset_libero
from mip.envs.robomimic.robomimic_env import make_vec_env as make_vec_env_robomimic
from mip.envs.pusht import make_vec_env as make_vec_env_pusht
from mip.envs.kitchen import make_vec_env as make_vec_env_kitchen
from mip.envs.libero import make_vec_env as make_vec_env_libero

def make_vec_env(task_config, seed=0):
    if getattr(task_config, "env_name", "").startswith("libero"):
        return make_vec_env_libero(task_config, seed=seed)
    if task_config.env_name == "pusht":
        return make_vec_env_pusht(task_config)
    if "kitchen" in task_config.env_name:
        return make_vec_env_kitchen(task_config, seed=seed)
    return make_vec_env_robomimic(task_config, seed=seed)

def make_dataset(task_config):
    if getattr(task_config, "env_name", "").startswith("libero"):
        return make_dataset_libero(task_config)
    if task_config.env_name == "pusht":
        return make_dataset_pusht(task_config)
    if "kitchen" in task_config.env_name:
        return make_dataset_kitchen(task_config)
    return make_dataset_robomimic(task_config)
from mip.torch_utils import set_seed

# Reuse setup helpers from collect_diversity_rollouts
from collect_diversity_rollouts import (
    load_config,
    parse_run_spec,
    setup_config_for_env,
    load_model,
    preprocess_obs,
    collect_gt_demos,
    STEER_K,
)


# ──────────────────────────────────────────────────────────────────────────────
# Critical state discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_critical_states(config, agent, dataset, envs, device,
                         n_rollouts=10, n_probe_samples=10,
                         is_flow_intent=False, intent_predictor=None,
                         num_steps=9):
    """Run rollouts and at each timestep sample n_probe actions to estimate variance.

    Returns list of dicts with keys: obs_raw, eef_pos, timestep, episode_idx, variance_score.
    Sorted by variance_score descending.
    """
    s = config.task.obs_steps - 1
    states = []
    n_done = 0

    while n_done < n_rollouts:
        obs, _ = envs.reset()
        ep_reward = np.zeros(config.task.num_envs)
        t = 0

        while t < config.task.max_episode_steps:
            # Preprocess observation
            obs_in, lowdim_t = preprocess_obs(
                obs, config, dataset, device, intent_predictor=intent_predictor
            )

            # Sample n_probe actions to estimate variance
            actions_list = []
            with torch.no_grad():
                for _ in range(n_probe_samples):
                    if is_flow_intent:
                        if config.task.obs_type == "image":
                            B_fi = next(iter(obs_in.values())).shape[0]
                            fi_obs = TensorDict(obs_in, batch_size=B_fi)
                        else:
                            fi_obs = lowdim_t
                        act_normed = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
                    else:
                        act_0 = torch.randn(
                            (config.task.num_envs, config.task.horizon, config.task.act_dim),
                            device=device,
                        )
                        sample_obs = (
                            {"state": obs_in} if config.task.obs_type == "state" else obs_in
                        )
                        act_normed = agent.sample(act_0=act_0, obs=sample_obs,
                                                   num_steps=num_steps, use_ema=True,
                                                   sample_mode="stochastic")

                    act_un = dataset.normalizer["action"].unnormalize(act_normed.cpu().numpy())
                    act_chunk = act_un[0, s:s + config.task.act_steps]  # (act_steps, act_dim)
                    actions_list.append(act_chunk.flatten())

            # Compute variance across probe samples
            probe_actions = np.stack(actions_list)  # (n_probe, flat_dim)
            variance_score = float(probe_actions.var(axis=0).mean())

            # Store state info
            if config.task.obs_type == "state":
                obs_raw_snapshot = obs.copy()
            else:
                obs_raw_snapshot = {k: v.copy() for k, v in obs.items()}

            # Get eef position from unnormalized obs
            if config.task.obs_type == "state":
                obs_f = obs.astype(np.float32)
            elif isinstance(obs_raw_snapshot, dict) and "state" in obs_raw_snapshot:
                obs_f = obs_raw_snapshot["state"].astype(np.float32)
            else:
                obs_f = None
            eef_pos = np.zeros(3)
            if obs_f is not None:
                intent_start = getattr(dataset, "intent_start", None)
                if intent_start is not None:
                    eef_pos = obs_f[0, -1, intent_start:intent_start + 3]

            states.append({
                "obs_raw": obs_raw_snapshot,
                "eef_pos": eef_pos,
                "timestep": t,
                "episode_idx": n_done,
                "variance_score": variance_score,
            })

            # Step environment with one sample
            with torch.no_grad():
                if is_flow_intent:
                    if config.task.obs_type == "image":
                        B_fi = next(iter(obs_in.values())).shape[0]
                        fi_obs = TensorDict(obs_in, batch_size=B_fi)
                    else:
                        fi_obs = lowdim_t
                    act_normed = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
                else:
                    act_0 = torch.randn(
                        (config.task.num_envs, config.task.horizon, config.task.act_dim),
                        device=device,
                    )
                    sample_obs = (
                        {"state": obs_in} if config.task.obs_type == "state" else obs_in
                    )
                    act_normed = agent.sample(act_0=act_0, obs=sample_obs,
                                               num_steps=num_steps, use_ema=True)

            act_un = dataset.normalizer["action"].unnormalize(act_normed.cpu().numpy())
            act_chunk = act_un[:, s:s + config.task.act_steps]

            # Undo rotation for env
            act_env = act_chunk
            _ABS_ACTION_ENVS = {"can", "lift", "square", "tool_hang", "transport"}
            if getattr(config.task, "abs_action", False) and config.task.env_name in _ABS_ACTION_ENVS:
                act_env = dataset.undo_transform_action(act_chunk)

            obs, reward, terminated, truncated, info = envs.step(act_env)
            ep_reward += reward
            t += config.task.act_steps

        n_done += config.task.num_envs

    # Sort by variance score (highest = most multimodal)
    states.sort(key=lambda x: x["variance_score"], reverse=True)
    return states


def sample_at_state(obs_raw, config, agent, dataset, device, n_samples,
                    is_flow_intent=False, intent_predictor=None, num_steps=9):
    """Given a frozen observation, sample n_samples action chunks.

    Returns:
        actions: np.ndarray (n_samples, act_steps, act_dim) — unnormalized
        intents: np.ndarray (n_samples, intent_dim) or None
    """
    s = config.task.obs_steps - 1
    actions_list = []
    intents_list = []

    obs_in, lowdim_t = preprocess_obs(
        obs_raw, config, dataset, device, intent_predictor=intent_predictor
    )

    with torch.no_grad():
        for _ in range(n_samples):
            if is_flow_intent:
                if config.task.obs_type == "image":
                    B_fi = next(iter(obs_in.values())).shape[0]
                    fi_obs = TensorDict(obs_in, batch_size=B_fi)
                else:
                    fi_obs = lowdim_t

                result = agent.sample(
                    obs=fi_obs, use_ema=True, num_steps=num_steps,
                    return_intent=True,
                )
                act_normed, intent_vec = result
                intents_list.append(intent_vec[0].cpu().numpy())
            else:
                act_0 = torch.randn(
                    (config.task.num_envs, config.task.horizon, config.task.act_dim),
                    device=device,
                )
                sample_obs = (
                    {"state": obs_in} if config.task.obs_type == "state" else obs_in
                )
                act_normed = agent.sample(act_0=act_0, obs=sample_obs,
                                           num_steps=num_steps, use_ema=True,
                                           sample_mode="stochastic")

            act_un = dataset.normalizer["action"].unnormalize(act_normed.cpu().numpy())
            act_chunk = act_un[0, s:s + config.task.act_steps]  # (act_steps, act_dim)
            actions_list.append(act_chunk)

    actions = np.stack(actions_list)  # (n_samples, act_steps, act_dim)
    intents = np.stack(intents_list) if intents_list else None
    return actions, intents


# ──────────────────────────────────────────────────────────────────────────────
# Standard rollout collection (reused from collect_diversity_rollouts)
# ──────────────────────────────────────────────────────────────────────────────

def collect_rollouts_simple(config, agent, dataset, envs, n_rollouts, device,
                            is_flow_intent=False, intent_predictor=None, num_steps=9):
    """Simplified rollout collection returning action chunks + success flags."""
    from collect_diversity_rollouts import collect_rollouts
    chunks, successes, steer_acts, steer_ints = collect_rollouts(
        config, agent, dataset, envs,
        n_rollouts=n_rollouts,
        device=device,
        is_flow_intent=is_flow_intent,
        intent_predictor=intent_predictor,
        num_steps=num_steps,
    )
    return chunks, successes, steer_acts, steer_ints


# ──────────────────────────────────────────────────────────────────────────────
# Append mode: add new variant(s) to an existing pkl
# ──────────────────────────────────────────────────────────────────────────────

def _run_append(args):
    """Append new variant(s) to an existing pkl without full recollection.

    Fast path: if the pkl already contains obs_raw in critical_states, we skip
    environment interaction entirely and just run sample_at_state + rollouts.

    Slow path: if obs_raw is absent (old pkl), we re-discover critical states
    using the first new variant (much faster than full collection).
    """
    if not Path(args.out).exists():
        print(f"[ERROR] --append requires existing pkl at {args.out}")
        return

    with open(args.out, "rb") as f:
        results = pickle.load(f)

    if args.task not in results:
        print(f"[ERROR] task '{args.task}' not found in {args.out}")
        return

    task_data = results[args.task]
    critical_meta = task_data.get("critical_states", {})
    obs_raws = critical_meta.get("obs_raw", None)

    # ── Load only new variants ────────────────────────────────────────────────
    loaded_variants = []
    for run_spec in args.runs:
        label, ckpt_path, user_overrides = parse_run_spec(run_spec)

        if label in task_data:
            print(f"[SKIP] '{label}' already in pkl — skipping")
            continue
        if not Path(ckpt_path).exists():
            print(f"[SKIP] {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'=' * 60}\nAppending: {label}  |  ckpt: {ckpt_path}\n{'=' * 60}")
        overrides = user_overrides + [
            f"optimization.device={args.device}",
            "task.num_envs=1",
            f"optimization.seed={args.seed}",
        ]
        config = load_config(overrides)
        envs = make_vec_env(config.task, seed=args.seed)
        setup_config_for_env(config, envs)
        dataset = make_dataset(config.task)

        try:
            agent, intent_predictor = load_model(ckpt_path, config, dataset, args.device)
        except Exception as e:
            print(f"[ERROR] Failed to load {label}: {e}")
            envs.close()
            continue

        arch_variant = getattr(config.network, "arch_variant", "flow_action")
        is_fi = (arch_variant == "flow_intent")
        from mip.samplers import get_default_step_list
        _num_steps = int(get_default_step_list(config.optimization.loss_type)[0])
        print(f"  arch={arch_variant}  obs_type={config.task.obs_type}  num_steps={_num_steps}")

        loaded_variants.append({
            "label": label, "config": config, "agent": agent,
            "intent_predictor": intent_predictor, "dataset": dataset,
            "envs": envs, "is_fi": is_fi, "num_steps": _num_steps,
        })

    if not loaded_variants:
        print("No new variants to append. Exiting.")
        return

    # ── Rollouts for new variants ─────────────────────────────────────────────
    print(f"\n{'=' * 60}\nRollouts for new variants\n{'=' * 60}")
    for v in loaded_variants:
        label = v["label"]
        print(f"\n  [{label}] Rolling out {args.n_rollouts} episodes...")
        chunks, successes, steer_acts, steer_ints = collect_rollouts_simple(
            v["config"], v["agent"], v["dataset"], v["envs"],
            n_rollouts=args.n_rollouts, device=args.device,
            is_flow_intent=v["is_fi"], intent_predictor=v["intent_predictor"],
            num_steps=v["num_steps"],
        )
        sr = np.mean(successes)
        print(f"  [{label}] SR={sr:.1%}  chunks={len(chunks)}")
        entry = {"rollout_chunks": np.array(chunks), "rollout_success": successes}
        if steer_acts is not None:
            entry["steer_actions"] = steer_acts
            entry["steer_intents"] = steer_ints
        task_data[label] = entry

    # ── Critical states: use stored obs_raw or re-discover ───────────────────
    if obs_raws is not None:
        print(f"\nUsing {len(obs_raws)} stored critical states from pkl (fast path)")
        selected = [
            {"obs_raw": obs_raws[i],
             "timestep": critical_meta["timestep"][i],
             "episode_idx": critical_meta["episode_idx"][i],
             "variance_score": critical_meta["variance_score"][i]}
            for i in range(len(obs_raws))
        ]
        for i, cs in enumerate(selected):
            print(f"  #{i}: ep={cs['episode_idx']} t={cs['timestep']} var={cs['variance_score']:.6f}")
    else:
        print(f"\nobs_raw not in pkl — re-discovering critical states using first new variant")
        dv = loaded_variants[0]
        dv["envs"].close()
        dv["envs"] = make_vec_env(dv["config"].task, seed=args.seed + 100)
        critical_states = find_critical_states(
            dv["config"], dv["agent"], dv["dataset"], dv["envs"], args.device,
            n_rollouts=min(args.n_rollouts, 10),
            n_probe_samples=args.n_probe,
            is_flow_intent=dv["is_fi"],
            intent_predictor=dv["intent_predictor"],
            num_steps=dv["num_steps"],
        )
        n_c = len(critical_meta.get("timestep", [])) or args.n_critical
        selected = critical_states[:n_c]
        print(f"  Re-discovered {len(selected)} critical states")
        for i, cs in enumerate(selected):
            print(f"    #{i}: ep={cs['episode_idx']} t={cs['timestep']} var={cs['variance_score']:.6f}")
        # Save obs_raw for future appends
        critical_meta["obs_raw"] = [cs["obs_raw"] for cs in selected]

    # ── Sample at critical states ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Sampling {args.n_samples} actions at {len(selected)} critical states")
    print(f"{'=' * 60}")
    for v in loaded_variants:
        label = v["label"]
        all_actions, all_intents = [], []
        for si, cs in enumerate(selected):
            print(f"  [{label}] state {si}/{len(selected)}...")
            actions, intents = sample_at_state(
                cs["obs_raw"], v["config"], v["agent"], v["dataset"], args.device,
                n_samples=args.n_samples, is_flow_intent=v["is_fi"],
                intent_predictor=v["intent_predictor"], num_steps=v["num_steps"],
            )
            all_actions.append(actions)
            if intents is not None:
                all_intents.append(intents)
            flat = actions.reshape(args.n_samples, -1)
            print(f"    action variance: {flat.var(axis=0).mean():.6f}")

        task_data[label]["critical_samples"] = np.stack(all_actions)
        task_data[label]["critical_intents"] = np.stack(all_intents) if all_intents else None

    # ── Cleanup & save ────────────────────────────────────────────────────────
    for v in loaded_variants:
        v["envs"].close()

    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nAppended to {args.out}")
    for v in loaded_variants:
        label = v["label"]
        cs_shape = task_data[label].get("critical_samples", np.array([])).shape
        print(f"  {label}: rollouts={task_data[label]['rollout_chunks'].shape}  critical_samples={cs_shape}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect data for standardized multimodality visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True,
                        help="Task label (e.g. lift_ph_state)")
    parser.add_argument("--run", action="append", required=True, dest="runs",
                        metavar="label:ckpt:override...",
                        help="Run spec (repeatable)")
    parser.add_argument("--n-rollouts", type=int, default=20,
                        help="Number of full rollouts per variant (for diversity + critical state discovery)")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of action samples per critical state per variant")
    parser.add_argument("--n-critical", type=int, default=10,
                        help="Number of critical states to select")
    parser.add_argument("--n-probe", type=int, default=10,
                        help="Number of probe samples for critical state discovery")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, help="Output .pkl path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true",
                        help="Append new variant(s) to existing pkl (skips variants already present). "
                             "Uses stored obs_raw for sampling if available, else re-discovers critical states.")
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.append:
        _run_append(args)
        return

    results = {args.task: {}}
    gt_demos_collected = False

    # First pass: load all variants, do rollouts, find critical states from first variant
    loaded_variants = []

    for run_spec in args.runs:
        label, ckpt_path, user_overrides = parse_run_spec(run_spec)

        if not Path(ckpt_path).exists():
            print(f"[SKIP] {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Label: {label}  |  ckpt: {ckpt_path}")
        print(f"{'=' * 60}")

        overrides = user_overrides + [
            f"optimization.device={args.device}",
            "task.num_envs=1",
            f"optimization.seed={args.seed}",
        ]
        config = load_config(overrides)
        envs = make_vec_env(config.task, seed=args.seed)
        setup_config_for_env(config, envs)
        dataset = make_dataset(config.task)

        try:
            agent, intent_predictor = load_model(ckpt_path, config, dataset, args.device)
        except Exception as e:
            print(f"[ERROR] Failed to load {label}: {e}")
            envs.close()
            continue

        arch_variant = getattr(config.network, "arch_variant", "flow_action")
        is_fi = (arch_variant == "flow_intent")

        from mip.samplers import get_default_step_list
        _num_steps = int(get_default_step_list(config.optimization.loss_type)[0])

        print(f"  arch={arch_variant}  obs_type={config.task.obs_type}  "
              f"intent_predictor={'yes' if intent_predictor else 'no/cv'}  "
              f"num_steps={_num_steps}")

        loaded_variants.append({
            "label": label,
            "config": config,
            "agent": agent,
            "intent_predictor": intent_predictor,
            "dataset": dataset,
            "envs": envs,
            "is_fi": is_fi,
            "num_steps": _num_steps,
        })

        # Collect GT demos once
        if not gt_demos_collected:
            print("  Collecting GT demo actions...")
            gt_chunks = collect_gt_demos(dataset, act_steps=config.task.act_steps,
                                          obs_steps=config.task.obs_steps)
            results[args.task]["gt_demos"] = {"action_chunks": gt_chunks}
            gt_demos_collected = True
            print(f"  GT demos: {gt_chunks.shape}")

    if not loaded_variants:
        print("No variants loaded. Exiting.")
        return

    # ── Phase 1: Full rollouts per variant ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase 1: Full rollouts for diversity analysis")
    print(f"{'=' * 60}")

    failed_labels = set()
    for v in loaded_variants:
        label = v["label"]
        print(f"\n  [{label}] Rolling out {args.n_rollouts} episodes...")
        try:
            chunks, successes, steer_acts, steer_ints = collect_rollouts_simple(
                v["config"], v["agent"], v["dataset"], v["envs"],
                n_rollouts=args.n_rollouts,
                device=args.device,
                is_flow_intent=v["is_fi"],
                intent_predictor=v["intent_predictor"],
                num_steps=v["num_steps"],
            )
        except Exception as e:
            print(f"  [{label}] [ERROR] Rollout failed: {e}")
            v["envs"].close()
            failed_labels.add(label)
            continue
        sr = np.mean(successes)
        print(f"  [{label}] Success rate: {sr:.1%}  |  Chunks: {len(chunks)}")

        entry = {
            "rollout_chunks": np.array(chunks),
            "rollout_success": successes,
        }
        if steer_acts is not None:
            entry["steer_actions"] = steer_acts
            entry["steer_intents"] = steer_ints
            print(f"  [{label}] Steer data: {steer_acts.shape}")

        results[args.task][label] = entry

    loaded_variants = [v for v in loaded_variants if v["label"] not in failed_labels]

    # ── Phase 2: Find critical states ─────────────────────────────────────────
    # Use the first variant (typically baseline) to discover critical states
    print(f"\n{'=' * 60}")
    print("Phase 2: Discovering critical states")
    print(f"{'=' * 60}")

    # Use the variant with best success rate (or first if all 0%)
    discovery_variant = max(loaded_variants,
                            key=lambda v: np.mean(results[args.task][v["label"]]["rollout_success"]))
    dv = discovery_variant
    print(f"  Using '{dv['label']}' for critical state discovery "
          f"(SR={np.mean(results[args.task][dv['label']]['rollout_success']):.0%})")

    # Reset envs for discovery
    dv["envs"].close()
    dv["envs"] = make_vec_env(dv["config"].task, seed=args.seed + 100)

    critical_states = find_critical_states(
        dv["config"], dv["agent"], dv["dataset"], dv["envs"], args.device,
        n_rollouts=min(args.n_rollouts, 10),
        n_probe_samples=args.n_probe,
        is_flow_intent=dv["is_fi"],
        intent_predictor=dv["intent_predictor"],
        num_steps=dv["num_steps"],
    )

    # Select top-N critical states
    selected = critical_states[:args.n_critical]
    print(f"  Found {len(critical_states)} states, selected top {len(selected)}")
    for i, cs in enumerate(selected):
        print(f"    #{i}: ep={cs['episode_idx']} t={cs['timestep']} "
              f"var={cs['variance_score']:.6f} eef={cs['eef_pos']}")

    # Store critical state metadata including obs_raw (needed for --append)
    results[args.task]["critical_states"] = {
        "obs_raw": [cs["obs_raw"] for cs in selected],
        "eef_pos": [cs["eef_pos"] for cs in selected],
        "timestep": [cs["timestep"] for cs in selected],
        "episode_idx": [cs["episode_idx"] for cs in selected],
        "variance_score": [cs["variance_score"] for cs in selected],
    }

    # ── Phase 3: Multi-sample at critical states ──────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase 3: Sampling {args.n_samples} actions at {len(selected)} critical states")
    print(f"{'=' * 60}")

    for v in loaded_variants:
        label = v["label"]
        all_actions = []
        all_intents = []

        for si, cs in enumerate(selected):
            print(f"  [{label}] Critical state {si}/{len(selected)} "
                  f"(ep={cs['episode_idx']} t={cs['timestep']})...")

            actions, intents = sample_at_state(
                cs["obs_raw"], v["config"], v["agent"], v["dataset"], args.device,
                n_samples=args.n_samples,
                is_flow_intent=v["is_fi"],
                intent_predictor=v["intent_predictor"],
                num_steps=v["num_steps"],
            )
            all_actions.append(actions)
            if intents is not None:
                all_intents.append(intents)

            # Print variance of samples
            flat = actions.reshape(args.n_samples, -1)
            var = flat.var(axis=0).mean()
            print(f"    action variance: {var:.6f}")

        results[args.task][label]["critical_samples"] = np.stack(all_actions)
        results[args.task][label]["critical_intents"] = (
            np.stack(all_intents) if all_intents else None
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for v in loaded_variants:
        v["envs"].close()

    # Save
    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {args.out}")
    for label, data in results[args.task].items():
        if isinstance(data, dict) and "rollout_chunks" in data:
            cs_shape = data.get("critical_samples", np.array([])).shape
            print(f"  {label}: rollouts={data['rollout_chunks'].shape}  "
                  f"critical_samples={cs_shape}")


if __name__ == "__main__":
    main()
