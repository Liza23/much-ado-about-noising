"""Collect fixed-intent rollout outcomes to visualise spatial multimodality.

For each critical state found during flow_intent rollouts:
  1. Sample N intent vectors from the flow model
  2. K-means cluster the intents into k groups
  3. For each cluster centroid: restore the simulator, run the policy forward
     for n_future_steps with that FIXED intent (bypassing the intent ODE)
  4. Record EEF trajectories + (optionally) rendered RGB frames

The resulting figures show whether different intent clusters lead to genuinely
different robot trajectories — answering "multimodal velocity vs multimodal goals".

Usage:
    python examples/collect_intent_outcomes.py \\
        --run "flow_intent:/ckpt.pt:task=lift_mh_state_flow_intent_emb:+network.arch_variant=flow_intent" \\
        --n-rollouts 10 \\
        --n-critical 5 \\
        --n-future-steps 80 \\
        --device cuda \\
        --out rollouts/lift_mh_state_intent_outcomes.pkl \\
        --out-dir rollouts/lift_mh_state_intent_outcomes_figs

Supports robomimic tasks (lift, square, can) and kitchen with flow_intent checkpoints.
"""

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
    preprocess_obs,
)
from mip.datasets.robomimic_dataset import make_dataset as make_dataset_robomimic
from mip.envs.robomimic.robomimic_env import make_vec_env as make_vec_env_robomimic
from mip.datasets.kitchen_dataset import make_dataset as make_dataset_kitchen
from mip.envs.kitchen import make_vec_env as make_vec_env_kitchen
from mip.datasets.pusht_dataset import make_dataset as make_dataset_pusht
from mip.envs.pusht import make_vec_env as make_vec_env_pusht
from sklearn.cluster import KMeans


def _is_kitchen(config):
    return "kitchen" in getattr(config.task, "env_name", "")


def _is_pusht(config):
    return getattr(config.task, "env_name", "") == "pusht"


def _is_flow_intent(config):
    return getattr(config.network, "arch_variant", "flow_action") == "flow_intent"


def make_vec_env(config, seed):
    if _is_kitchen(config):
        return make_vec_env_kitchen(config.task, seed=seed)
    if _is_pusht(config):
        return make_vec_env_pusht(config.task, seed=seed)
    return make_vec_env_robomimic(config.task, seed=seed)


def make_dataset(config):
    if _is_kitchen(config):
        return make_dataset_kitchen(config.task)
    if _is_pusht(config):
        return make_dataset_pusht(config.task)
    return make_dataset_robomimic(config.task)


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

_ABS_ACTION_ENVS = {"can", "lift", "square", "tool_hang", "transport"}


# ── Robomimic-specific helpers ───────────────────────────────────────────────

def get_lowdim_wrapper(envs):
    """Walk wrapper chain to find MultiStepWrapper (top-level, has get_observation via __getattr__)."""
    e = envs.envs[0]
    while e is not None:
        if hasattr(e, 'get_observation') and hasattr(e, 'obs_keys'):
            return e
        e = getattr(e, 'env', None)
    raise RuntimeError("Could not find LowdimWrapper in env chain")


def get_inner_lowdim(envs):
    """Walk wrapper chain to find the inner RobomimicLowdimWrapper/ImageWrapper.

    Identified by having get_observation defined directly on the class.
    This excludes MultiStepWrapper, which only forwards through __getattr__.
    """
    e = envs.envs[0]
    while e is not None:
        if 'get_observation' in type(e).__dict__:
            return e
        e = getattr(e, 'env', None)
    raise RuntimeError("Could not find inner RobomimicLowdimWrapper/ImageWrapper in env chain")


def get_robomimic_env(envs):
    """Walk wrapper chain to find the raw robomimic env (has get_state/reset_to)."""
    e = envs.envs[0]
    while e is not None:
        if hasattr(e, 'get_state') and hasattr(e, 'reset_to'):
            return e
        e = getattr(e, 'env', None)
    raise RuntimeError("Could not find robomimic env with get_state() in wrapper chain")


# ── Kitchen-specific helpers ─────────────────────────────────────────────────

def _get_kitchen_lowdim(envs):
    """Walk wrapper chain to find KitchenLowdimWrapper."""
    from mip.envs.kitchen.kitchen_thirdparty.kitchen_lowdim_wrapper import KitchenLowdimWrapper
    e = envs.envs[0]
    while e is not None:
        if isinstance(e, KitchenLowdimWrapper):
            return e
        e = getattr(e, 'env', None)
    raise RuntimeError("Could not find KitchenLowdimWrapper in env chain")


def _get_kitchen_base(envs):
    """Walk wrapper chain to find the raw kitchen MuJoCo env (KitchenBase)."""
    from mip.envs.kitchen.kitchen_thirdparty.base import KitchenBase
    e = envs.envs[0]
    while e is not None:
        if isinstance(e, KitchenBase):
            return e
        e = getattr(e, 'env', None)
    raise RuntimeError("Could not find KitchenBase in env chain")


def _get_pusht_base(envs):
    """Walk wrapper chain to find the raw PushT environment."""
    from mip.envs.pusht.pusht_env import PushTEnv

    e = envs.envs[0]
    while e is not None:
        if isinstance(e, PushTEnv):
            return e
        e = getattr(e, 'env', None)
    raise RuntimeError("Could not find PushTEnv in env chain")


# ── Dispatch helpers (work for both robomimic and kitchen) ───────────────────

def get_sim_state(envs, config):
    """Capture full MuJoCo simulator state."""
    if _is_kitchen(config):
        base = _get_kitchen_base(envs)
        qpos = np.array(base.sim.data.qpos).copy()
        qvel = np.array(base.sim.data.qvel).copy()
        return np.concatenate([qpos, qvel])
    if _is_pusht(config):
        base = _get_pusht_base(envs)
        return {
            "agent_pos": np.array(base.agent.position).copy(),
            "agent_vel": np.array(base.agent.velocity).copy(),
            "block_pos": np.array(base.block.position).copy(),
            "block_angle": float(base.block.angle),
            "block_vel": np.array(base.block.velocity).copy(),
            "block_angvel": float(base.block.angular_velocity),
        }
    return get_robomimic_env(envs).get_state()["states"].copy()


def restore_env_obs(envs, sim_state, config, ensure_fresh_reset=False):
    """Restore env to sim_state; return obs_buf in the right shape."""
    if _is_kitchen(config):
        base = _get_kitchen_base(envs)
        nq = base.model.nq
        qpos = sim_state[:nq]
        qvel = sim_state[nq:]
        base.set_state(qpos, qvel)
        base.tasks_to_complete = list(base.TASK_ELEMENTS)
        cur = base.get_observation()
        return np.stack([cur] * config.task.obs_steps, axis=0)[np.newaxis]

    if _is_pusht(config):
        base = _get_pusht_base(envs)
        # PushT restore is most reliable from a freshly rebuilt pymunk world.
        try:
            base.reset()
        except Exception:
            pass
        pos_state = np.array(
            list(np.asarray(sim_state["agent_pos"]).reshape(-1))
            + list(np.asarray(sim_state["block_pos"]).reshape(-1))
            + [float(sim_state["block_angle"])],
            dtype=np.float64,
        )
        base._set_state(pos_state)
        agent_vel = tuple(np.asarray(sim_state["agent_vel"], dtype=np.float64).reshape(-1)[:2].tolist())
        block_vel = tuple(np.asarray(sim_state["block_vel"], dtype=np.float64).reshape(-1)[:2].tolist())
        base.agent.velocity = agent_vel
        base.block.velocity = block_vel
        base.block.angular_velocity = float(sim_state["block_angvel"])
        base.latest_action = None
        cur = base._get_obs()
        if config.task.obs_type == "state":
            return np.stack([cur] * config.task.obs_steps, axis=0)[np.newaxis]
        if isinstance(cur, dict):
            return {
                k: np.stack([v] * config.task.obs_steps, axis=0)[np.newaxis]
                for k, v in cur.items()
            }
        raise RuntimeError(f"Unsupported PushT obs_type for restore: {config.task.obs_type}")

    robo_env = get_robomimic_env(envs)
    inner = get_inner_lowdim(envs)
    if ensure_fresh_reset:
        try:
            robo_env.reset()
        except Exception:
            pass
    robo_env.reset_to({"states": sim_state})
    if config.task.obs_type == "state":
        cur = inner.get_observation()
        return np.stack([cur] * config.task.obs_steps, axis=0)[np.newaxis]
    else:
        cur = inner.get_observation()
        return {k: np.stack([v] * config.task.obs_steps, axis=0)[np.newaxis]
                for k, v in cur.items()}


def _get_inner_step_env(envs, config):
    """Get the inner env whose .step() uses the old 4-value gym API."""
    if _is_kitchen(config):
        return _get_kitchen_lowdim(envs)
    if _is_pusht(config):
        return _get_pusht_base(envs)
    return get_inner_lowdim(envs)


def update_obs_buf(obs_buf, envs, config):
    """Roll obs_buf one step and insert latest observation."""
    if _is_kitchen(config):
        base = _get_kitchen_base(envs)
        new_obs = base.get_observation()
        obs_buf = np.roll(obs_buf, -1, axis=1)
        obs_buf[0, -1] = new_obs
        return obs_buf

    if _is_pusht(config):
        base = _get_pusht_base(envs)
        new_obs = base._get_obs()
        if config.task.obs_type == "state":
            obs_buf = np.roll(obs_buf, -1, axis=1)
            obs_buf[0, -1] = new_obs
            return obs_buf
        if isinstance(new_obs, dict):
            for k in obs_buf:
                obs_buf[k] = np.roll(obs_buf[k], -1, axis=1)
                obs_buf[k][0, -1] = new_obs[k]
            return obs_buf
        raise RuntimeError(f"Unsupported PushT obs_type for update: {config.task.obs_type}")

    inner = get_inner_lowdim(envs)
    if config.task.obs_type == "state":
        new_obs = inner.get_observation()
        obs_buf = np.roll(obs_buf, -1, axis=1)
        obs_buf[0, -1] = new_obs
    else:
        new_obs = inner.get_observation()
        for k in obs_buf:
            obs_buf[k] = np.roll(obs_buf[k], -1, axis=1)
            obs_buf[k][0, -1] = new_obs[k]
    return obs_buf


_KITCHEN_EEF_SITE_ID = 3  # "end_effector" site in franka_kitchen XML


def get_eef_from_envs(envs, config):
    """Get Cartesian EEF position (site_xpos for kitchen, robot0_eef_pos for robomimic)."""
    if _is_kitchen(config):
        base = _get_kitchen_base(envs)
        base.sim.forward()
        return np.array(base.sim.data.site_xpos[_KITCHEN_EEF_SITE_ID]).copy()
    if _is_pusht(config):
        base = _get_pusht_base(envs)
        # For PushT the task outcome is governed by the block, not the agent cursor.
        # Plotting block motion is a better task-level analogue of an EEF trajectory.
        block_pos = np.array(base.block.position).copy()
        return np.array([block_pos[0], block_pos[1], 0.0], dtype=np.float64)
    try:
        return get_robomimic_env(envs).get_observation()["robot0_eef_pos"].copy()
    except Exception:
        return np.zeros(3)


def _trajectory_axis_labels(config, use_2d=False):
    if use_2d and _is_pusht(config):
        return "Block X (px)", "Block Y (px)"
    if use_2d:
        return "X", "Y"
    return "X (m)", "Y (m)"


def _trajectory_entity_name(config, use_2d=False):
    if use_2d and _is_pusht(config):
        return "Block"
    return "EEF"


def render_frame(envs, config):
    """Render an RGB frame from the underlying env wrapper."""
    if _is_kitchen(config):
        lowdim = _get_kitchen_lowdim(envs)
        frame = lowdim.render(mode="rgb_array")
        frame = _normalize_render_frame(frame)
        if frame is None:
            return None
        return np.array(frame, copy=True)

    if _is_pusht(config):
        base = _get_pusht_base(envs)
        frame = base.render(mode="rgb_array")
        frame = _normalize_render_frame(frame)
        if frame is None:
            return None
        return np.array(frame, copy=True)

    inner = get_inner_lowdim(envs)
    frame = inner.render(mode="rgb_array")

    frame = _normalize_render_frame(frame)
    if frame is None:
        return None
    return np.array(frame, copy=True)


def undo_action(action, config, dataset):
    """Undo rotation transform for absolute-action envs (no-op for kitchen)."""
    if getattr(config.task, "abs_action", False) and config.task.env_name in _ABS_ACTION_ENVS:
        return dataset.undo_transform_action(action[np.newaxis])[0]
    return action


def _step_inner_env(inner, action):
    """Step an inner env wrapper and normalize old/new gym APIs to (reward, done, info)."""
    result = inner.step(action)
    if len(result) == 5:
        _, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
    else:
        _, reward, done, info = result
        done = bool(done)
    return float(reward), done, info


def _step_success(config, reward, done, info):
    """Task-specific per-step success predicate used during short rollouts."""
    if _is_pusht(config):
        return bool(done)
    return bool(info.get("success", False))


def _episode_success(config, total_reward, success_reached):
    """Task-specific success aggregation for full-episode trials."""
    if _is_pusht(config):
        return bool(success_reached)
    return total_reward > 0


def _state_signature(sim_state):
    """Build a hashable signature for duplicate frozen-state detection."""
    if isinstance(sim_state, dict):
        parts = []
        for key in sorted(sim_state):
            val = np.asarray(sim_state[key], dtype=np.float64).reshape(-1)
            parts.append((key, tuple(np.round(val, 6).tolist())))
        return tuple(parts)
    arr = np.asarray(sim_state, dtype=np.float64).reshape(-1)
    return tuple(np.round(arr, 6).tolist())


def _snap_centroids_to_samples(samples, raw_centroids):
    """Replace each KMeans centroid by its nearest actual sample.

    This keeps representative intents on the data manifold, which matters for
    pose-valued intents where naive centroid averaging can yield invalid
    quaternions or otherwise unrealistic intermediate targets.
    """
    samples = np.asarray(samples, dtype=np.float64)
    raw_centroids = np.asarray(raw_centroids, dtype=np.float64)
    snapped = np.empty_like(raw_centroids)
    for ki in range(len(raw_centroids)):
        dists = np.linalg.norm(samples - raw_centroids[ki], axis=1)
        snapped[ki] = samples[np.argmin(dists)]
    return snapped


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_fi_obs(obs_buf, config, dataset, device):
    """Preprocess obs_buf and return the tensor used by FlowIntentAgent."""
    obs_in, lowdim_t = preprocess_obs(obs_buf, config, dataset, device)
    if config.task.obs_type == "image":
        from tensordict import TensorDict
        B = next(iter(obs_in.values())).shape[0]
        return TensorDict(obs_in, batch_size=B), lowdim_t
    return lowdim_t, lowdim_t  # fi_obs, lowdim_t


def sample_intents(obs_buf, agent, config, dataset, device, n=50, num_steps=9):
    """Sample n intent vectors via the flow model from frozen obs_buf."""
    fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
    intents = []
    with torch.no_grad():
        for _ in range(n):
            _, iv = agent.sample(obs=fi_obs, use_ema=True,
                                  num_steps=num_steps, return_intent=True)
            intents.append(iv[0].cpu().numpy())
    return np.stack(intents)  # (n, intent_dim)


def decode_fixed_intent(obs_buf, fixed_intent, agent, config, dataset, device):
    """Bypass ODE — encode obs then decode action with a fixed intent centroid."""
    fi_obs, lowdim_t = to_fi_obs(obs_buf, config, dataset, device)
    intent_t = torch.tensor(fixed_intent, device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        obs_emb = agent.encoder_ema(fi_obs, None)
        if obs_emb.dim() == 2:
            obs_emb = obs_emb.unsqueeze(1)
        action_norm = agent.action_decoder_ema(obs_emb, intent_t)  # (1, horizon, act_dim)
    return dataset.normalizer["action"].unnormalize(action_norm.cpu().numpy())  # (1, H, D)


def sample_policy_actions(obs_buf, agent, config, dataset, device, intent_predictor,
                          n=50, num_steps=9):
    """Sample n stochastic baseline action chunks from a frozen observation."""
    s = config.task.obs_steps - 1
    chunks = []
    for _ in range(n):
        obs_in, _ = preprocess_obs(
            obs_buf, config, dataset, device, intent_predictor=intent_predictor
        )
        act_0 = torch.randn(
            (1, config.task.horizon, config.task.act_dim), device=device
        )
        sample_obs = {"state": obs_in} if config.task.obs_type == "state" else obs_in
        with torch.no_grad():
            act_norm = agent.sample(
                act_0=act_0,
                obs=sample_obs,
                use_ema=True,
                num_steps=num_steps,
                sample_mode="stochastic",
            )
        act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
        chunks.append(act_un[0, s:s + config.task.act_steps])
    return np.stack(chunks)  # (n, act_steps, act_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Full-episode runners (for success-rate evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def run_full_episode_fixed_intent(envs, sim_state, fixed_intent, agent, config,
                                   dataset, device, num_steps=9):
    """Restore sim_state, then run to episode completion with a fixed intent.

    Success = cumulative reward > 0, matching collect_diversity_rollouts convention
    (the robot can succeed mid-episode and then drop the object, so last-step
    info["success"] is unreliable).
    Returns (success: bool, n_steps: int).
    """
    obs_buf = restore_env_obs(envs, sim_state, config)
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    done = False
    total_reward = 0.0
    total_steps = 0
    max_steps = config.task.max_episode_steps
    success_reached = False

    while not done and total_steps < max_steps:
        action_un = decode_fixed_intent(obs_buf, fixed_intent, agent, config, dataset, device)
        for a_i in range(act_steps):
            if total_steps >= max_steps:
                break
            act = undo_action(action_un[0, s + a_i], config, dataset)
            reward, done, info = _step_inner_env(inner, act)
            total_reward += reward
            success_reached = success_reached or _step_success(config, reward, done, info)
            total_steps += 1
            if done:
                break
        if not done:
            obs_buf = update_obs_buf(obs_buf, envs, config)

    return _episode_success(config, total_reward, success_reached), total_steps


def run_full_episode_policy(envs, sim_state, agent, config, dataset, device,
                            intent_predictor=None, num_steps=9):
    """Restore sim_state, then run to completion with the variant's standard inference.

    For flow_intent this means freely sampled intents each chunk.
    For baselines / hierarchical models this means stochastic action-flow sampling.
    Success = cumulative reward > 0.
    Returns (success: bool, n_steps: int).
    """
    obs_buf = restore_env_obs(envs, sim_state, config)
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    done = False
    total_reward = 0.0
    total_steps = 0
    max_steps = config.task.max_episode_steps
    success_reached = False

    while not done and total_steps < max_steps:
        if _is_flow_intent(config):
            fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
            with torch.no_grad():
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
        else:
            obs_in, _ = preprocess_obs(
                obs_buf, config, dataset, device, intent_predictor=intent_predictor
            )
            act_0 = torch.randn(
                (1, config.task.horizon, config.task.act_dim), device=device
            )
            sample_obs = {"state": obs_in} if config.task.obs_type == "state" else obs_in
            with torch.no_grad():
                act_norm = agent.sample(
                    act_0=act_0,
                    obs=sample_obs,
                    use_ema=True,
                    num_steps=num_steps,
                    sample_mode="stochastic",
                )
        act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
        for a_i in range(act_steps):
            if total_steps >= max_steps:
                break
            act = undo_action(act_un[0, s + a_i], config, dataset)
            reward, done, info = _step_inner_env(inner, act)
            total_reward += reward
            success_reached = success_reached or _step_success(config, reward, done, info)
            total_steps += 1
            if done:
                break
        if not done:
            obs_buf = update_obs_buf(obs_buf, envs, config)

    return _episode_success(config, total_reward, success_reached), total_steps


def run_full_episode_intent_seed(envs, sim_state, seed_intent, agent, config,
                                  dataset, device, num_steps=9):
    """Restore sim_state, execute ONE policy chunk with a fixed seed intent, then ODE.

    This tests: 'does the CHOICE of intent at this critical timestep affect the outcome?'
    - Step 1: run act_steps env steps using the fixed seed_intent (one policy chunk)
    - Step 2: run the rest of the episode with normal ODE sampling (intent resampled each chunk)

    Success = cumulative reward > 0.
    Returns (success: bool, n_steps: int).
    """
    obs_buf = restore_env_obs(envs, sim_state, config)
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    done = False
    total_reward = 0.0
    total_steps = 0
    max_steps = config.task.max_episode_steps
    first_chunk_done = False
    success_reached = False

    while not done and total_steps < max_steps:
        if not first_chunk_done:
            action_un = decode_fixed_intent(obs_buf, seed_intent, agent, config, dataset, device)
            first_chunk_done = True
        else:
            fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
            with torch.no_grad():
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
            action_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())

        for a_i in range(act_steps):
            if total_steps >= max_steps:
                break
            act = undo_action(action_un[0, s + a_i], config, dataset)
            reward, done, info = _step_inner_env(inner, act)
            total_reward += reward
            success_reached = success_reached or _step_success(config, reward, done, info)
            total_steps += 1
            if done:
                break
        if not done:
            obs_buf = update_obs_buf(obs_buf, envs, config)

    return _episode_success(config, total_reward, success_reached), total_steps


def run_full_episode_action_seed(envs, sim_state, seed_chunk, agent, config,
                                 dataset, device, intent_predictor=None, num_steps=9):
    """Restore sim_state, execute ONE action chunk, then continue standard policy sampling."""
    obs_buf = restore_env_obs(envs, sim_state, config)
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    done = False
    total_reward = 0.0
    total_steps = 0
    max_steps = config.task.max_episode_steps
    first_chunk_done = False
    success_reached = False

    while not done and total_steps < max_steps:
        if not first_chunk_done:
            action_chunk = np.asarray(seed_chunk, dtype=np.float32)
            first_chunk_done = True
        else:
            obs_in, _ = preprocess_obs(
                obs_buf, config, dataset, device, intent_predictor=intent_predictor
            )
            act_0 = torch.randn(
                (1, config.task.horizon, config.task.act_dim), device=device
            )
            sample_obs = {"state": obs_in} if config.task.obs_type == "state" else obs_in
            with torch.no_grad():
                act_norm = agent.sample(
                    act_0=act_0,
                    obs=sample_obs,
                    use_ema=True,
                    num_steps=num_steps,
                    sample_mode="stochastic",
                )
            act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
            action_chunk = act_un[0, s:s + act_steps]

        for a_i in range(act_steps):
            if total_steps >= max_steps:
                break
            act = undo_action(action_chunk[a_i], config, dataset)
            reward, done, info = _step_inner_env(inner, act)
            total_reward += reward
            success_reached = success_reached or _step_success(config, reward, done, info)
            total_steps += 1
            if done:
                break
        if not done:
            obs_buf = update_obs_buf(obs_buf, envs, config)

    return _episode_success(config, total_reward, success_reached), total_steps


# ──────────────────────────────────────────────────────────────────────────────
# Core collection
# ──────────────────────────────────────────────────────────────────────────────

def run_fixed_intent_rollout(envs, obs_buf, fixed_intent, agent, config, dataset,
                              device, n_future_steps, capture_render, num_steps=9):
    """Step env forward after one seeded intent chunk, then continue policy sampling.

    This mirrors the baseline / hierarchical seeded-chunk visualization:
    - first chunk: decode actions from the clustered intent centroid
    - later chunks: resume the policy's standard stochastic inference
    """
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    eef_traj = [get_eef_from_envs(envs, config)]
    frames = []
    if capture_render:
        f0 = render_frame(envs, config)
        if f0 is not None:
            frames.append(f0)
    done = False
    info = {}
    success_reached = False
    first_chunk_done = False

    ignore_done = _is_kitchen(config)
    for step_i in range(0, n_future_steps, act_steps):
        remaining = n_future_steps - step_i
        if not first_chunk_done:
            action_un = decode_fixed_intent(obs_buf, fixed_intent, agent, config, dataset, device)
            first_chunk_done = True
        else:
            fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
            with torch.no_grad():
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
            action_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
        for a_i in range(min(act_steps, remaining)):
            act = undo_action(action_un[0, s + a_i], config, dataset)
            reward, done, info = _step_inner_env(inner, act)
            done = bool(done) and not ignore_done
            success_reached = success_reached or _step_success(config, reward, done, info)
            eef_traj.append(get_eef_from_envs(envs, config))
            if capture_render:
                f = render_frame(envs, config)
                if f is not None:
                    frames.append(f)
            if done:
                break
        obs_buf = update_obs_buf(obs_buf, envs, config)
        if done:
            break

    return {
        "eef_trajectory": np.array(eef_traj),
        "frames": frames,
        "success": bool(success_reached),
        "n_steps": len(eef_traj) - 1,
    }


def run_seeded_chunk_rollout(envs, obs_buf, seed_chunk, agent, config, dataset,
                             device, n_future_steps, capture_render,
                             intent_predictor=None, num_steps=9):
    """Step env forward after executing one seeded action chunk, then continue policy sampling."""
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    inner = _get_inner_step_env(envs, config)
    eef_traj = [get_eef_from_envs(envs, config)]
    frames = []
    if capture_render:
        f0 = render_frame(envs, config)
        if f0 is not None:
            frames.append(f0)
    done = False
    info = {}
    success_reached = False
    first_chunk_done = False

    for step_i in range(0, n_future_steps, act_steps):
        remaining = n_future_steps - step_i
        if not first_chunk_done:
            action_chunk = np.asarray(seed_chunk, dtype=np.float32)
            first_chunk_done = True
        else:
            obs_in, _ = preprocess_obs(
                obs_buf, config, dataset, device, intent_predictor=intent_predictor
            )
            act_0 = torch.randn(
                (1, config.task.horizon, config.task.act_dim), device=device
            )
            sample_obs = {"state": obs_in} if config.task.obs_type == "state" else obs_in
            with torch.no_grad():
                act_norm = agent.sample(
                    act_0=act_0,
                    obs=sample_obs,
                    use_ema=True,
                    num_steps=num_steps,
                    sample_mode="stochastic",
                )
            act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
            action_chunk = act_un[0, s:s + act_steps]

        for a_i in range(min(act_steps, remaining)):
            act = undo_action(action_chunk[a_i], config, dataset)
            reward, done, info = _step_inner_env(inner, act)
            success_reached = success_reached or _step_success(config, reward, done, info)
            eef_traj.append(get_eef_from_envs(envs, config))
            if capture_render:
                f = render_frame(envs, config)
                if f is not None:
                    frames.append(f)
            if done:
                break
        obs_buf = update_obs_buf(obs_buf, envs, config)
        if done:
            break

    return {
        "eef_trajectory": np.array(eef_traj),
        "frames": frames,
        "success": bool(success_reached),
        "n_steps": len(eef_traj) - 1,
    }


def collect_outcomes(config, agent, intent_predictor, dataset, envs, args, device):
    """Main loop: find critical states + collect k-cluster fixed-intent outcomes."""
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    num_steps = 9
    is_fi = _is_flow_intent(config)

    # ── Phase 1: probe rollouts to find high-variance states ──────────────────
    print(f"Probing {args.n_rollouts} rollouts (n_probe={args.n_probe} per state)...")
    all_candidates = []
    n_done = 0

    while n_done < args.n_rollouts:
        if _is_pusht(config):
            obs, _ = envs.reset(seed=args.seed + n_done)
        else:
            obs, _ = envs.reset()
        t = 0

        while t < config.task.max_episode_steps:
            if is_fi:
                fi_obs, _ = to_fi_obs(obs, config, dataset, device)
                probe_intents = []
                with torch.no_grad():
                    for _ in range(args.n_probe):
                        _, iv = agent.sample(
                            obs=fi_obs, use_ema=True,
                            num_steps=num_steps, return_intent=True,
                        )
                        probe_intents.append(iv[0].cpu().numpy())
                variance = float(np.stack(probe_intents).var(axis=0).mean())
            else:
                probe_chunks = sample_policy_actions(
                    obs, agent, config, dataset, device, intent_predictor,
                    n=args.n_probe, num_steps=num_steps,
                )
                variance = float(probe_chunks.reshape(args.n_probe, -1).var(axis=0).mean())

            sim_state = get_sim_state(envs, config)
            eef_pos = get_eef_from_envs(envs, config)
            obs_snap = obs.copy() if config.task.obs_type == "state" else \
                       {k2: v.copy() for k2, v in obs.items()}

            all_candidates.append({
                "obs_raw": obs_snap,
                "sim_state": sim_state,
                "eef_pos": eef_pos,
                "timestep": t,
                "episode_idx": n_done,
                "variance": variance,
            })

            # Step env with one standard sample
            if is_fi:
                with torch.no_grad():
                    act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
            else:
                obs_in, _ = preprocess_obs(
                    obs, config, dataset, device, intent_predictor=intent_predictor
                )
                act_0 = torch.randn(
                    (1, config.task.horizon, config.task.act_dim), device=device
                )
                sample_obs = {"state": obs_in} if config.task.obs_type == "state" else obs_in
                with torch.no_grad():
                    act_norm = agent.sample(
                        act_0=act_0,
                        obs=sample_obs,
                        use_ema=True,
                        num_steps=num_steps,
                        sample_mode="stochastic",
                    )
            act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
            act_chunk = act_un[:, s:s + act_steps]
            if getattr(config.task, "abs_action", False) and \
               config.task.env_name in _ABS_ACTION_ENVS:
                act_chunk = dataset.undo_transform_action(act_chunk)
            obs, _, terminated, truncated, _ = envs.step(act_chunk)
            t += act_steps
            if terminated.any() or truncated.any():
                break

        n_done += config.task.num_envs
        print(f"  Episode {n_done}/{args.n_rollouts}")

    # Filter: only states with enough episode steps remaining for a meaningful rollout
    # and optionally only early-episode states where goal choice hasn't been made yet
    min_remaining = args.n_future_steps
    all_candidates = [c for c in all_candidates
                      if c["timestep"] + min_remaining < config.task.max_episode_steps]
    if args.max_timestep is not None:
        early = [c for c in all_candidates if c["timestep"] <= args.max_timestep]
        if early:
            all_candidates = early
            print(f"  (filtered to {len(all_candidates)} candidates with t <= {args.max_timestep})")
        else:
            print(f"  Warning: no candidates with t <= {args.max_timestep}; using all {len(all_candidates)}")
    if not all_candidates:
        raise RuntimeError(
            f"No candidates with >= {min_remaining} steps remaining. "
            "Try reducing --n-future-steps or increasing --n-rollouts."
        )

    # Select top critical states by variance
    all_candidates.sort(key=lambda x: x["variance"], reverse=True)
    unique_candidates = []
    seen_signatures = set()
    for cand in all_candidates:
        sig = (cand["timestep"], _state_signature(cand["sim_state"]))
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        unique_candidates.append(cand)
    if len(unique_candidates) < len(all_candidates):
        print(f"  (deduplicated {len(all_candidates) - len(unique_candidates)} repeated frozen states)")
    critical = unique_candidates[:args.n_critical]
    print(f"\nTop {len(critical)} critical states selected.")

    # ── Phase 2: collect fixed-intent outcomes for each critical state ─────────
    outcomes = []
    for ci, state in enumerate(critical):
        print(f"\nCritical state {ci+1}/{len(critical)}  "
              f"t={state['timestep']}  var={state['variance']:.5f}")

        obs_raw = state["obs_raw"]

        if is_fi:
            sample_bank = sample_intents(
                obs_raw, agent, config, dataset, device,
                n=args.n_intents, num_steps=num_steps,
            )
            feature_bank = sample_bank
            print(f"  Sampled {len(sample_bank)} intents, shape={sample_bank.shape}")
        else:
            sample_bank = sample_policy_actions(
                obs_raw, agent, config, dataset, device, intent_predictor,
                n=args.n_intents, num_steps=num_steps,
            )
            feature_bank = sample_bank.reshape(len(sample_bank), -1)
            print(f"  Sampled {len(sample_bank)} action chunks, shape={sample_bank.shape}")

        # K-means cluster
        n_clusters = min(args.k_clusters, len(feature_bank))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(feature_bank)
        centroids = km.cluster_centers_  # (k, intent_dim)
        if is_fi:
            centroids = _snap_centroids_to_samples(sample_bank, centroids)

        # Roll out from each centroid with fixed intent / representative action chunk
        cluster_outcomes = []
        for ki in range(n_clusters):
            if is_fi:
                desc = centroids[ki][: min(3, centroids.shape[1])].round(3)
                print(f"  Cluster {ki+1}/{n_clusters} (intent={desc})")
            else:
                desc = centroids[ki][: min(4, centroids.shape[1])].round(3)
                print(f"  Cluster {ki+1}/{n_clusters} (action={desc})")
            obs_buf = restore_env_obs(
                envs,
                state["sim_state"],
                config,
                ensure_fresh_reset=bool(args.render),
            )
            if is_fi:
                result = run_fixed_intent_rollout(
                    envs, obs_buf, centroids[ki], agent, config, dataset, device,
                    n_future_steps=args.n_future_steps,
                    capture_render=args.render,
                    num_steps=num_steps,
                )
                result["intent"] = centroids[ki].copy()
                result["intent_xyz_world"] = _unnormalize_intent_xyz(dataset, centroids[ki])
            else:
                seed_chunk = centroids[ki].reshape(config.task.act_steps, config.task.act_dim)
                result = run_seeded_chunk_rollout(
                    envs, obs_buf, seed_chunk, agent, config, dataset, device,
                    n_future_steps=args.n_future_steps,
                    capture_render=args.render,
                    intent_predictor=intent_predictor,
                    num_steps=num_steps,
                )
                result["seed_action_chunk"] = seed_chunk.copy()
            cluster_outcomes.append(result)
            print(f"    → {result['n_steps']} steps, success={result['success']}, "
                  f"eef_end={result['eef_trajectory'][-1].round(3)}")

        # ── Phase 2b: success-rate trials (if requested) ──────────────────────
        sr_trials = {}
        if args.n_trials > 0:
            rng = np.random.default_rng(42 + ci)
            if is_fi:
                intent_dim = sample_bank.shape[1]
                print(f"  Running {args.n_trials} full-episode trials per condition "
                      f"(seed-then-ODE)...")
                for ki in range(n_clusters):
                    successes = []
                    for trial in range(args.n_trials):
                        ok, nsteps = run_full_episode_intent_seed(
                            envs, state["sim_state"], centroids[ki],
                            agent, config, dataset, device, num_steps=num_steps)
                        successes.append(ok)
                    sr_trials[f"cluster_{ki}"] = successes
                    print(f"    cluster_{ki}: {sum(successes)}/{args.n_trials} successes")

                policy_successes = []
                for trial in range(args.n_trials):
                    ok, nsteps = run_full_episode_policy(
                        envs, state["sim_state"], agent, config, dataset, device,
                        intent_predictor=intent_predictor, num_steps=num_steps)
                    policy_successes.append(ok)
                sr_trials["policy"] = policy_successes
                print(f"    policy(full):  {sum(policy_successes)}/{args.n_trials} successes")

                rand_successes = []
                for trial in range(args.n_trials):
                    rand_intent = rng.standard_normal(intent_dim).astype(np.float32)
                    ok, nsteps = run_full_episode_intent_seed(
                        envs, state["sim_state"], rand_intent,
                        agent, config, dataset, device, num_steps=num_steps)
                    rand_successes.append(ok)
                sr_trials["random"] = rand_successes
                print(f"    random seed:   {sum(rand_successes)}/{args.n_trials} successes")
            else:
                print(f"  Running {args.n_trials} full-episode trials per condition "
                      f"(seeded chunk then policy)...")
                for ki in range(n_clusters):
                    seed_chunk = centroids[ki].reshape(config.task.act_steps, config.task.act_dim)
                    successes = []
                    for trial in range(args.n_trials):
                        ok, nsteps = run_full_episode_action_seed(
                            envs, state["sim_state"], seed_chunk,
                            agent, config, dataset, device,
                            intent_predictor=intent_predictor, num_steps=num_steps)
                        successes.append(ok)
                    sr_trials[f"cluster_{ki}"] = successes
                    print(f"    cluster_{ki}: {sum(successes)}/{args.n_trials} successes")

                policy_successes = []
                for trial in range(args.n_trials):
                    ok, nsteps = run_full_episode_policy(
                        envs, state["sim_state"], agent, config, dataset, device,
                        intent_predictor=intent_predictor, num_steps=num_steps)
                    policy_successes.append(ok)
                sr_trials["policy"] = policy_successes
                print(f"    policy(full):  {sum(policy_successes)}/{args.n_trials} successes")

                rand_successes = []
                for trial in range(args.n_trials):
                    rand_chunk = rng.uniform(
                        low=float(envs.single_action_space.low.min()),
                        high=float(envs.single_action_space.high.max()),
                        size=(config.task.act_steps, config.task.act_dim),
                    ).astype(np.float32)
                    ok, nsteps = run_full_episode_action_seed(
                        envs, state["sim_state"], rand_chunk,
                        agent, config, dataset, device,
                        intent_predictor=intent_predictor, num_steps=num_steps)
                    rand_successes.append(ok)
                sr_trials["random"] = rand_successes
                print(f"    random seed:   {sum(rand_successes)}/{args.n_trials} successes")

        outcomes.append({
            "obs_raw": obs_raw,
            "sim_state": state["sim_state"],
            "eef_pos": state["eef_pos"],
            "timestep": state["timestep"],
            "variance": state["variance"],
            "intents": sample_bank if is_fi else None,
            "seed_action_chunks": sample_bank if not is_fi else None,
            "cluster_labels": labels,
            "centroids": centroids,
            "cluster_outcomes": cluster_outcomes,
            "sr_trials": sr_trials,  # dict of condition → [bool, ...]
        })

    return outcomes


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_render_frame(frame):
    """Convert env render output into an imshow-compatible numeric array."""
    if frame is None:
        return None

    arr = frame
    # Some wrappers return nested single-item containers or object arrays.
    for _ in range(4):
        if isinstance(arr, (list, tuple)) and len(arr) == 1:
            arr = arr[0]
            continue
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
            arr = arr.reshape(-1)[0]
            continue
        break

    arr = np.asarray(arr)
    if arr.dtype == object:
        if arr.size == 1:
            arr = np.asarray(arr.reshape(-1)[0])
        else:
            return None

    # Vec envs may return (1, H, W, C); unwrap the leading batch dim.
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    # Accept grayscale (H, W) or color (H, W, C).
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        pass
    else:
        return None

    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(np.float32)
        except Exception:
            return None
    return arr


def _intent_xyz_for_plot(intent_vec):
    """First 3 components of an intent vector as Cartesian xyz for 3D plots.

    NOTE: this returns raw (still-normalized) values. Use intent_xyz_world from
    cluster outcomes instead whenever a dataset normalizer is available.
    """
    if intent_vec is None:
        return None
    z = np.asarray(intent_vec, dtype=np.float64).reshape(-1)
    if z.size < 3:
        return None
    return z[:3]


def _unnormalize_intent_xyz(dataset, intent_norm):
    """Unnormalize the EEF position part of a normalized 7D intent vector to world space.

    The model is trained to generate normalized intent vectors:
      intent = [robot0_eef_pos(3) + robot0_eef_quat(4)]  in MinMaxNorm [-1, 1] space.

    State dataset: single MinMaxNormalizer("state") covers the full concatenated obs;
      intent lives at indices [dataset.intent_start : dataset.intent_start + 3].
    Image dataset: per-key normalizer at normalizer["obs"]["robot0_eef_pos"].

    Returns world-space xyz as float64 array of shape (3,), or None on failure.
    """
    if intent_norm is None:
        return None
    intent_norm = np.asarray(intent_norm, dtype=np.float64).reshape(-1)
    if intent_norm.size < 3:
        return None
    try:
        obs_norm = dataset.normalizer["obs"]
        if "state" in obs_norm:
            # State dataset: unnormalize the eef_pos slice within the full obs normalizer
            norm = obs_norm["state"]
            s = dataset.intent_start  # byte-offset of robot0_eef_pos in concatenated obs
            pos_world = (intent_norm[:3] + 1.0) / 2.0 * norm.range[s:s + 3] + norm.min[s:s + 3]
        elif "robot0_eef_pos" in obs_norm:
            # Image dataset: dedicated per-key normalizer
            norm = obs_norm["robot0_eef_pos"]
            pos_world = norm.unnormalize(intent_norm[:3].reshape(1, 3)).reshape(-1)
        else:
            return None
        return pos_world.astype(np.float64)
    except Exception:
        return None


def fig_eef_trajectories(
    outcomes,
    out_dir,
    task_name,
    obs_type,
    config,
    show_intent_position=False,
    figure_filename="eef_trajectories.png",
    use_2d=False,
):
    """Plot EEF trajectories (3D or 2D) per critical state, colored by intent cluster."""
    if not use_2d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(outcomes)
    if n == 0:
        return
    k = len(outcomes[0]["cluster_outcomes"])
    cluster_colors = plt.cm.tab10(np.arange(k) / max(k - 1, 1))
    x_label, y_label = _trajectory_axis_labels(config, use_2d=use_2d)
    entity_name = _trajectory_entity_name(config, use_2d=use_2d)

    fig = plt.figure(figsize=(4.5 * n, 4.5))
    if use_2d:
        axes = [fig.add_subplot(1, n, ci + 1) for ci in range(n)]
    else:
        axes = [fig.add_subplot(1, n, ci + 1, projection="3d") for ci in range(n)]

    for ci, (outcome, ax) in enumerate(zip(outcomes, axes)):
        start_eef = outcome["eef_pos"]
        ts = outcome["timestep"]
        xy_points = [start_eef[:2]] if use_2d else [start_eef]

        # Starting position
        if use_2d:
            ax.scatter(start_eef[0], start_eef[1], c="black", s=80, marker="o",
                       label="start" if ci == 0 else None, zorder=10)
        else:
            ax.scatter(*start_eef, c="black", s=80, marker="o", depthshade=False,
                       label="start" if ci == 0 else None, zorder=10)

        intent_legend_done = False
        for ki, co in enumerate(outcome["cluster_outcomes"]):
            traj = co["eef_trajectory"]   # (T, 3)
            c = cluster_colors[ki]
            label = f"Cluster {ki+1}" if ci == 0 else None
            if use_2d:
                xy_points.append(traj[:, :2])
                ax.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.85, lw=1.8, label=label)
                ax.scatter(traj[-1, 0], traj[-1, 1], color=c, s=80, marker="*", zorder=5)
            else:
                xy_points.append(traj)
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                        color=c, alpha=0.85, lw=1.8, label=label)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                           color=c, s=80, marker="*", depthshade=False, zorder=5)
            if show_intent_position and not use_2d:
                pz = co.get("intent_xyz_world")  # world-space eef_pos from unnormalized intent
                if pz is not None:
                    xy_points.append(pz)
                    ilab = "intent EEF pos (world)" if (ci == 0 and not intent_legend_done) else None
                    if ilab is not None:
                        intent_legend_done = True
                    ax.scatter(
                        pz[0],
                        pz[1],
                        pz[2],
                        color=c,
                        s=85,
                        marker="^",
                        depthshade=False,
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=8,
                        label=ilab,
                    )

        # Keep all axes at comparable scale so geometry is not visually distorted.
        pts = np.vstack(xy_points)
        pts_min = pts.min(axis=0)
        pts_max = pts.max(axis=0)
        center = 0.5 * (pts_min + pts_max)
        max_range = float(np.max(pts_max - pts_min))
        if max_range < 1e-6:
            max_range = 1e-3
        half = 0.55 * max_range
        if use_2d:
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_aspect("equal")
            ax.set_xlabel(x_label, fontsize=7, labelpad=2)
            ax.set_ylabel(y_label, fontsize=7, labelpad=2)
        else:
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)
            ax.view_init(elev=25, azim=-55)
            ax.set_xlabel(x_label, fontsize=7, labelpad=2)
            ax.set_ylabel(y_label, fontsize=7, labelpad=2)
            ax.set_zlabel("Z (m)", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.set_title(f"S{ci}  t={ts}", fontsize=8)
        if ci == 0:
            ax.legend(fontsize=7, loc="upper left")

    sub = "(●=start  ★=end"
    if show_intent_position:
        sub += "  ^=intent EEF pos (world space)"
    sub += ")"
    fig.suptitle(
        f"[{task_name}] {entity_name} trajectories under fixed intent clusters\n{sub}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / figure_filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_rendered_frames(outcomes, out_dir, task_name):
    """Show start frame + final frame per cluster for each critical state."""
    fig_rendered_frames_filtered(outcomes, out_dir, task_name, success_only=False, success_threshold=0.5)


def _cluster_success_rate(outcome, cluster_idx):
    """Return success rate for a cluster, preferring full-episode trials when available."""
    sr = outcome.get("sr_trials", {})
    key = f"cluster_{cluster_idx}"
    vals = sr.get(key, [])
    if vals:
        return float(np.mean(vals))
    # Fallback: short fixed-intent rollout flag (single boolean)
    co = outcome["cluster_outcomes"][cluster_idx]
    return 1.0 if bool(co.get("success", False)) else 0.0


def _select_cluster_indices(outcome, success_only=False, success_threshold=0.5):
    k = len(outcome["cluster_outcomes"])
    if not success_only:
        return list(range(k))
    keep = []
    for ki in range(k):
        if _cluster_success_rate(outcome, ki) >= success_threshold:
            keep.append(ki)
    return keep


def fig_rendered_frames_filtered(outcomes, out_dir, task_name, success_only=False, success_threshold=0.5):
    """Show start frame + final frame per cluster; optionally keep successful clusters only."""
    n = len(outcomes)
    if n == 0:
        return

    cluster_indices_per_state = [
        _select_cluster_indices(o, success_only=success_only, success_threshold=success_threshold)
        for o in outcomes
    ]
    max_cols_clusters = max((len(v) for v in cluster_indices_per_state), default=0)
    if max_cols_clusters == 0:
        print("No successful clusters to render; skipping rendered_frames.png")
        return
    n_cols = max_cols_clusters + 1  # start + selected cluster ends

    fig, axes = plt.subplots(n, n_cols, figsize=(3.5 * n_cols, 3 * n))
    if n == 1:
        axes = axes.reshape(1, n_cols)

    for ci, (outcome, keep_indices) in enumerate(zip(outcomes, cluster_indices_per_state)):
        ts = outcome["timestep"]
        # Column 0: start frame (first frame of cluster 0 rollout)
        ax = axes[ci, 0]
        start_frames = outcome["cluster_outcomes"][0]["frames"]
        start_im = _normalize_render_frame(start_frames[0]) if start_frames else None
        if start_im is not None:
            ax.imshow(start_im)
        else:
            ax.text(0.5, 0.5, "No frame", ha="center", va="center", fontsize=8)
        ax.set_title(f"S{ci} start\nt={ts}", fontsize=8)
        ax.axis("off")

        for col_i, ki in enumerate(keep_indices, start=1):
            co = outcome["cluster_outcomes"][ki]
            ax = axes[ci, col_i]
            frames = co["frames"]
            end_im = _normalize_render_frame(frames[-1]) if frames else None
            if end_im is not None:
                ax.imshow(end_im)
            else:
                ax.text(0.5, 0.5, "No frame", ha="center", va="center", fontsize=8)
            sr = _cluster_success_rate(outcome, ki)
            ax.set_title(f"C{ki+1} end ({sr:.2f})", fontsize=8)
            ax.axis("off")

        # Hide any unused columns for this row.
        for col_i in range(len(keep_indices) + 1, n_cols):
            axes[ci, col_i].axis("off")

    subtitle = "successful clusters only" if success_only else "all clusters"
    fig.suptitle(
        f"[{task_name}] Start vs end frames per intent cluster ({subtitle})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "rendered_frames.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_trajectories_with_frames(
    outcomes,
    out_dir,
    task_name,
    config,
    success_only=False,
    success_threshold=0.5,
    show_intent_position=False,
    figure_filename="eef_trajectories_with_frames.png",
    use_2d=False,
):
    """Combined view: EEF trajectory (3D or 2D) + rendered start/end frames per state."""
    if not use_2d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(outcomes)
    if n == 0:
        return
    k = len(outcomes[0]["cluster_outcomes"])
    cluster_colors = plt.cm.tab10(np.arange(k) / max(k - 1, 1))
    x_label, y_label = _trajectory_axis_labels(config, use_2d=use_2d)
    entity_name = _trajectory_entity_name(config, use_2d=use_2d)
    cluster_indices_per_state = [
        _select_cluster_indices(o, success_only=success_only, success_threshold=success_threshold)
        for o in outcomes
    ]
    max_cols_clusters = max((len(v) for v in cluster_indices_per_state), default=0)
    if max_cols_clusters == 0:
        print("No successful clusters to render; skipping eef_trajectories_with_frames.png")
        return
    n_cols = max_cols_clusters + 2  # traj + start + selected cluster ends

    fig = plt.figure(figsize=(3.2 * n_cols, 3.2 * n))

    for ci, (outcome, keep_indices) in enumerate(zip(outcomes, cluster_indices_per_state)):
        ts = outcome["timestep"]
        start_eef = outcome["eef_pos"]
        pts = [start_eef[:2]] if use_2d else [start_eef]

        # Column 0: EEF trajectories
        if use_2d:
            ax3d = fig.add_subplot(n, n_cols, ci * n_cols + 1)
            ax3d.scatter(
                start_eef[0], start_eef[1],
                c="black", s=60, marker="o",
                label="start" if ci == 0 else None,
                zorder=10,
            )
        else:
            ax3d = fig.add_subplot(n, n_cols, ci * n_cols + 1, projection="3d")
            ax3d.scatter(
                *start_eef,
                c="black",
                s=60,
                marker="o",
                depthshade=False,
                label="start" if ci == 0 else None,
                zorder=10,
            )
        intent_legend_done = False
        for ki in keep_indices:
            co = outcome["cluster_outcomes"][ki]
            traj = co["eef_trajectory"]
            c = cluster_colors[ki]
            label = f"C{ki+1}" if ci == 0 else None
            if use_2d:
                pts.append(traj[:, :2])
                ax3d.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.85, lw=1.6, label=label)
                ax3d.scatter(traj[-1, 0], traj[-1, 1], color=c, s=55, marker="*", zorder=5)
            else:
                pts.append(traj)
                ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=c, alpha=0.85, lw=1.6, label=label)
                ax3d.scatter(
                    traj[-1, 0],
                    traj[-1, 1],
                    traj[-1, 2],
                    color=c,
                    s=55,
                    marker="*",
                    depthshade=False,
                    zorder=5,
                )
            if show_intent_position and not use_2d:
                pz = co.get("intent_xyz_world")  # world-space eef_pos from unnormalized intent
                if pz is not None:
                    pts.append(pz)
                    ilab = "intent EEF pos (world)" if (ci == 0 and not intent_legend_done) else None
                    if ilab is not None:
                        intent_legend_done = True
                    ax3d.scatter(
                        pz[0],
                        pz[1],
                        pz[2],
                        color=c,
                        s=70,
                        marker="^",
                        depthshade=False,
                        edgecolors="black",
                        linewidths=0.45,
                        zorder=8,
                        label=ilab,
                    )

        all_pts = np.vstack(pts)
        pts_min = all_pts.min(axis=0)
        pts_max = all_pts.max(axis=0)
        center = 0.5 * (pts_min + pts_max)
        max_range = float(np.max(pts_max - pts_min))
        if max_range < 1e-6:
            max_range = 1e-3
        half = 0.55 * max_range
        if use_2d:
            ax3d.set_xlim(center[0] - half, center[0] + half)
            ax3d.set_ylim(center[1] - half, center[1] + half)
            ax3d.set_aspect("equal")
            ax3d.set_xlabel(x_label, fontsize=7, labelpad=1)
            ax3d.set_ylabel(y_label, fontsize=7, labelpad=1)
        else:
            ax3d.set_xlim(center[0] - half, center[0] + half)
            ax3d.set_ylim(center[1] - half, center[1] + half)
            ax3d.set_zlim(center[2] - half, center[2] + half)
            ax3d.view_init(elev=25, azim=-55)
            ax3d.set_xlabel(x_label, fontsize=7, labelpad=1)
            ax3d.set_ylabel(y_label, fontsize=7, labelpad=1)
            ax3d.set_zlabel("Z", fontsize=7, labelpad=1)
        ax3d.tick_params(labelsize=6)
        ax3d.set_title(f"S{ci} traj\nt={ts}", fontsize=8)
        if ci == 0:
            ax3d.legend(fontsize=7, loc="upper left")

        # Column 1: start frame
        ax_start = fig.add_subplot(n, n_cols, ci * n_cols + 2)
        start_frames = outcome["cluster_outcomes"][0]["frames"]
        start_im = _normalize_render_frame(start_frames[0]) if start_frames else None
        if start_im is not None:
            ax_start.imshow(start_im)
        else:
            ax_start.text(0.5, 0.5, "No frame", ha="center", va="center", fontsize=8)
        ax_start.set_title(f"S{ci} start", fontsize=8)
        ax_start.axis("off")

        # Remaining columns: end frame for each cluster
        for col_i, ki in enumerate(keep_indices, start=0):
            co = outcome["cluster_outcomes"][ki]
            ax = fig.add_subplot(n, n_cols, ci * n_cols + 3 + col_i)
            frames = co["frames"]
            end_im = _normalize_render_frame(frames[-1]) if frames else None
            if end_im is not None:
                ax.imshow(end_im)
            else:
                ax.text(0.5, 0.5, "No frame", ha="center", va="center", fontsize=8)
            sr = _cluster_success_rate(outcome, ki)
            ax.set_title(f"C{ki+1} end ({sr:.2f})", fontsize=8)
            ax.axis("off")

        # Hide unused columns in this row.
        for col_i in range(len(keep_indices), max_cols_clusters):
            ax_empty = fig.add_subplot(n, n_cols, ci * n_cols + 3 + col_i)
            ax_empty.axis("off")

    subtitle = "successful clusters only" if success_only else "all clusters"
    legend_bits = "●=start  ★=end"
    if show_intent_position:
        legend_bits += "  ^=intent EEF pos (world)"
    fig.suptitle(
        f"[{task_name}] {entity_name} trajectories + rendered outcomes per intent cluster\n"
        f"({legend_bits}, {subtitle})",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / figure_filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_success_rates(outcomes, out_dir, task_name, k_clusters):
    """Bar chart of success rate per clustered seed condition, one panel per critical state."""
    # Only include outcomes that have trial data
    outcomes_with_sr = [o for o in outcomes if o.get("sr_trials")]
    if not outcomes_with_sr:
        return

    n = len(outcomes_with_sr)
    cluster_keys = sorted(
        {
            key
            for outcome in outcomes_with_sr
            for key in outcome["sr_trials"]
            if key.startswith("cluster_")
        },
        key=lambda k: int(k.split("_")[1]),
    )
    extras = [name for name in ["policy", "random"] if any(name in o["sr_trials"] for o in outcomes_with_sr)]
    all_conditions = cluster_keys + extras

    label_map = {"policy": "Policy\n(full)", "random": "Random"}
    cond_labels = [f"C{int(k.split('_')[1]) + 1}" for k in cluster_keys] + [
        label_map.get(name, name) for name in extras
    ]

    # Colors: tab10 for clusters, dark gray for policy, light gray for random
    tab10 = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = [tab10[i % 10] for i in range(len(cluster_keys))]
    if "policy" in extras:
        colors.append(np.array([0.3, 0.3, 0.3, 1.0]))
    if "random" in extras:
        colors.append(np.array([0.7, 0.7, 0.7, 1.0]))

    fig, axes = plt.subplots(1, n, figsize=(max(4, 2.5 * len(all_conditions)) * n, 4),
                              sharey=True)
    if n == 1:
        axes = [axes]

    for ci, outcome in enumerate(outcomes_with_sr):
        ax = axes[ci]
        sr = outcome["sr_trials"]
        ts = outcome["timestep"]
        n_trials = max(len(v) for v in sr.values())

        means, errs = [], []
        for cond in all_conditions:
            vals = sr.get(cond, [])
            mean = np.mean(vals) if vals else 0.0
            # Wilson score interval half-width for binary proportion
            n_v = len(vals)
            stderr = np.sqrt(mean * (1 - mean) / n_v) if n_v > 0 else 0.0
            means.append(mean)
            errs.append(stderr)

        xs = np.arange(len(all_conditions))
        bars = ax.bar(xs, means, color=colors, alpha=0.85, width=0.6,
                      yerr=errs, capsize=4, error_kw={"linewidth": 1.2})

        # Annotate with raw counts
        for x, mean, cond in zip(xs, means, all_conditions):
            vals = sr.get(cond, [])
            ax.text(x, mean + 0.04, f"{sum(vals)}/{len(vals)}", ha="center",
                    va="bottom", fontsize=7)

        ax.set_xticks(xs)
        ax.set_xticklabels(cond_labels, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"S{ci}  t={ts}", fontsize=8)
        if ci == 0:
            ax.set_ylabel("Success Rate", fontsize=9)
        ax.axhline(0.5, color="black", lw=0.6, ls="--", alpha=0.4)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"[{task_name}] Success rate per seed condition  (n={n_trials} trials each)\n"
        "C1…Ck = clustered seed for one chunk  |  Policy = standard inference  |  Random = random seeded chunk/intent",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "success_rates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="'label:ckpt_path:task=X[:overrides]'")
    parser.add_argument("--n-rollouts", type=int, default=10,
                        help="Rollout episodes for critical state discovery")
    parser.add_argument("--n-critical", type=int, default=5,
                        help="Number of critical states to collect outcomes for")
    parser.add_argument("--n-probe", type=int, default=20,
                        help="Probe samples per state for variance probing")
    parser.add_argument("--n-intents", type=int, default=50,
                        help="Samples at critical states for clustering (intents for flow_intent, action chunks for baselines)")
    parser.add_argument("--k-clusters", type=int, default=3,
                        help="Number of clusters")
    parser.add_argument("--n-future-steps", type=int, default=80,
                        help="Steps to run forward with a fixed seed branch")
    parser.add_argument("--n-trials", type=int, default=0,
                        help="Full-episode trials per seed condition for success-rate "
                             "evaluation. 0 = skip. Runs k_clusters + policy + random "
                             "conditions, n_trials each. Recommended: 10-20.")
    parser.add_argument("--max-timestep", type=int, default=None,
                        help="Only pick critical states at or before this episode timestep "
                             "(default: no limit). Use e.g. 100 to focus on early-episode "
                             "states where high-level goal choice is being made.")
    parser.add_argument("--render", action="store_true",
                        help="Capture rendered RGB frames during rollouts")
    parser.add_argument(
        "--plot-intent-xyz",
        action="store_true",
        help="Also save eef_trajectories_with_intent_xyz.png and "
        "(with --render) eef_trajectories_with_frames_intent_xyz.png: same panels as "
        "the default figures, plus ^ markers at intent z[:3]. Default PNGs are unchanged.",
    )
    parser.add_argument("--render-success-only", action="store_true",
                        help="When --render is enabled, only include successful cluster "
                             "outcomes in rendered figures (uses full-episode trial "
                             "success rates if available, else short-rollout flag).")
    parser.add_argument("--success-threshold", type=float, default=0.5,
                        help="Success-rate threshold used by --render-success-only "
                             "(default: 0.5).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="Output .pkl path")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for figures (default: alongside --out)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label, ckpt_path, overrides = parse_run_spec(args.run)
    print(f"[collect_intent_outcomes] variant={label}  ckpt={ckpt_path}")

    # Load config
    config = load_config(overrides)
    config.optimization.device = args.device
    config.task.num_envs = 1
    # Reuse save_video flag to request offscreen rendering contexts for state envs.
    config.task.save_video = bool(args.render)

    # Setup env + dataset
    envs = make_vec_env(config, seed=args.seed)
    setup_config_for_env(config, envs)
    dataset = make_dataset(config)

    # Load model
    agent, intent_predictor = load_model(ckpt_path, config, dataset, args.device)
    agent.eval()

    print(f"Task: {config.task.env_name}  obs_type: {config.task.obs_type}  "
          f"intent_dim: {getattr(config.task, 'intent_dim', '?')}")

    # Collect
    outcomes = collect_outcomes(config, agent, intent_predictor, dataset, envs, args, args.device)

    # Save pkl
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task": config.task.env_name,
        "obs_type": config.task.obs_type,
        "arch_variant": getattr(config.network, "arch_variant", "flow_action"),
        "intent_dim": getattr(config.task, "intent_dim", 7),
        "k_clusters": args.k_clusters,
        "n_future_steps": args.n_future_steps,
        "outcomes": outcomes,
    }
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved pkl: {out_path}")

    # Figures
    out_dir = Path(args.out_dir) if args.out_dir else out_path.parent / (out_path.stem + "_figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"{config.task.env_name} [{label}]"
    use_2d = _is_pusht(config)
    fig_eef_trajectories(
        outcomes,
        out_dir,
        task_name,
        config.task.obs_type,
        config,
        show_intent_position=False,
        use_2d=use_2d,
    )
    if args.plot_intent_xyz:
        fig_eef_trajectories(
            outcomes,
            out_dir,
            task_name,
            config.task.obs_type,
            config,
            show_intent_position=True,
            figure_filename="eef_trajectories_with_intent_xyz.png",
            use_2d=use_2d,
        )
    if args.n_trials > 0:
        fig_success_rates(outcomes, out_dir, task_name, args.k_clusters)
    if args.render and outcomes and outcomes[0]["cluster_outcomes"][0]["frames"]:
        fig_rendered_frames_filtered(
            outcomes,
            out_dir,
            task_name,
            success_only=args.render_success_only,
            success_threshold=args.success_threshold,
        )
        fig_trajectories_with_frames(
            outcomes,
            out_dir,
            task_name,
            config,
            success_only=args.render_success_only,
            success_threshold=args.success_threshold,
            show_intent_position=False,
            use_2d=use_2d,
        )
        if args.plot_intent_xyz:
            fig_trajectories_with_frames(
                outcomes,
                out_dir,
                task_name,
                config,
                success_only=args.render_success_only,
                success_threshold=args.success_threshold,
                show_intent_position=True,
                figure_filename="eef_trajectories_with_frames_intent_xyz.png",
                use_2d=use_2d,
            )

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
