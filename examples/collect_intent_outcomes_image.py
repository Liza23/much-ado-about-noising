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

Only supports robomimic tasks (lift, square, can) with flow_intent checkpoints.
"""

import argparse
import os
import pickle
import re
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
from mip.datasets.libero_dataset import make_dataset as make_dataset_libero
from mip.envs.libero import make_vec_env as make_vec_env_libero
from mip.agent import TrainingAgent
from robosuite.models.objects import MujocoXMLObject
from sklearn.cluster import KMeans


class BaselineAgentAdapter:
    """Wraps TrainingAgent to provide a FlowIntentAgent-compatible interface.

    The "intent" for a baseline model is the sampled action (flattened).
    Fixed-intent rollout = repeat the same action chunk at every step.
    This lets collect_intent_outcomes_image.py visualize baseline action diversity.
    """

    _is_baseline_adapter = True

    def __init__(self, inner_agent, config):
        self._inner = inner_agent
        self._config = config
        self.encoder_ema = inner_agent.encoder_ema

    def sample(self, obs, use_ema=True, num_steps=9, return_intent=False):
        B = obs.shape[0]
        horizon = self._config.task.horizon
        act_dim = self._config.task.act_dim
        device = obs.device
        sample_mode = getattr(self._config.optimization, "sample_mode", "stochastic")
        if sample_mode == "zero":
            act_0 = torch.zeros(B, horizon, act_dim, device=device)
        else:
            act_0 = torch.randn(B, horizon, act_dim, device=device)
        act = self._inner.sample(act_0, obs, use_ema=use_ema, num_steps=num_steps)
        if return_intent:
            iv = act.reshape(B, -1).clone()
            return act, iv
        return act

    def eval(self):
        self._inner.eval()
        return self


def _is_kitchen(config):
    return "kitchen" in getattr(config.task, "env_name", "")


def _is_libero(config):
    return getattr(config.task, "env_name", "").startswith("libero")


def maybe_register_libero_pro_objects(config):
    """Load LIBERO-PRO custom object classes into the active LIBERO registry.

    Some LIBERO-PRO checkouts ship custom object classes whose XML paths are not
    valid relative to the current environment. We register corrected runtime
    classes here so perturbation suites can instantiate without modifying the
    external LIBERO trees.
    """
    if not _is_libero(config):
        return

    bddl_file = os.path.expanduser(getattr(config.task, "bddl_file", "") or "")
    if "LIBERO-PRO" not in bddl_file:
        return

    from libero.libero.envs.base_object import OBJECTS_DICT

    def _make_runtime_object_class(class_name, xml_path, rotation, rotation_axis):
        xml_path = os.path.abspath(xml_path)

        def __init__(self, name=None, obj_name=None):
            name = name or "_".join(re.sub(r"([A-Z])", r" \1", class_name).split()).lower()
            MujocoXMLObject.__init__(
                self,
                xml_path,
                name=name,
                joints=[dict(type="free", damping="0.0005")],
                obj_type="all",
                duplicate_collision_geoms=False,
            )
            self.category_name = "_".join(
                re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
            ).lower()
            self.object_properties = {"vis_site_names": {}}
            self.rotation = rotation
            self.rotation_axis = rotation_axis

        return type(class_name, (MujocoXMLObject,), {"__init__": __init__})

    zero_rot = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }
    mug_rot = {
        "x": (-np.pi / 2, -np.pi / 2),
        "y": (-np.pi, -np.pi),
        "z": (np.pi, np.pi),
    }

    asset_specs = {
        "red_sticker": (
            "RedSticker",
            os.path.expanduser(
                "~/LIBERO-PRO/libero/libero/assets/stable_scanned_objects/red_sticker/red_sticker.xml"
            ),
            zero_rot,
            "z",
        ),
        "blue_red_sticker": (
            "BlueRedSticker",
            str(ROOT / "examples" / "assets" / "blue_red_sticker.xml"),
            zero_rot,
            "z",
        ),
        "red_box": (
            "RedBox",
            str(ROOT / "examples" / "assets" / "red_box.xml"),
            zero_rot,
            "z",
        ),
        "libero_mug_yellow": (
            "LiberoMugYellow",
            os.path.expanduser(
                "~/LIBERO-PRO/notebooks/custom_assets/libero_mug_yellow/libero_mug_yellow.xml"
            ),
            mug_rot,
            None,
        ),
    }

    for key, (class_name, xml_path, rotation, rotation_axis) in asset_specs.items():
        if not os.path.exists(xml_path):
            continue
        OBJECTS_DICT[key] = _make_runtime_object_class(
            class_name, xml_path, rotation, rotation_axis
        )


def make_vec_env(config, seed):
    if _is_libero(config):
        return make_vec_env_libero(config.task, seed=seed)
    if _is_kitchen(config):
        return make_vec_env_kitchen(config.task, seed=seed)
    return make_vec_env_robomimic(config.task, seed=seed)


def make_dataset(config):
    if _is_libero(config):
        return make_dataset_libero(config.task)
    if _is_kitchen(config):
        return make_dataset_kitchen(config.task)
    return make_dataset_robomimic(config.task)


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

_ABS_ACTION_ENVS = {"can", "lift", "square", "tool_hang", "transport"}


def get_lowdim_wrapper(envs):
    """Walk wrapper chain to find MultiStepWrapper (top-level, has get_observation via __getattr__)."""
    e = envs.envs[0]
    while e is not None:
        if hasattr(e, 'get_observation') and hasattr(e, 'obs_keys'):
            return e  # MultiStepWrapper (forwards attrs to inner lowdim)
        e = getattr(e, 'env', None)
    raise RuntimeError("Could not find LowdimWrapper in env chain")


def get_inner_lowdim(envs):
    """Walk wrapper chain to find the inner RobomimicLowdimWrapper/ImageWrapper.

    Unlike get_lowdim_wrapper (which may return MultiStepWrapper), this returns
    the wrapper whose .step() has the old 4-value gym API (obs, reward, done, info)
    and has no internal done-state that persists across episodes.

    Identified by having get_observation defined as a direct class method:
    - RobomimicLowdimWrapper  → has get_observation in class __dict__ ✓
    - RobomimicImageWrapper   → has get_observation in class __dict__ ✓
    - MultiStepWrapper        → only forwards via __getattr__, NOT in class __dict__ ✗
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


def _get_libero_base(envs):
    """Walk wrapper chain to find the base LIBERO gym wrapper."""
    from mip.envs.libero.libero_env_wrapper import LiberoGymWrapper

    e = envs.envs[0]
    while e is not None:
        if isinstance(e, LiberoGymWrapper):
            return e
        e = getattr(e, 'env', None)
    raise RuntimeError("Could not find LiberoGymWrapper in env chain")


def _stack_obs_for_policy(cur, config):
    """Expand a single observation into an obs_steps buffer."""
    if config.task.obs_type == "state":
        return np.stack([cur] * config.task.obs_steps, axis=0)[np.newaxis]
    return {k: np.stack([v] * config.task.obs_steps, axis=0)[np.newaxis] for k, v in cur.items()}


def _get_current_obs(envs, config):
    """Return the latest single-step observation from the active env."""
    if _is_libero(config):
        base = _get_libero_base(envs)
        cur = getattr(base, "_last_obs", None)
        if cur is None:
            raw_obs = base._env.regenerate_obs_from_state(base._env.get_sim_state())
            base._last_obs = base._get_obs(raw_obs)
            cur = base._last_obs
        if config.task.obs_type == "state":
            return cur.copy()
        return {k: v.copy() for k, v in cur.items()}

    inner = get_inner_lowdim(envs)
    return inner.get_observation()


def _get_inner_step_env(envs, config):
    """Return the env object to step one action at a time."""
    if _is_libero(config):
        return _get_libero_base(envs)
    return get_inner_lowdim(envs)


def _step_inner(inner, action):
    """Compatibility helper for old Gym 4-tuple and Gymnasium 5-tuple step APIs."""
    result = inner.step(action)
    if len(result) == 5:
        _, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
    else:
        _, reward, done, info = result
        done = bool(done)
    return float(reward), done, info


def get_sim_state(envs, config):
    """Capture full simulator state."""
    if _is_libero(config):
        return _get_libero_base(envs)._env.get_sim_state().copy()
    return get_robomimic_env(envs).get_state()["states"].copy()


def restore_env_obs(envs, sim_state, config, ensure_fresh_reset=False):
    """Restore env to sim_state; return obs_buf in the right shape.

    Uses get_inner_lowdim (RobomimicLowdimWrapper) so that only the MuJoCo state
    is restored — we do NOT go through MultiStepWrapper.reset(), which would trigger
    a full random robomimic reset and clobber the desired state.
    """
    if _is_libero(config):
        base = _get_libero_base(envs)
        # LIBERO / robosuite can keep an internal terminated flag after a
        # successful rollout; a fresh reset clears it before state restore.
        try:
            base._env.reset()
        except Exception:
            pass
        raw_obs = base._env.regenerate_obs_from_state(sim_state)
        base._last_obs = base._get_obs(raw_obs)
        return _stack_obs_for_policy(_get_current_obs(envs, config), config)

    robo_env = get_robomimic_env(envs)
    inner    = get_inner_lowdim(envs)
    if ensure_fresh_reset:
        # Offscreen render buffers in robosuite can become stale after many reset_to calls.
        # A fresh reset before reset_to keeps camera output stable while preserving state.
        try:
            robo_env.reset()
        except Exception:
            pass
    robo_env.reset_to({"states": sim_state})
    cur = inner.get_observation()
    return _stack_obs_for_policy(cur, config)


def update_obs_buf(obs_buf, envs, config):
    """Roll obs_buf one step and insert latest observation."""
    new_obs = _get_current_obs(envs, config)
    if config.task.obs_type == "state":
        obs_buf = np.roll(obs_buf, -1, axis=1)
        obs_buf[0, -1] = new_obs
    else:
        for k in obs_buf:
            obs_buf[k] = np.roll(obs_buf[k], -1, axis=1)
            obs_buf[k][0, -1] = new_obs[k]
    return obs_buf


def get_eef_from_envs(envs, config):
    """Get EEF position from the active env (works for robomimic + LIBERO image/state)."""
    if _is_libero(config):
        try:
            state = _get_current_obs(envs, config)
            if isinstance(state, dict):
                state = state["state"]
            return np.asarray(state[:3], dtype=np.float32).copy()
        except Exception:
            return np.zeros(3)
    try:
        return get_robomimic_env(envs).get_observation()["robot0_eef_pos"].copy()
    except Exception:
        return np.zeros(3)


def render_frame(envs, config):
    """Render an RGB frame from the active env.

    For image obs: get_inner_lowdim() returns RobomimicImageWrapper, whose
    render() returns render_cache — the agentview/eye-in-hand frame cached by
    the last get_observation() call.
    """
    if _is_libero(config):
        inner = _get_libero_base(envs)
    else:
        inner = get_inner_lowdim(envs)  # RobomimicImageWrapper or RobomimicLowdimWrapper
    frame = inner.render(mode="rgb_array")

    # Robosuite / mujoco can return buffers backed by mutable internal memory.
    # Normalize and deep-copy so saved frames stay stable.
    frame = _normalize_render_frame(frame)
    if frame is None:
        return None
    return np.array(frame, copy=True)


def undo_action(action, config, dataset):
    """Undo rotation transform for absolute-action envs."""
    if getattr(config.task, "abs_action", False) and config.task.env_name in _ABS_ACTION_ENVS:
        return dataset.undo_transform_action(action[np.newaxis])[0]
    return action


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
    """Bypass ODE — encode obs then decode action with a fixed intent centroid.

    For baseline adapters: fixed_intent is a flattened action chunk.  It is
    reshaped back to (1, horizon, act_dim) and returned directly (the robot
    repeats this chunk regardless of current obs).
    """
    fi_obs, lowdim_t = to_fi_obs(obs_buf, config, dataset, device)
    if getattr(agent, "_is_baseline_adapter", False):
        # Baseline: fixed_intent is the flattened action → reshape and return directly
        horizon = config.task.horizon
        act_dim = config.task.act_dim
        act_norm = torch.tensor(
            fixed_intent, device=lowdim_t.device, dtype=torch.float32
        ).unsqueeze(0).reshape(1, horizon, act_dim)
        return dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
    intent_t = torch.tensor(fixed_intent, device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        obs_emb = agent.encoder_ema(fi_obs, None)
        if obs_emb.dim() == 2:
            obs_emb = obs_emb.unsqueeze(1)
        action_norm = agent.action_decoder_ema(obs_emb, intent_t)  # (1, horizon, act_dim)
    return dataset.normalizer["action"].unnormalize(action_norm.cpu().numpy())  # (1, H, D)


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

    while not done and total_steps < max_steps:
        action_un = decode_fixed_intent(obs_buf, fixed_intent, agent, config, dataset, device)
        for a_i in range(act_steps):
            if total_steps >= max_steps:
                break
            act = undo_action(action_un[0, s + a_i], config, dataset)
            reward, done, info = _step_inner(inner, act)
            total_reward += reward
            total_steps += 1
            if done:
                break
        if not done:
            obs_buf = update_obs_buf(obs_buf, envs, config)

    return total_reward > 0, total_steps


def run_full_episode_ode(envs, sim_state, agent, config, dataset, device, num_steps=9):
    """Restore sim_state, then run to completion with ODE-sampled intents (standard inference).

    This is the baseline — the model freely samples a new intent at every policy step
    via the flow ODE, exactly as it would during normal evaluation.
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

    while not done and total_steps < max_steps:
        fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
        with torch.no_grad():
            act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
        act_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())
        for a_i in range(act_steps):
            if total_steps >= max_steps:
                break
            act = undo_action(act_un[0, s + a_i], config, dataset)
            reward, done, info = _step_inner(inner, act)
            total_reward += reward
            total_steps += 1
            if done:
                break
        if not done:
            obs_buf = update_obs_buf(obs_buf, envs, config)

    return total_reward > 0, total_steps


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

    while not done and total_steps < max_steps:
        if not first_chunk_done:
            # Seed chunk: force the given intent
            action_un = decode_fixed_intent(obs_buf, seed_intent, agent, config, dataset, device)
            first_chunk_done = True
        else:
            # Subsequent chunks: let ODE resample intent freely
            fi_obs, _ = to_fi_obs(obs_buf, config, dataset, device)
            with torch.no_grad():
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
            action_un = dataset.normalizer["action"].unnormalize(act_norm.cpu().numpy())

        for a_i in range(act_steps):
            if total_steps >= max_steps:
                break
            act = undo_action(action_un[0, s + a_i], config, dataset)
            reward, done, info = _step_inner(inner, act)
            total_reward += reward
            total_steps += 1
            if done:
                break
        if not done:
            obs_buf = update_obs_buf(obs_buf, envs, config)

    return total_reward > 0, total_steps


# ──────────────────────────────────────────────────────────────────────────────
# Core collection
# ──────────────────────────────────────────────────────────────────────────────

def run_fixed_intent_rollout(envs, obs_buf, fixed_intent, agent, config, dataset,
                              device, n_future_steps, capture_render, num_steps=9):
    """Step env for n_future_steps with a fixed intent centroid.

    Uses the inner single-step env directly so that:
    - Actions are applied one-at-a-time (not as chunks)
    - MultiStepWrapper's stale done-state from prior episodes is bypassed
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

    for step_i in range(0, n_future_steps, act_steps):
        action_un = decode_fixed_intent(obs_buf, fixed_intent, agent, config, dataset, device)
        remaining = n_future_steps - step_i
        for a_i in range(min(act_steps, remaining)):
            act = undo_action(action_un[0, s + a_i], config, dataset)
            _, done, info = _step_inner(inner, act)
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
        "success": bool(info.get("success", False)),  # last-step only (used for viz label)
        "n_steps": len(eef_traj) - 1,
    }


def collect_outcomes(config, agent, dataset, envs, args, device):
    """Main loop: find critical states + collect k-cluster fixed-intent outcomes."""
    s = config.task.obs_steps - 1
    act_steps = config.task.act_steps
    num_steps = 9

    # ── Phase 1: probe rollouts to find high-variance states ──────────────────
    print(f"Probing {args.n_rollouts} rollouts (n_probe={args.n_probe} per state)...")
    all_candidates = []
    n_done = 0

    while n_done < args.n_rollouts:
        obs, _ = envs.reset()
        t = 0

        while t < config.task.max_episode_steps:
            fi_obs, _ = to_fi_obs(obs, config, dataset, device)

            # Probe variance
            probe_intents = []
            with torch.no_grad():
                for _ in range(args.n_probe):
                    _, iv = agent.sample(obs=fi_obs, use_ema=True,
                                          num_steps=num_steps, return_intent=True)
                    probe_intents.append(iv[0].cpu().numpy())
            variance = float(np.stack(probe_intents).var(axis=0).mean())

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
            with torch.no_grad():
                act_norm = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
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
    critical = all_candidates[:args.n_critical]
    print(f"\nTop {len(critical)} critical states selected.")

    # ── Phase 2: collect fixed-intent outcomes for each critical state ─────────
    outcomes = []
    for ci, state in enumerate(critical):
        print(f"\nCritical state {ci+1}/{len(critical)}  "
              f"t={state['timestep']}  var={state['variance']:.5f}")

        obs_raw = state["obs_raw"]

        # Sample n_intents intents
        intents = sample_intents(obs_raw, agent, config, dataset, device,
                                  n=args.n_intents, num_steps=num_steps)
        print(f"  Sampled {len(intents)} intents, shape={intents.shape}")

        # K-means cluster
        n_clusters = min(args.k_clusters, len(intents))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(intents)
        centroids = km.cluster_centers_  # (k, intent_dim)

        # Roll out from each centroid with fixed intent
        cluster_outcomes = []
        for ki in range(n_clusters):
            print(f"  Cluster {ki+1}/{n_clusters} (intent={centroids[ki][:3].round(3)})")
            obs_buf = restore_env_obs(
                envs,
                state["sim_state"],
                config,
                ensure_fresh_reset=bool(args.render),
            )
            result = run_fixed_intent_rollout(
                envs, obs_buf, centroids[ki], agent, config, dataset, device,
                n_future_steps=args.n_future_steps,
                capture_render=args.render,
                num_steps=num_steps,
            )
            result["intent"] = centroids[ki].copy()
            cluster_outcomes.append(result)
            print(f"    → {result['n_steps']} steps, success={result['success']}, "
                  f"eef_end={result['eef_trajectory'][-1].round(3)}")

        # ── Phase 2b: success-rate trials (if requested) ──────────────────────
        sr_trials = {}
        if args.n_trials > 0:
            intent_dim = intents.shape[1]
            rng = np.random.default_rng(42 + ci)

            print(f"  Running {args.n_trials} full-episode trials per condition "
                  f"(seed-then-ODE)...")
            # Cluster centroids: force intent for ONE chunk, then ODE takes over
            for ki in range(n_clusters):
                successes = []
                for trial in range(args.n_trials):
                    ok, nsteps = run_full_episode_intent_seed(
                        envs, state["sim_state"], centroids[ki],
                        agent, config, dataset, device, num_steps=num_steps)
                    successes.append(ok)
                sr_trials[f"cluster_{ki}"] = successes
                print(f"    cluster_{ki}: {sum(successes)}/{args.n_trials} successes")

            # ODE baseline — model freely samples intent for ALL chunks (full ODE)
            ode_successes = []
            for trial in range(args.n_trials):
                ok, nsteps = run_full_episode_ode(
                    envs, state["sim_state"], agent, config, dataset, device,
                    num_steps=num_steps)
                ode_successes.append(ok)
            sr_trials["ode"] = ode_successes
            print(f"    ode (full):    {sum(ode_successes)}/{args.n_trials} successes")

            # Random Gaussian seed intent: one bad chunk, then ODE
            rand_successes = []
            for trial in range(args.n_trials):
                rand_intent = rng.standard_normal(intent_dim).astype(np.float32)
                ok, nsteps = run_full_episode_intent_seed(
                    envs, state["sim_state"], rand_intent,
                    agent, config, dataset, device, num_steps=num_steps)
                rand_successes.append(ok)
            sr_trials["random"] = rand_successes
            print(f"    random seed:   {sum(rand_successes)}/{args.n_trials} successes")

        outcomes.append({
            "obs_raw": obs_raw,
            "sim_state": state["sim_state"],
            "eef_pos": state["eef_pos"],
            "timestep": state["timestep"],
            "variance": state["variance"],
            "intents": intents,
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

    Matches training targets where intent begins with EE position (e.g. LIBERO
    ``ee_states``: pos3 + axis-angle3 — only position is drawn here).
    """
    if intent_vec is None:
        return None
    z = np.asarray(intent_vec, dtype=np.float64).reshape(-1)
    if z.size < 3:
        return None
    return z[:3]


def fig_eef_trajectories(
    outcomes,
    out_dir,
    task_name,
    obs_type,
    show_intent_position=False,
    figure_filename="eef_trajectories.png",
):
    """Plot EEF trajectories in 3D per critical state, colored by intent cluster."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(outcomes)
    if n == 0:
        return
    k = len(outcomes[0]["cluster_outcomes"])
    cluster_colors = plt.cm.tab10(np.arange(k) / max(k - 1, 1))

    fig = plt.figure(figsize=(4.5 * n, 4.5))
    axes = [fig.add_subplot(1, n, ci + 1, projection="3d") for ci in range(n)]

    for ci, (outcome, ax) in enumerate(zip(outcomes, axes)):
        start_eef = outcome["eef_pos"]
        ts = outcome["timestep"]
        var = outcome["variance"]
        xyz_points = [start_eef]

        # Starting position
        ax.scatter(*start_eef, c="black", s=80, marker="o", depthshade=False,
                   label="start" if ci == 0 else None, zorder=10)

        intent_legend_done = False
        for ki, co in enumerate(outcome["cluster_outcomes"]):
            traj = co["eef_trajectory"]   # (T, 3)
            xyz_points.append(traj)
            c = cluster_colors[ki]
            label = f"Cluster {ki+1}" if ci == 0 else None
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=c, alpha=0.85, lw=1.8, label=label)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                       color=c, s=80, marker="*", depthshade=False, zorder=5)
            if show_intent_position:
                pz = _intent_xyz_for_plot(co.get("intent"))
                if pz is not None:
                    xyz_points.append(pz)
                    ilab = "intent z[:3]" if (ci == 0 and not intent_legend_done) else None
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

        # Keep all axes at comparable scale so 3D geometry is not visually distorted.
        xyz = np.vstack(xyz_points)
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        center = 0.5 * (xyz_min + xyz_max)
        max_range = float(np.max(xyz_max - xyz_min))
        if max_range < 1e-6:
            max_range = 1e-3
        half = 0.55 * max_range
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.view_init(elev=25, azim=-55)

        ax.set_xlabel("X (m)", fontsize=7, labelpad=2)
        ax.set_ylabel("Y (m)", fontsize=7, labelpad=2)
        ax.set_zlabel("Z (m)", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.set_title(f"S{ci}  t={ts}\nvar={var:.4f}", fontsize=8)
        if ci == 0:
            ax.legend(fontsize=7, loc="upper left")

    sub = "(●=start  ★=end"
    if show_intent_position:
        sub += "  ^=centroid intent, z[:3] (target EE pos in training space)"
    sub += ")"
    fig.suptitle(
        f"[{task_name}] EEF trajectories under fixed intent clusters\n{sub}",
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
    success_only=False,
    success_threshold=0.5,
    show_intent_position=False,
    figure_filename="eef_trajectories_with_frames.png",
):
    """Combined view: 3D EEF trajectory + rendered start/end frames per state."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(outcomes)
    if n == 0:
        return
    k = len(outcomes[0]["cluster_outcomes"])
    cluster_colors = plt.cm.tab10(np.arange(k) / max(k - 1, 1))
    cluster_indices_per_state = [
        _select_cluster_indices(o, success_only=success_only, success_threshold=success_threshold)
        for o in outcomes
    ]
    max_cols_clusters = max((len(v) for v in cluster_indices_per_state), default=0)
    if max_cols_clusters == 0:
        print("No successful clusters to render; skipping eef_trajectories_with_frames.png")
        return
    n_cols = max_cols_clusters + 2  # 3D traj + start + selected cluster ends

    fig = plt.figure(figsize=(3.2 * n_cols, 3.2 * n))

    for ci, (outcome, keep_indices) in enumerate(zip(outcomes, cluster_indices_per_state)):
        ts = outcome["timestep"]
        var = outcome["variance"]
        start_eef = outcome["eef_pos"]
        xyz_points = [start_eef]

        # Column 0: 3D EEF trajectories
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
            xyz_points.append(traj)
            c = cluster_colors[ki]
            label = f"C{ki+1}" if ci == 0 else None
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
            if show_intent_position:
                pz = _intent_xyz_for_plot(co.get("intent"))
                if pz is not None:
                    xyz_points.append(pz)
                    ilab = "intent z[:3]" if (ci == 0 and not intent_legend_done) else None
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

        xyz = np.vstack(xyz_points)
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        center = 0.5 * (xyz_min + xyz_max)
        max_range = float(np.max(xyz_max - xyz_min))
        if max_range < 1e-6:
            max_range = 1e-3
        half = 0.55 * max_range
        ax3d.set_xlim(center[0] - half, center[0] + half)
        ax3d.set_ylim(center[1] - half, center[1] + half)
        ax3d.set_zlim(center[2] - half, center[2] + half)
        ax3d.view_init(elev=25, azim=-55)
        ax3d.set_xlabel("X", fontsize=7, labelpad=1)
        ax3d.set_ylabel("Y", fontsize=7, labelpad=1)
        ax3d.set_zlabel("Z", fontsize=7, labelpad=1)
        ax3d.tick_params(labelsize=6)
        ax3d.set_title(f"S{ci} traj\nt={ts} var={var:.4f}", fontsize=8)
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
        legend_bits += "  ^=intent z[:3]"
    fig.suptitle(
        f"[{task_name}] Trajectories + rendered outcomes per intent cluster\n"
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
    """Bar chart of success rate per intent condition, one panel per critical state.

    Conditions:
      cluster_0 … cluster_{k-1}  — fixed centroid intent
      ode                         — standard ODE inference (GT probe)
      random                      — random Gaussian intent (sanity check)
    """
    # Only include outcomes that have trial data
    outcomes_with_sr = [o for o in outcomes if o.get("sr_trials")]
    if not outcomes_with_sr:
        return

    n = len(outcomes_with_sr)
    cluster_keys = [f"cluster_{ki}" for ki in range(k_clusters)]
    all_conditions = cluster_keys + ["ode", "random"]
    cond_labels = [f"C{ki+1}" for ki in range(k_clusters)] + ["ODE\n(GT)", "Random"]

    # Colors: tab10 for clusters, dark gray for ODE, light gray for random
    tab10 = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = [tab10[ki] for ki in range(k_clusters)] + \
             [np.array([0.3, 0.3, 0.3, 1.0]), np.array([0.7, 0.7, 0.7, 1.0])]

    fig, axes = plt.subplots(1, n, figsize=(max(4, 2.5 * len(all_conditions)) * n, 4),
                              sharey=True)
    if n == 1:
        axes = [axes]

    for ci, outcome in enumerate(outcomes_with_sr):
        ax = axes[ci]
        sr = outcome["sr_trials"]
        ts = outcome["timestep"]
        var = outcome["variance"]
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
        ax.set_title(f"S{ci}  t={ts}\nvar={var:.4f}", fontsize=8)
        if ci == 0:
            ax.set_ylabel("Success Rate", fontsize=9)
        ax.axhline(0.5, color="black", lw=0.6, ls="--", alpha=0.4)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"[{task_name}] Success rate per intent condition  (n={n_trials} trials each)\n"
        "C1…Ck = cluster-centroid seed for ONE chunk, then ODE  |  "
        "ODE = standard inference (free every chunk)  |  Random = N(0,I) seed then ODE",
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
                        help="Intent samples per state for variance probing")
    parser.add_argument("--n-intents", type=int, default=50,
                        help="Intent samples at critical states (for clustering)")
    parser.add_argument("--k-clusters", type=int, default=3,
                        help="Number of intent clusters")
    parser.add_argument("--n-future-steps", type=int, default=80,
                        help="Steps to run forward with fixed intent")
    parser.add_argument("--n-trials", type=int, default=0,
                        help="Full-episode trials per intent condition for success-rate "
                             "evaluation. 0 = skip. Runs k_clusters + ODE + random "
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
    maybe_register_libero_pro_objects(config)

    # Setup env + dataset
    envs = make_vec_env(config, seed=args.seed)
    setup_config_for_env(config, envs)
    dataset = make_dataset(config)

    # Load model
    agent, _ = load_model(ckpt_path, config, dataset, args.device)
    agent.eval()

    # Wrap baseline (TrainingAgent) to expose FlowIntentAgent-compatible interface.
    # The "intent" for a baseline is the sampled action (flattened); clustering it
    # reveals whether the model exhibits multi-modal action distributions.
    if isinstance(agent, TrainingAgent):
        agent = BaselineAgentAdapter(agent, config)

    print(f"Task: {config.task.env_name}  obs_type: {config.task.obs_type}  "
          f"intent_dim: {getattr(config.task, 'intent_dim', '?')}")

    # Collect
    outcomes = collect_outcomes(config, agent, dataset, envs, args, args.device)

    # Save pkl
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task": config.task.env_name,
        "obs_type": config.task.obs_type,
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
    fig_eef_trajectories(
        outcomes,
        out_dir,
        task_name,
        config.task.obs_type,
        show_intent_position=False,
    )
    if args.plot_intent_xyz:
        fig_eef_trajectories(
            outcomes,
            out_dir,
            task_name,
            config.task.obs_type,
            show_intent_position=True,
            figure_filename="eef_trajectories_with_intent_xyz.png",
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
            success_only=args.render_success_only,
            success_threshold=args.success_threshold,
            show_intent_position=False,
        )
        if args.plot_intent_xyz:
            fig_trajectories_with_frames(
                outcomes,
                out_dir,
                task_name,
                success_only=args.render_success_only,
                success_threshold=args.success_threshold,
                show_intent_position=True,
                figure_filename="eef_trajectories_with_frames_intent_xyz.png",
            )

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
