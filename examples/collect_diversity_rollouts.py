"""Stage 1: Roll out policy variants and save action chunks + visited states to a .pkl file.

Rolls out N episodes per variant, collecting predicted action chunks at every
timestep. Also gathers steerability data for flow_intent variants (K independent
intent samples from the same observation).

Usage:
    python examples/collect_diversity_rollouts.py \
        --task lift_ph \
        --run "baseline:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_success0.pt:task=lift_ph_state" \
        --run "flow_intent:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt:task=lift_ph_state_flow_intent:network.arch_variant=flow_intent" \
        --run "hierarchical_emb:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_intent_learned_joint_emb_success98.pt:task=lift_ph_state_hierarchical_emb" \
        --n-rollouts 100 \
        --device cuda \
        --out rollouts/lift_ph_diversity.pkl

Each --run is: "label:ckpt_path:hydra_override1:hydra_override2:..."
The task= override selects the Hydra task config (e.g. task=lift_ph_state_flow_intent).
Additional overrides follow standard Hydra syntax (key=value).

Output .pkl structure:
    {
      "<task>": {
        "<label>": {
          "action_chunks": np.ndarray (N_total, flat_dim),  # flattened per chunk
          "success":       list[bool],                       # per episode
          "obs_states":    np.ndarray (N_steps_total, obs_dim),  # raw env states per step
          # flow_intent only:
          "steer_actions": np.ndarray (N_eps, K, flat_dim),
          "steer_intents": np.ndarray (N_eps, K, intent_dim),
        },
        "gt_demos": {
          "action_chunks": np.ndarray (M, flat_dim),
          "obs_states":    np.ndarray (M, obs_dim),  # demo states from dataset
        },
      }
    }
"""

import argparse
import os
import pickle
import resource
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

# Raise fd limit as high as the hard limit allows (bash ulimit may be capped lower).
_fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
resource.setrlimit(resource.RLIMIT_NOFILE, (_fd_hard, _fd_hard))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

warnings.filterwarnings("ignore")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tensordict import TensorDict
from mip.agent import TrainingAgent
from mip.flow_intent_agent import FlowIntentAgent
from mip.intent_predictor import IntentPredictor
from mip.datasets.robomimic_dataset import make_dataset as make_dataset_robomimic
from mip.envs.robomimic.robomimic_env import make_vec_env as make_vec_env_robomimic
from mip.datasets.kitchen_dataset import make_dataset as make_dataset_kitchen
from mip.envs.kitchen import make_vec_env as make_vec_env_kitchen
from mip.datasets.libero_dataset import make_dataset as make_dataset_libero
from mip.envs.libero import make_vec_env as make_vec_env_libero
from mip.torch_utils import set_seed
from mip.residual_parl.parl_agent import ResidualPARLAgent, PARLConfig
from mip.dsrl.dsrl_agent import DSRLSACAgent, DSRLConfig
from mip.residual_sac.sac_agent import ResidualSACAgent, SACConfig

# On compute nodes, Python's import system may resolve .venv/lib64 (a lib64->lib symlink)
# as the canonical path, so robosuite.__file__ contains "lib64".  edit_model_xml() in
# robosuite/environments/base.py rebuilds all mesh/texture paths from robosuite.__file__,
# so every asset ends up with a lib64 prefix that MuJoCo's C resolver can't follow on NFS.
#
# Fix 1: redirect robosuite.__file__ via realpath (works if lib64->lib resolves on the node).
import robosuite as _rs
_rs.__file__ = os.path.realpath(_rs.__file__)

# Fix 2: belt-and-suspenders — patch edit_model_xml to replace any residual lib64 prefix.
import robosuite.environments.base as _rsb
_venv_lib64 = str(ROOT / ".venv" / "lib64")
_venv_lib   = str(ROOT / ".venv" / "lib")
_EnvCls = getattr(_rsb, "EnvBase", None) or _rsb.MujocoEnv
_orig_edit_xml = _EnvCls.edit_model_xml
def _patched_edit_xml(self, xml_str):
    xml_str = _orig_edit_xml(self, xml_str)
    return xml_str.replace(_venv_lib64 + "/", _venv_lib + "/")
_EnvCls.edit_model_xml = _patched_edit_xml


def _is_kitchen(task_config):
    return "kitchen" in getattr(task_config, "env_name", "")


def _is_libero(task_config):
    return getattr(task_config, "env_name", "").startswith("libero")


def make_vec_env(task_config, seed=0):
    if _is_libero(task_config):
        return make_vec_env_libero(task_config, seed=seed)
    if _is_kitchen(task_config):
        return make_vec_env_kitchen(task_config, seed=seed)
    return make_vec_env_robomimic(task_config, seed=seed)


def make_dataset(task_config):
    if _is_libero(task_config):
        return make_dataset_libero(task_config)
    if _is_kitchen(task_config):
        return make_dataset_kitchen(task_config)
    return make_dataset_robomimic(task_config)


STEER_K = 10  # intent samples for steerability


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config(overrides: list[str]):
    """Load Hydra config from a flat list of override strings (e.g. ['task=lift_ph_state', 'task.num_envs=1'])."""
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/configs"), version_base=None
    ):
        cfg = compose("main", overrides=overrides)
    return cfg


def parse_run_spec(run_spec: str):
    """Parse a --run argument of the form 'label:ckpt_path:override1:override2:...'

    Returns:
        label       : str
        ckpt_path   : str
        overrides   : list[str]
    """
    parts = run_spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"--run must be 'label:ckpt_path[:override ...]', got: {run_spec}")
    label = parts[0]
    ckpt_path = parts[1]
    overrides = parts[2:]
    return label, ckpt_path, overrides


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def setup_config_for_env(config, envs):
    """Mirror what train_robomimic.py:main() does to set obs_dim at runtime."""
    obs, _ = envs.reset()
    arch_variant = getattr(config.network, "arch_variant", "flow_action")

    if config.task.obs_type == "state":
        base_obs_dim = obs.shape[-1]
        config.task.obs_dim = base_obs_dim

        if getattr(config.task, "intent_conditioning", False):
            intent_type = getattr(config.task, "intent_type", "mean")
            if intent_type == "sequence":
                config.task.intent_dim = config.task.intent_horizon * 7

            if arch_variant != "flow_intent":
                _cond_dim = (
                    getattr(config.task, "intent_emb_dim", 64)
                    if intent_type == "encoded_mean"
                    else config.task.intent_dim
                )
                config.task.obs_dim = base_obs_dim + _cond_dim
    else:
        config.task.obs_dim = config.network.emb_dim
        if (
            _is_libero(config.task)
            and getattr(config.task, "intent_conditioning", False)
            and arch_variant != "flow_intent"
            and isinstance(obs, dict)
            and "state" in obs
            and "shape_meta" in config.task
            and "state" in config.task.shape_meta["obs"]
        ):
            intent_type = getattr(config.task, "intent_type", "mean")
            cond_dim = (
                getattr(config.task, "intent_emb_dim", 64)
                if intent_type == "encoded_mean"
                else config.task.intent_dim
            )
            base_state_dim = int(obs["state"].shape[-1])
            config.task.shape_meta["obs"]["state"]["shape"] = [base_state_dim + cond_dim]

    return obs  # the reset obs, reuse to avoid re-reset


def _patch_config_from_checkpoint(checkpoint_path: str, config):
    """Peek at checkpoint weights to infer obs_dim/horizon/obs_steps and patch config.

    This handles cases where the checkpoint was trained with different
    hyperparameters than what the yaml config specifies (e.g. h6 vs h16).
    """
    import torch as _torch
    sd = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch_variant = getattr(config.network, "arch_variant", "flow_action")

    if arch_variant == "flow_intent":
        # FlowIntentAgent has a separate flow_map for intent — skip for now
        return

    fm = sd.get("flow_map", {})
    out_w = fm.get("net.main_output.weight")  # (act_dim * Ta, emb_dim)
    in_w = fm.get("net.input_proj.weight")    # (emb_dim, input_dim)
    if out_w is None or in_w is None:
        return

    act_dim = config.task.act_dim
    Ta = out_w.shape[0] // act_dim  # infer horizon from output shape
    input_dim = in_w.shape[1]

    # Check if VanillaMLP (has Fourier frequencies) or plain MLP (+2 for s,t)
    freq = fm.get("net.frequencies")
    if freq is not None:
        timestep_emb_dim = freq.shape[0] * 2  # num_frequencies = timestep_emb_dim // 2
        time_contribution = 2 * timestep_emb_dim
    else:
        time_contribution = 2  # plain MLP: scalar s and t

    obs_flat = input_dim - act_dim * Ta - time_contribution

    # obs_flat = emb_dim * To  (get_network uses emb_dim as per-step obs_dim)
    # emb_dim is the network hidden dim = encoder output per step
    emb_dim = in_w.shape[0]  # flow network hidden dim (= emb_dim in config)
    if obs_flat % emb_dim == 0:
        To = obs_flat // emb_dim
        config.task.obs_steps = To
    # Don't touch network.emb_dim — it controls hidden dim, not just obs_dim

    config.task.horizon = Ta
    # Don't override act_steps — it's a deployment choice set in the yaml config,
    # not an architecture constraint inferred from weights.


class ResidualPARLWrapper:
    """Wraps FlowIntentAgent + ResidualPARLAgent behind the FlowIntentAgent .sample() interface.

    Sampling path (mirrors is_flow_intent=True):
        1. flow_intent.sample(obs, return_intent=True) → (base_actions, intent_vec)
        2. parl.sample_action(obs_flat, base_chunk_flat, deterministic=True) → a_exec per env
        3. Compose: replace act_steps slice in base_actions with PARL output, return full horizon tensor.
    """

    def __init__(self, flow_intent: FlowIntentAgent, parl: ResidualPARLAgent, config):
        self.flow_intent = flow_intent
        self.parl = parl
        self.config = config

    def eval(self):
        self.flow_intent.eval()
        return self

    def sample_intent(self, obs=None, use_ema: bool = True, num_steps: int = -1, **kwargs):
        """Delegate intent sampling to the underlying FlowIntentAgent."""
        return self.flow_intent.sample_intent(obs=obs, use_ema=use_ema, num_steps=num_steps, **kwargs)

    def sample_given_intent(self, obs=None, intent_vec=None, use_ema: bool = True,
                            num_steps: int = -1, **kwargs):
        """Run flow-intent decoder with fixed intent, then apply PARL refinement."""
        base_actions = self.flow_intent.sample_given_intent(
            obs=obs, intent_vec=intent_vec, use_ema=use_ema, num_steps=num_steps, **kwargs
        )
        # Apply PARL residual on top of the committed-intent base actions
        B = base_actions.shape[0]
        s = self.config.task.obs_steps - 1
        act_steps = self.config.task.act_steps
        base_chunk = base_actions[:, s:s + act_steps, :]
        base_flat_np = base_chunk.reshape(B, -1).cpu().numpy()
        obs_flat_np = obs.reshape(B, -1).cpu().numpy()
        composed_list = []
        for i in range(B):
            a_exec, _, _ = self.parl.sample_action(
                obs_flat_np[i], base_flat_np[i], deterministic=True
            )
            composed_list.append(a_exec)
        composed_flat = np.stack(composed_list)
        composed_chunk = torch.tensor(
            composed_flat, device=base_actions.device, dtype=base_actions.dtype
        ).reshape(B, act_steps, -1)
        composed_full = base_actions.clone()
        composed_full[:, s:s + act_steps, :] = composed_chunk
        return composed_full

    def sample(self, obs=None, use_ema: bool = True, num_steps: int = -1,
               return_intent: bool = False, **kwargs):
        """obs: (B, obs_steps, obs_dim) tensor — same as FlowIntentAgent.sample()."""
        base_actions, intent_vec = self.flow_intent.sample(
            obs=obs, use_ema=use_ema, num_steps=num_steps, return_intent=True
        )
        # base_actions: (B, horizon, act_dim) normalized

        B = base_actions.shape[0]
        s = self.config.task.obs_steps - 1
        act_steps = self.config.task.act_steps

        # Extract act_steps chunk and flatten for PARL
        base_chunk = base_actions[:, s:s + act_steps, :]           # (B, act_steps, act_dim)
        base_flat_np = base_chunk.reshape(B, -1).cpu().numpy()      # (B, act_steps*act_dim)
        obs_flat_np = obs.reshape(B, -1).cpu().numpy()              # (B, obs_steps*obs_dim)

        composed_list = []
        for i in range(B):
            a_exec, _, _ = self.parl.sample_action(
                obs_flat_np[i], base_flat_np[i], deterministic=True
            )
            composed_list.append(a_exec)
        composed_flat = np.stack(composed_list)                     # (B, act_steps*act_dim)

        composed_chunk = torch.tensor(
            composed_flat, device=base_actions.device, dtype=base_actions.dtype
        ).reshape(B, act_steps, -1)                                 # (B, act_steps, act_dim)

        # Embed back into full horizon
        composed_full = base_actions.clone()
        composed_full[:, s:s + act_steps, :] = composed_chunk

        if return_intent:
            return composed_full, intent_vec
        return composed_full


class DSRLWrapper:
    """Wraps FlowIntentAgent + DSRLSACAgent. DSRL actor picks intent ODE noise x_0.

    Each sample_intent call draws a stochastic x_0 from the DSRL actor, runs the
    intent ODE from that noise, and returns the resulting intent_vec — giving each
    ghost rollout a different DSRL-steered intent.
    """

    def __init__(self, flow_intent: FlowIntentAgent, dsrl: DSRLSACAgent, config):
        self.flow_intent = flow_intent
        self.dsrl = dsrl
        self.config = config

    def eval(self):
        self.flow_intent.eval()
        return self

    def sample_intent(self, obs=None, use_ema: bool = True, num_steps: int = -1, **kwargs):
        B = obs.shape[0]
        obs_flat_np = obs.reshape(B, -1).cpu().numpy()
        noises = []
        for i in range(B):
            x0_np, _ = self.dsrl.sample_noise(obs_flat_np[i], deterministic=False)
            noises.append(x0_np)
        intent_noise = torch.tensor(
            np.stack(noises), device=obs.device, dtype=obs.dtype
        ).unsqueeze(1)  # (B, 1, intent_dim)
        return self.flow_intent.sample_intent_from_noise(obs, intent_noise, use_ema=use_ema, num_steps=num_steps)

    def sample_given_intent(self, obs=None, intent_vec=None, use_ema: bool = True,
                            num_steps: int = -1, **kwargs):
        return self.flow_intent.sample_given_intent(
            obs=obs, intent_vec=intent_vec, use_ema=use_ema, num_steps=num_steps
        )

    def sample(self, obs=None, use_ema: bool = True, num_steps: int = -1,
               return_intent: bool = False, **kwargs):
        intent_vec = self.sample_intent(obs=obs, use_ema=use_ema, num_steps=num_steps)
        actions = self.sample_given_intent(obs=obs, intent_vec=intent_vec, use_ema=use_ema, num_steps=num_steps)
        if return_intent:
            return actions, intent_vec
        return actions


class PlainDSRLWrapper:
    """Wraps TrainingAgent + DSRLSACAgent for plain (non-intent) flow DSRL.

    DSRL actor picks action ODE noise x_0; each ghost gets a different noise sample.
    """

    def __init__(self, flow_agent: TrainingAgent, dsrl: DSRLSACAgent, config):
        self.flow_agent = flow_agent
        self.dsrl = dsrl
        self.config = config

    def eval(self):
        self.flow_agent.eval()
        return self

    def sample(self, obs=None, use_ema: bool = True, num_steps: int = -1, **kwargs):
        obs_tensor = obs["state"] if isinstance(obs, dict) else obs
        B = obs_tensor.shape[0]
        obs_flat_np = obs_tensor.reshape(B, -1).cpu().numpy()
        noises = []
        for i in range(B):
            x0_np, _ = self.dsrl.sample_noise(obs_flat_np[i], deterministic=False)
            noises.append(x0_np)
        # x0 shape: (B, Ta * act_dim) → (B, Ta, act_dim)
        Ta = self.config.task.horizon
        act_dim = self.config.task.act_dim
        act_0 = torch.tensor(
            np.stack(noises), device=obs_tensor.device, dtype=obs_tensor.dtype
        ).reshape(B, Ta, act_dim)
        return self.flow_agent.sample(obs=obs, use_ema=use_ema, num_steps=num_steps, act_0=act_0)


class ResidualSACWrapper:
    """Wraps FlowIntentAgent + ResidualSACAgent. Intent from flow, action refined by SAC."""

    def __init__(self, flow_intent: FlowIntentAgent, sac: ResidualSACAgent, config):
        self.flow_intent = flow_intent
        self.sac = sac
        self.config = config

    def eval(self):
        self.flow_intent.eval()
        return self

    def sample_intent(self, obs=None, use_ema: bool = True, num_steps: int = -1, **kwargs):
        return self.flow_intent.sample_intent(obs=obs, use_ema=use_ema, num_steps=num_steps)

    def sample_given_intent(self, obs=None, intent_vec=None, use_ema: bool = True,
                            num_steps: int = -1, **kwargs):
        base_actions = self.flow_intent.sample_given_intent(
            obs=obs, intent_vec=intent_vec, use_ema=use_ema, num_steps=num_steps
        )
        B = base_actions.shape[0]
        s = self.config.task.obs_steps - 1
        act_steps = self.config.task.act_steps
        base_chunk = base_actions[:, s:s + act_steps, :]
        base_flat_np = base_chunk.reshape(B, -1).cpu().numpy()
        obs_flat_np = obs.reshape(B, -1).cpu().numpy()
        composed_list = []
        for i in range(B):
            a_exec, _, _ = self.sac.sample_action(obs_flat_np[i], base_flat_np[i], deterministic=True)
            composed_list.append(a_exec)
        composed_flat = np.stack(composed_list)
        composed_chunk = torch.tensor(
            composed_flat, device=base_actions.device, dtype=base_actions.dtype
        ).reshape(B, act_steps, -1)
        composed_full = base_actions.clone()
        composed_full[:, s:s + act_steps, :] = composed_chunk
        return composed_full

    def sample(self, obs=None, use_ema: bool = True, num_steps: int = -1,
               return_intent: bool = False, **kwargs):
        intent_vec = self.sample_intent(obs=obs, use_ema=use_ema, num_steps=num_steps)
        actions = self.sample_given_intent(obs=obs, intent_vec=intent_vec, use_ema=use_ema, num_steps=num_steps)
        if return_intent:
            return actions, intent_vec
        return actions


def load_model(checkpoint_path: str, config, dataset, device: str,
               flow_intent_ckpt: str | None = None,
               agent_type: str | None = None):
    """Load agent and (if needed) intent predictor from checkpoint.

    Returns:
        agent            : TrainingAgent or FlowIntentAgent
        intent_predictor : IntentPredictor | None

    Note: return_intent=True is only used on FlowIntentAgent (Config A).
    TrainingAgent (Config B) does not support it and is never called with it —
    steer collection is gated behind `is_flow_intent` in collect_rollouts().

    For encoded_mean intent (Config B) with a learned predictor, we also load
    the _hl_policy.pt sidecar checkpoint if it exists next to the main ckpt.
    For image obs the CV proxy is not valid for encoded_mean, so we raise if
    the predictor is required but the sidecar is missing.
    """
    # Detect RL-style checkpoints (PARL/DSRL/SAC) by state-dict keys.
    _probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(_probe, dict) and "actor" in _probe and "critic" in _probe:
        if flow_intent_ckpt is None:
            raise ValueError(
                "RL checkpoint detected but flow_intent_ckpt not provided. "
                "Pass flow_intent_ckpt=... in the run spec. "
                "(For plain_dsrl, pass the baseline flow checkpoint as flow_intent_ckpt.)"
            )

        _has_log_alpha = "log_alpha" in _probe  # DSRL/SAC have it; PARL does not

        obs_dim_per_step = dataset[0]["obs"]["state"].shape[-1]
        obs_dim_flat = config.task.obs_steps * obs_dim_per_step
        act_dim = config.task.act_dim
        act_steps = config.task.act_steps

        if agent_type in ("dsrl", "plain_dsrl"):
            is_intent = agent_type == "dsrl"
            if is_intent:
                _patch_config_from_checkpoint(flow_intent_ckpt, config)
                fi_agent = FlowIntentAgent(config)
                fi_agent.load(flow_intent_ckpt, load_optimizer=False)
                fi_agent.eval()
                noise_dim = config.task.intent_dim
            else:
                _patch_config_from_checkpoint(flow_intent_ckpt, config)
                flow_agent = TrainingAgent(config)
                flow_agent.load(flow_intent_ckpt, load_optimizer=False)
                flow_agent.eval()
                noise_dim = config.task.horizon * act_dim
            dsrl_cfg = DSRLConfig(device=device)
            dsrl_agent = DSRLSACAgent(config=dsrl_cfg, obs_dim=obs_dim_flat, Ta=1, act_dim=noise_dim)
            dsrl_agent.load(checkpoint_path)
            if is_intent:
                return DSRLWrapper(fi_agent, dsrl_agent, config), None
            else:
                return PlainDSRLWrapper(flow_agent, dsrl_agent, config), None

        if agent_type == "residual_sac":
            _patch_config_from_checkpoint(flow_intent_ckpt, config)
            fi_agent = FlowIntentAgent(config)
            fi_agent.load(flow_intent_ckpt, load_optimizer=False)
            fi_agent.eval()
            sac_cfg = SACConfig(device=device)
            sac_agent = ResidualSACAgent(sac_cfg, obs_dim=obs_dim_flat, act_dim=act_dim, query_freq=act_steps)
            sac_agent.load(checkpoint_path)
            return ResidualSACWrapper(fi_agent, sac_agent, config), None

        # Default: PARL (no log_alpha)
        _patch_config_from_checkpoint(flow_intent_ckpt, config)
        fi_agent = FlowIntentAgent(config)
        fi_agent.load(flow_intent_ckpt, load_optimizer=False)
        fi_agent.eval()
        parl_config = PARLConfig(device=device)
        parl_agent = ResidualPARLAgent(
            parl_config, obs_dim=obs_dim_flat, act_dim=act_dim, query_freq=act_steps
        )
        parl_agent.set_flow_intent(fi_agent)
        parl_agent.load(checkpoint_path)
        wrapper = ResidualPARLWrapper(fi_agent, parl_agent, config)
        return wrapper, None

    _patch_config_from_checkpoint(checkpoint_path, config)
    arch_variant = getattr(config.network, "arch_variant", "flow_action")
    if arch_variant == "flow_intent":
        agent = FlowIntentAgent(config)
    else:
        agent = TrainingAgent(config)
    agent.load(checkpoint_path, load_optimizer=False)
    agent.eval()

    intent_predictor = None
    needs_predictor = (
        getattr(config.task, "intent_conditioning", False)
        and getattr(config.task, "intent_predictor", False)
        and arch_variant != "flow_intent"
    )
    if needs_predictor:
        # Determine predictor input dim: for image obs use lowdim keys; for state use base obs dim
        if config.task.obs_type == "image":
            _base_obs_dim = sum(
                config.task.shape_meta["obs"][k]["shape"][0]
                for k in dataset.lowdim_keys
            )
            # For PushT image, intent is concatenated into agent_pos in shape_meta ([4] not [2]).
            # Subtract intent_dim to get the raw low-dim size that the predictor expects.
            if getattr(config.task, "env_name", "") == "pusht":
                _base_obs_dim -= config.task.intent_dim
        else:
            # base_obs_dim already set on config by setup_config_for_env
            _base_obs_dim = config.task.obs_dim - (
                getattr(config.task, "intent_emb_dim", 64)
                if getattr(config.task, "intent_type", "mean") == "encoded_mean"
                else config.task.intent_dim
            )
        _pred_out_dim = (
            getattr(config.task, "intent_emb_dim", 64)
            if getattr(config.task, "intent_type", "mean") == "encoded_mean"
            else config.task.intent_dim
        )
        intent_predictor = IntentPredictor(
            obs_steps=config.task.obs_steps,
            base_obs_dim=_base_obs_dim,
            intent_dim=_pred_out_dim,
        ).to(device)

        # Look for sidecar _hl_policy.pt next to the main checkpoint
        hl_path = Path(checkpoint_path).with_name(
            Path(checkpoint_path).stem.split("_success")[0] + "_hl_policy.pt"
        )
        if hl_path.exists():
            try:
                intent_predictor.load_state_dict(
                    torch.load(hl_path, map_location=device, weights_only=True)
                )
                intent_predictor.eval()
                print(f"  Loaded intent predictor from {hl_path}")
            except Exception as e:
                # If sidecar is corrupted, try extracting from main checkpoint's training_state
                print(f"  [WARN] hl_policy.pt load failed ({e}); trying training_state in main ckpt...")
                main_sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
                pred_state = main_sd.get("training_state", {}).get("intent_predictor_state")
                if pred_state is not None:
                    intent_predictor.load_state_dict(pred_state)
                    intent_predictor.eval()
                    # Also fix the sidecar for next time
                    torch.save(pred_state, hl_path)
                    print(f"  Recovered predictor from training_state; re-saved sidecar to {hl_path}")
                else:
                    print(f"  [WARN] No predictor state found; falling back to CV proxy.")
                    intent_predictor = None
        else:
            # No sidecar — try loading predictor state from training_state in main checkpoint.
            # train_pusht.py saves intent_predictor_state inside the main ckpt's training_state.
            main_sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
            pred_state = main_sd.get("training_state", {}).get("intent_predictor_state")
            if pred_state is not None:
                intent_predictor.load_state_dict(pred_state)
                intent_predictor.eval()
                # Save sidecar so future loads skip this path
                torch.save(pred_state, hl_path)
                print(f"  Loaded predictor from training_state; saved sidecar to {hl_path}")
            else:
                print(f"  [WARN] No predictor state in ckpt; falling back to CV proxy.")
                intent_predictor = None

    return agent, intent_predictor


# ──────────────────────────────────────────────────────────────────────────────
# Observation preprocessing (mirrors train_robomimic.py eval loop)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_obs(obs_raw, config, dataset, device: str, intent_predictor=None):
    """Normalize obs and build intent conditioning. Handles both state and image obs.

    For state obs: obs_raw is np.ndarray (B, obs_steps, obs_dim).
    For image obs: obs_raw is dict of np.ndarrays, keyed by obs key.

    Returns:
        obs_out      : tensor or dict — ready to pass to agent.sample()
        lowdim_t     : (B, obs_steps, lowdim_dim) normalized lowdim tensor,
                       used to compute CV intent proxy and for FlowIntentAgent state input.
    """
    arch_variant = getattr(config.network, "arch_variant", "flow_action")
    has_intent = getattr(config.task, "intent_conditioning", False)
    intent_type = getattr(config.task, "intent_type", "mean")

    if config.task.obs_type == "state":
        obs_f = obs_raw.astype(np.float32)
        obs_norm = dataset.normalizer["obs"]["state"].normalize(obs_f)
        obs_t = torch.tensor(obs_norm, device=device, dtype=torch.float32)
        lowdim_t = obs_t  # (B, obs_steps, base_obs_dim)

        if has_intent and arch_variant != "flow_intent":
            if intent_predictor is not None:
                with torch.no_grad():
                    intent_proxy = intent_predictor(obs_t)  # (B, cond_dim)
            else:
                intent_slices = getattr(dataset, "intent_slices", None)
                intent_start = getattr(dataset, "intent_start", None)
                intent_end = getattr(dataset, "intent_end", None)
                if intent_start is None or intent_end is None:
                    raise RuntimeError(
                        "Intent CV proxy requires dataset.intent_start/end; "
                        "provide an intent predictor or a dataset with intent indices."
                    )
                half_h = (config.task.intent_horizon + 1) / 2.0
                proxies = []
                for s, e in (intent_slices or [(intent_start, intent_end)]):
                    feat_now  = obs_t[:, -1, s:e]
                    feat_prev = obs_t[:, -2, s:e]
                    velocity = feat_now - feat_prev
                    if intent_type == "sequence":
                        ks = torch.arange(1, config.task.intent_horizon + 1,
                                          dtype=torch.float32, device=device)
                        future_steps = feat_now.unsqueeze(1) + ks.view(-1, 1) * velocity.unsqueeze(1)
                        proxies.append(future_steps.reshape(obs_t.shape[0], -1))
                    else:
                        proxies.append(feat_now + half_h * velocity)
                intent_proxy = torch.cat(proxies, dim=-1)

            intent_expanded = intent_proxy.unsqueeze(1).expand(-1, config.task.obs_steps, -1)
            obs_t = torch.cat([obs_t, intent_expanded], dim=-1)

        return obs_t, lowdim_t

    else:  # image obs — mirrors train_robomimic.py eval loop lines 788-828
        obs_dict = {}
        for k, v in obs_raw.items():
            v_f = v.astype(np.float32)
            normalizer = dataset.normalizer["obs"].get(k, None)
            if normalizer is not None:
                v_f = normalizer.normalize(v_f)
            obs_dict[k] = torch.tensor(v_f, device=device, dtype=torch.float32)

        # Lowdim tensor for intent proxy: concatenate all lowdim keys
        lowdim_keys = list(getattr(dataset, "lowdim_keys", None) or ["state"])
        lowdim_parts = [obs_dict[k] for k in lowdim_keys]
        lowdim_t = (
            torch.cat(lowdim_parts, dim=-1)
            if len(lowdim_parts) > 1
            else lowdim_parts[0]
        )  # (B, obs_steps, lowdim_dim)

        if has_intent and arch_variant != "flow_intent":
            if intent_predictor is not None:
                with torch.no_grad():
                    intent_proxy = intent_predictor(lowdim_t)  # (B, cond_dim)
            else:
                # CV proxy (only valid for non-encoded_mean types)
                intent_start = getattr(dataset, "intent_start", None)
                intent_end = getattr(dataset, "intent_end", None)
                if intent_start is None or intent_end is None:
                    raise RuntimeError(
                        "Intent CV proxy requires dataset.intent_start/end; "
                        "provide an intent predictor or a dataset with intent indices."
                    )
                eef_now  = lowdim_t[:, -1, intent_start:intent_end]
                eef_prev = lowdim_t[:, -2, intent_start:intent_end]
                velocity = eef_now - eef_prev
                half_h = (config.task.intent_horizon + 1) / 2.0
                intent_proxy = eef_now + half_h * velocity

            intent_expanded = intent_proxy.unsqueeze(1).expand(
                -1, config.task.obs_steps, -1
            )  # (B, obs_steps, cond_dim)
            # PushT image: intent is concatenated into agent_pos (shape_meta specifies [4]).
            # LIBERO image: intent is concatenated into obs["state"].
            # Robomimic image: intent is a separate lowdim key added to obs_dict.
            if getattr(config.task, "env_name", "") == "pusht":
                obs_dict["agent_pos"] = torch.cat(
                    [obs_dict["agent_pos"], intent_expanded.contiguous()], dim=-1
                )
            elif _is_libero(config.task) and "state" in obs_dict:
                obs_dict["state"] = torch.cat(
                    [obs_dict["state"], intent_expanded.contiguous()], dim=-1
                )
            else:
                obs_dict["intent"] = intent_expanded.contiguous()

        return obs_dict, lowdim_t


# ──────────────────────────────────────────────────────────────────────────────
# Rollout collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_rollouts(config, agent, dataset, envs, n_rollouts: int, device: str,
                     is_flow_intent: bool = False, intent_predictor=None, num_steps: int = 9):
    """Run n_rollouts and return action chunks + visited states + success flags.

    Works for both state and image observations.
    For flow_intent: also collects steerability data at the first timestep.

    Returns:
        action_chunks: list of 1D np arrays (flattened chunk per timestep)
        ep_success:    list of bool, one per episode
        obs_states:    np.ndarray (N_steps_total, obs_dim) — raw env obs, last step only
        steer_actions: np.ndarray (n_eps, STEER_K, flat_dim) or None
        steer_intents: np.ndarray (n_eps, STEER_K, intent_dim) or None
    """
    action_chunks = []
    ep_success = []
    obs_states = []  # raw (unnormalized) obs at each step: (obs_dim,) for state obs
    steer_actions_list = []
    steer_intents_list = []
    s = config.task.obs_steps - 1  # action slice start

    n_done = 0
    while n_done < n_rollouts:
        obs, _ = envs.reset()
        ep_reward = np.zeros(config.task.num_envs)
        ep_success_flags = np.zeros(config.task.num_envs, dtype=bool)
        t = 0
        first_step = True

        while t < config.task.max_episode_steps:
            # Record raw env state (most recent obs step, before normalization)
            if config.task.obs_type == "state":
                obs_states.append(obs[0, -1, :].copy())  # (obs_dim,), env 0 only
            elif isinstance(obs, dict) and "state" in obs:
                obs_states.append(obs["state"][0, -1, :].copy())

            obs_in, lowdim_t = preprocess_obs(
                obs, config, dataset, device, intent_predictor=intent_predictor
            )

            # For flow_intent: use TensorDict for image obs, lowdim_t for state
            if is_flow_intent:
                if config.task.obs_type == "image":
                    B_fi = next(iter(obs_in.values())).shape[0]
                    fi_obs = TensorDict(obs_in, batch_size=B_fi)
                else:
                    fi_obs = lowdim_t

            # Steerability: at first timestep, sample STEER_K intent/action pairs
            if first_step and is_flow_intent:
                ep_steer_acts = []
                ep_steer_ints = []
                with torch.no_grad():
                    for _ in range(STEER_K):
                        act_k, intent_k = agent.sample(
                            obs=fi_obs, use_ema=True, num_steps=num_steps,
                            return_intent=True,
                        )
                        act_k_un = dataset.normalizer["action"].unnormalize(act_k.cpu().numpy())
                        ep_steer_acts.append(act_k_un[0, s:s + config.task.act_steps].flatten())
                        ep_steer_ints.append(intent_k[0].cpu().numpy())
                steer_actions_list.append(np.stack(ep_steer_acts))
                steer_intents_list.append(np.stack(ep_steer_ints))
                first_step = False

            # Regular sampling
            with torch.no_grad():
                if is_flow_intent:
                    act_normed = agent.sample(obs=fi_obs, use_ema=True, num_steps=num_steps)
                else:
                    act_0 = torch.randn(
                        (config.task.num_envs, config.task.horizon, config.task.act_dim),
                        device=device,
                    )
                    # For state obs, wrap in {"state": ...}; for image, obs_in is already a dict
                    sample_obs = (
                        {"state": obs_in} if config.task.obs_type == "state" else obs_in
                    )
                    act_normed = agent.sample(act_0=act_0, obs=sample_obs, num_steps=num_steps, use_ema=True)

            act_un = dataset.normalizer["action"].unnormalize(act_normed.cpu().numpy())
            act_chunk = act_un[:, s:s + config.task.act_steps]  # (B, act_steps, act_dim)
            action_chunks.append(act_chunk[0].flatten())  # save before rotation undo

            # Convert rotation_6d → axis-angle for env (mirrors train_robomimic.py:908-915)
            act_env = act_chunk
            _ABS_ACTION_ENVS = {"can", "lift", "square", "tool_hang", "transport"}
            if getattr(config.task, "abs_action", False) and config.task.env_name in _ABS_ACTION_ENVS:
                act_env = dataset.undo_transform_action(act_chunk)

            obs, reward, terminated, truncated, info = envs.step(act_env)
            ep_reward += reward
            t += config.task.act_steps

            if "_final_info" in info and "final_info" in info:
                for i in range(config.task.num_envs):
                    if info["_final_info"][i]:
                        final_info = info["final_info"][i]
                        if final_info and "success" in final_info:
                            ep_success_flags[i] = ep_success_flags[i] or bool(
                                np.asarray(final_info["success"]).any()
                            )
            if "success" in info:
                success_info = info["success"]
                for i in range(config.task.num_envs):
                    success_value = (
                        success_info[i] if hasattr(success_info, "__len__") else success_info
                    )
                    ep_success_flags[i] = ep_success_flags[i] or bool(
                        np.asarray(success_value).any()
                    )

        for i in range(config.task.num_envs):
            ep_success.append(bool(ep_success_flags[i] or (ep_reward[i] > 0)))
        n_done += config.task.num_envs

    steer_actions = np.stack(steer_actions_list) if steer_actions_list else None
    steer_intents = np.stack(steer_intents_list) if steer_intents_list else None
    obs_states_arr = np.array(obs_states, dtype=np.float32) if obs_states else None
    return action_chunks, ep_success, obs_states_arr, steer_actions, steer_intents


# ──────────────────────────────────────────────────────────────────────────────
# GT demo action collection (no env needed)
# ──────────────────────────────────────────────────────────────────────────────

def collect_gt_demos(dataset, act_steps: int, obs_steps: int, max_samples: int = 2000):
    """Pull action chunks directly from the dataset (no env rollout).

    Works for both state and image datasets — both return batch["action"]
    as normalized actions with shape (B, horizon, act_dim).

    Slice: act[:, s:s+act_steps] where s = obs_steps - 1.
    This matches exactly the slice used in collect_rollouts().
    """
    from torch.utils.data import DataLoader
    # Use a single worker so demo collection works in restricted sandboxed
    # environments that do not permit the multiprocessing listener socket.
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    s = obs_steps - 1
    chunks = []
    for batch in loader:
        act = batch["action"].numpy()   # (B, horizon, act_dim) — normalized
        chunk_norm = act[:, s:s + act_steps, :]
        chunk = dataset.normalizer["action"].unnormalize(chunk_norm)  # raw action space
        for i in range(chunk.shape[0]):
            chunks.append(chunk[i].flatten())
        if len(chunks) >= max_samples:
            break
    return np.array(chunks[:max_samples])


def collect_gt_states(dataset, obs_steps: int, max_samples: int = 5000):
    """Pull raw (unnormalized) obs states from the demo dataset.

    Uses the most-recent obs step (index obs_steps-1) as the environment state
    at each timestep, matching what collect_rollouts() records from the env.

    Returns:
        states: np.ndarray (N, obs_dim) — unnormalized demo states
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    s = obs_steps - 1
    states = []
    for batch in loader:
        obs = batch.get("obs", {})
        if isinstance(obs, dict):
            state_norm = obs.get("state", None)
        else:
            state_norm = obs
        if state_norm is None:
            break
        state_norm = state_norm.numpy()          # (B, obs_steps, obs_dim)
        state_raw = dataset.normalizer["obs"]["state"].unnormalize(state_norm[:, s, :])
        for i in range(state_raw.shape[0]):
            states.append(state_raw[i])
        if len(states) >= max_samples:
            break
    return np.array(states[:max_samples], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect diversity rollouts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Each --run is: "label:ckpt_path:override1:override2:..."
  label      — name shown in plots (e.g. baseline, flow_intent)
  ckpt_path  — path to .pt checkpoint
  overrides  — Hydra overrides selecting the right config (task=..., network.arch_variant=...)

Example:
  --run "baseline:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_success0.pt:task=lift_ph_state"
  --run "flow_intent:checkpoints/lift_ph_state_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt:task=lift_ph_state_flow_intent:network.arch_variant=flow_intent"
""",
    )
    parser.add_argument("--task", required=True,
                        help="Task label used as top-level key in the output pkl (e.g. lift_ph)")
    parser.add_argument("--run", action="append", required=True, dest="runs",
                        metavar="label:ckpt:override...",
                        help="Run spec (repeatable). Format: label:ckpt_path[:hydra_override ...]")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, help="Output .pkl path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    results = {args.task: {}}
    gt_demos_collected = False

    for run_spec in args.runs:
        label, ckpt_path, user_overrides = parse_run_spec(run_spec)

        if not Path(ckpt_path).exists():
            print(f"[SKIP] {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Label: {label}  |  ckpt: {ckpt_path}")
        print(f"Overrides: {user_overrides}")
        print(f"{'='*60}")

        # Merge user overrides with runtime defaults
        overrides = user_overrides + [
            f"optimization.device={args.device}",
            "task.num_envs=1",
            f"optimization.seed={args.seed}",
        ]
        config = load_config(overrides)

        # Set up env to get obs_dim at runtime (mirrors train_robomimic.py main())
        envs = make_vec_env(config.task, seed=args.seed)
        setup_config_for_env(config, envs)

        # Dataset (for normalizer + GT demos)
        dataset = make_dataset(config.task)

        # Load model (+ intent predictor sidecar if needed)
        try:
            agent, intent_predictor = load_model(ckpt_path, config, dataset, args.device)
        except Exception as e:
            print(f"[ERROR] Failed to load {label}: {e}")
            envs.close()
            continue

        arch_variant = getattr(config.network, "arch_variant", "flow_action")
        is_fi = (arch_variant == "flow_intent")
        print(f"arch_variant={arch_variant}  obs_type={config.task.obs_type}  "
              f"intent_predictor={'yes' if intent_predictor else 'no/cv-proxy'}  "
              f"|  Rolling out {args.n_rollouts} episodes...")

        # Use same num_steps as training eval (9 for flow, 1 for regression/mip)
        from mip.samplers import get_default_step_list
        _num_steps = int(get_default_step_list(config.optimization.loss_type)[0])

        try:
            chunks, successes, obs_states_arr, steer_acts, steer_ints = collect_rollouts(
                config, agent, dataset, envs,
                n_rollouts=args.n_rollouts,
                device=args.device,
                is_flow_intent=is_fi,
                intent_predictor=intent_predictor,
                num_steps=_num_steps,
            )
        except Exception as e:
            print(f"[ERROR] Rollout failed for {label}: {e}")
            envs.close()
            continue
        envs.close()

        sr = np.mean(successes)
        print(f"  Success rate: {sr:.1%}  |  Total chunks: {len(chunks)}")

        entry = {
            "action_chunks": np.array(chunks),
            "success": successes,
        }
        if obs_states_arr is not None:
            entry["obs_states"] = obs_states_arr
            print(f"  States: {obs_states_arr.shape}")
        if steer_acts is not None:
            entry["steer_actions"] = steer_acts
            entry["steer_intents"] = steer_ints
            print(f"  Steer data: {steer_acts.shape}")

        results[args.task][label] = entry

        # Collect GT demos once (using normalizer from whichever dataset loads first)
        if not gt_demos_collected:
            print("Collecting GT demo actions from dataset...")
            gt_chunks = collect_gt_demos(dataset, act_steps=config.task.act_steps, obs_steps=config.task.obs_steps)
            print("Collecting GT demo states from dataset...")
            gt_states = collect_gt_states(dataset, obs_steps=config.task.obs_steps)
            results[args.task]["gt_demos"] = {
                "action_chunks": gt_chunks,
                "obs_states": gt_states,
            }
            gt_demos_collected = True
            print(f"  GT demos: {gt_chunks.shape}  |  GT states: {gt_states.shape}")

    # Save
    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {args.out}")
    for label, data in results[args.task].items():
        if "action_chunks" in data:
            print(f"  {label}: {data['action_chunks'].shape}")


if __name__ == "__main__":
    main()
