"""LIBERO environment wrapper.

Wraps LIBERO's ControlEnv in a Gymnasium-compatible interface and stacks
observations / multi-steps using the shared MIP env utilities.

The LIBERO HDF5 demos store obs with keys like ``ee_states``,
``gripper_states``, ``joint_states`` which are *different* from the raw
robosuite obs keys (``robot0_eef_pos``, ``robot0_eef_quat``, etc.).
This wrapper bridges that gap by mapping the live-env obs to the same
representation as the HDF5 dataset.

HDF5-to-env key mapping (matching create_dataset.py):
    ee_pos          (3)  <->  robot0_eef_pos                      (3)
    ee_ori          (3)  <->  quat2axisangle(robot0_eef_quat)     (4 -> 3)
    ee_states       (6)  <->  [ee_pos, quat2axisangle(eef_quat)]
    gripper_states  (2)  <->  robot0_gripper_qpos                 (2)
    joint_states    (7)  <->  robot0_joint_pos                    (7)
"""

import math

import mip.envs.libero._robosuite_compat  # noqa: F401  — patches must run first

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from mip.config import TaskConfig
from mip.env_utils import MultiStepWrapper, VideoRecorder, VideoRecordingWrapper

_HDF5_KEY_TO_ENV = {
    "ee_pos": ("robot0_eef_pos",),
    "ee_ori": ("robot0_eef_quat",),  # needs quat -> axis-angle conversion
    "ee_states": ("robot0_eef_pos", "robot0_eef_quat"),  # concat pos + quat2axisangle
    "gripper_states": ("robot0_gripper_qpos",),
    "joint_states": ("robot0_joint_pos",),
}


def _quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle (3D).

    Matches robosuite's T.quat2axisangle used in LIBERO's create_dataset.py.
    """
    w = q[..., 3].copy()
    w = np.clip(w, -1.0, 1.0)
    # For batched or single input
    xyz = q[..., :3]
    den = np.sqrt(1.0 - w * w)
    # avoid division by zero (near-zero rotation)
    safe = den > 1e-10
    angle = 2.0 * np.arccos(w)
    if q.ndim == 1:
        if safe:
            return xyz * angle / den
        return np.zeros(3)
    result = np.where(safe[..., None], xyz * (angle / np.where(safe, den, 1.0))[..., None], 0.0)
    return result


def make_vec_env(task_config: TaskConfig, seed=None):
    """Create a vectorized LIBERO environment."""
    obs_type = getattr(task_config, "obs_type", "state")
    if task_config.num_envs == 1 or task_config.save_video or obs_type == "image":
        # Image obs: async forked workers compete for EGL contexts and fail.
        # Use sync for sequential, single-process offscreen rendering.
        vec_cls = gym.vector.SyncVectorEnv
    else:
        vec_cls = gym.vector.AsyncVectorEnv

    envs = vec_cls(
        [
            _make_single_env(task_config, idx, task_config.save_video, seed=seed)
            for idx in range(task_config.num_envs)
        ]
    )
    return envs


def _make_single_env(task_config: TaskConfig, idx: int, render: bool = False, seed=None):
    def thunk():
        obs_type = getattr(task_config, "obs_type", "state")
        image_obs_keys = list(getattr(task_config, "image_obs_keys", None) or [])
        render_camera = getattr(task_config, "render_camera_name", "agentview")
        env = LiberoGymWrapper(
            bddl_file=task_config.bddl_file,
            obs_keys=task_config.obs_keys,
            obs_type=obs_type,
            image_obs_keys=image_obs_keys,
            render_camera=render_camera,
            offscreen_render=render,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=10,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        )
        file_path = None if not render else "results/video.mp4"
        env = VideoRecordingWrapper(
            env, video_recorder, file_path=file_path, steps_per_render=1
        )
        env = MultiStepWrapper(
            env,
            n_obs_steps=task_config.obs_steps,
            n_action_steps=task_config.act_steps,
            max_episode_steps=task_config.max_episode_steps,
        )
        if seed is not None:
            env.seed(seed + idx)
        return env

    return thunk


_IMG_KEY_TO_CAMERA = {
    "agentview_rgb": "agentview",
    "eye_in_hand_rgb": "robot0_eye_in_hand",
}


class LiberoGymWrapper(gym.Env):
    """Gymnasium wrapper around LIBERO's ControlEnv.

    Accepts obs_keys in HDF5 naming convention (``ee_states``,
    ``gripper_states``, ``joint_states``) and automatically maps them
    to the live robosuite env's observation keys, including
    quaternion → axis-angle conversion for orientation.

    For image observations, set obs_type="image" and provide image_obs_keys
    (e.g. ["agentview_rgb", "eye_in_hand_rgb"]). The observation_space
    becomes a Dict with both state ("state") and image keys.
    """

    def __init__(
        self,
        bddl_file: str,
        obs_keys: list[str],
        obs_type: str = "state",
        image_obs_keys: list[str] | None = None,
        render_hw: tuple[int, int] = (256, 256),
        render_camera: str = "agentview",
        offscreen_render: bool = False,
    ):
        from libero.libero.envs.env_wrapper import ControlEnv

        self.obs_keys = obs_keys
        self.obs_type = obs_type
        self.image_obs_keys = image_obs_keys or []
        self.render_hw = render_hw
        self.render_camera = render_camera

        use_images = obs_type == "image" and bool(self.image_obs_keys)
        need_offscreen = use_images or offscreen_render
        camera_names = [_IMG_KEY_TO_CAMERA[k] for k in self.image_obs_keys if k in _IMG_KEY_TO_CAMERA]
        # When rendering for state obs, use agentview (or configured camera) for render() calls.
        render_cam_list = camera_names if use_images else ([render_camera] if offscreen_render else [])
        render_heights = [128] * len(camera_names) if use_images else ([render_hw[0]] if offscreen_render else [])
        render_widths = [128] * len(camera_names) if use_images else ([render_hw[1]] if offscreen_render else [])

        self._env = ControlEnv(
            bddl_file_name=bddl_file,
            has_renderer=False,
            has_offscreen_renderer=need_offscreen,
            use_camera_obs=use_images,
            camera_names=render_cam_list,
            camera_heights=render_heights,
            camera_widths=render_widths,
            control_freq=20,
        )

        raw_obs = self._env.reset()
        act_dim = self._env.env.action_dim

        if use_images:
            flat_state = self._flatten_state(raw_obs)
            state_dim = flat_state.shape[0]
            spaces_dict = {"state": Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)}
            for img_key in self.image_obs_keys:
                spaces_dict[img_key] = Box(low=0.0, high=1.0, shape=(3, 128, 128), dtype=np.float32)
            self.observation_space = gym.spaces.Dict(spaces_dict)
        else:
            flat_obs = self._flatten_state(raw_obs)
            obs_dim = flat_obs.shape[0]
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.action_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self._last_obs = self._get_obs(raw_obs)

    def _extract_state_key(self, obs_dict: dict, hdf5_key: str) -> np.ndarray:
        """Extract a single HDF5-named state key from the live env obs dict."""
        if hdf5_key not in _HDF5_KEY_TO_ENV:
            return obs_dict[hdf5_key].astype(np.float32).ravel()
        env_keys = _HDF5_KEY_TO_ENV[hdf5_key]
        parts = []
        for ek in env_keys:
            val = obs_dict[ek].astype(np.float32)
            if ek == "robot0_eef_quat" and hdf5_key in ("ee_ori", "ee_states"):
                val = _quat_to_axisangle(val)
            if val.ndim == 0:
                val = val[None]
            parts.append(val.ravel())
        return np.concatenate(parts)

    def _flatten_state(self, obs_dict: dict) -> np.ndarray:
        parts = [self._extract_state_key(obs_dict, k) for k in self.obs_keys]
        return np.concatenate(parts, axis=0)

    def _get_image(self, obs_dict: dict, img_key: str) -> np.ndarray:
        """Get image from env obs dict as float32 CHW in [0, 1]."""
        cam = _IMG_KEY_TO_CAMERA.get(img_key, img_key)
        # robosuite camera obs key format: "{camera_name}_image"
        env_key = f"{cam}_image"
        img = obs_dict[env_key]  # (H, W, C) uint8
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0  # (C, H, W)

    def _get_obs(self, obs_dict: dict):
        """Return obs in the correct format (flat array or dict)."""
        state = self._flatten_state(obs_dict)
        if self.obs_type == "image" and self.image_obs_keys:
            result = {"state": state}
            for img_key in self.image_obs_keys:
                result[img_key] = self._get_image(obs_dict, img_key)
            return result
        return state

    # Number of open-gripper warm-up steps applied after every reset.
    # Needed because LIBERO resets to the robot's default joint config
    # (gripper_qpos ≈ 0.021), but all demos start with the gripper fully
    # open (gripper_qpos ≈ 0.036).  Without this the normalizer maps the
    # reset observation into the "grasping" zone, causing the policy to
    # predict close-gripper throughout evaluation.
    _GRIPPER_WARMUP_STEPS: int = 10

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env.seed(seed)
        raw_obs = self._env.reset()
        # Warm-up: send open-gripper actions to match the demo initial state.
        open_action = np.zeros(self._env.env.action_dim, dtype=np.float32)
        open_action[-1] = -1.0  # gripper convention: -1 = open
        for _ in range(self._GRIPPER_WARMUP_STEPS):
            raw_obs, _, done, _ = self._env.step(open_action)
            if done:
                raw_obs = self._env.reset()
                break
        self._last_obs = self._get_obs(raw_obs)
        return self._last_obs, {}

    def set_init_state(self, init_state):
        """Set a fixed MuJoCo initial state (for LIBERO-PRO evaluation)."""
        raw_obs = self._env.set_init_state(init_state)
        self._last_obs = self._get_obs(raw_obs)
        return self._last_obs, {}

    def step(self, action):
        raw_obs, reward, done, info = self._env.step(action)
        obs = self._get_obs(raw_obs)
        self._last_obs = obs
        success = bool(self._env.check_success())
        info["success"] = success
        terminated = bool(done) or success
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        imgs = self._env.env.sim.render(
            height=h, width=w, camera_name=self.render_camera
        )
        return imgs[::-1]

    def seed(self, seed=None):
        if seed is not None:
            try:
                self._env.seed(seed)
            except TypeError:
                # robosuite 1.5.x may not expose seed() on the env
                np.random.seed(seed)

    def close(self):
        self._env.close()
