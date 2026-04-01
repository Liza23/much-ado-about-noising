"""LIBERO environment wrapper.

Wraps LIBERO's ControlEnv in a Gymnasium-compatible interface and stacks
observations / multi-steps using the shared MIP env utilities.
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

# Robosuite 1.5.x compatibility: patch renamed/removed APIs before LIBERO imports them.
import robosuite as _suite
if not hasattr(_suite, "load_controller_config"):
    _suite.load_controller_config = _suite.load_part_controller_config

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv as _ManipEnv
_orig_manip_init = _ManipEnv.__init__
def _patched_manip_init(self, *args, **kwargs):
    kwargs.pop("mount_types", None)  # removed in robosuite 1.5.x
    _orig_manip_init(self, *args, **kwargs)
_ManipEnv.__init__ = _patched_manip_init

# LIBERO custom robot models: patch for robosuite 1.5.x API changes.
# - arms = ["right"] required by FixedBaseRobot
# - default_mount -> default_base
# - default_gripper returns str -> dict {"right": ...}
from libero.libero.envs.robots.mounted_panda import MountedPanda as _MountedPanda
from libero.libero.envs.robots.on_the_ground_panda import OnTheGroundPanda as _OnTheGroundPanda

for _cls in (_MountedPanda, _OnTheGroundPanda):
    if not hasattr(_cls, "arms"):
        _cls.arms = ["right"]
    if not hasattr(_cls, "default_base"):
        _mount = _cls.default_mount.fget(_cls) if isinstance(_cls.default_mount, property) else _cls.default_mount
        _cls.default_base = property(lambda self, _m=_mount: _m)
    # robosuite 1.5.x expects default_gripper to return {"right": gripper_name}
    _orig_gripper = _cls.default_gripper
    if isinstance(_orig_gripper, property):
        _cls.default_gripper = property(
            lambda self, _p=_orig_gripper: {"right": _p.fget(self)}
            if isinstance(_p.fget(self), str) else _p.fget(self)
        )

from mip.config import TaskConfig
from mip.env_utils import MultiStepWrapper, VideoRecorder, VideoRecordingWrapper


def make_vec_env(task_config: TaskConfig, seed=None):
    """Create a vectorized LIBERO environment.

    Args:
        task_config: Task config. Must include:
            - bddl_file: path to the task BDDL file
            - obs_keys: list of robosuite observation keys to concatenate
            - obs_steps, act_steps, max_episode_steps, num_envs, save_video
        seed: Random seed.

    Returns:
        Vectorized gymnasium environment.
    """
    if task_config.num_envs == 1 or task_config.save_video:
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
        env = LiberoGymWrapper(
            bddl_file=task_config.bddl_file,
            obs_keys=task_config.obs_keys,
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


class LiberoGymWrapper(gym.Env):
    """Gymnasium wrapper around LIBERO's ControlEnv.

    Converts the robosuite obs dict → flat numpy array and exposes the
    standard Gymnasium API (reset/step with terminated/truncated).

    Args:
        bddl_file: Absolute path to the BDDL problem file for the task.
        obs_keys: Robosuite obs keys to concatenate into the state vector.
        render_hw: (height, width) for render().
        render_camera: Camera name for render().
    """

    def __init__(
        self,
        bddl_file: str,
        obs_keys: list[str],
        render_hw: tuple[int, int] = (256, 256),
        render_camera: str = "agentview",
    ):
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv

        self.obs_keys = obs_keys
        self.render_hw = render_hw
        self.render_camera = render_camera

        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera=render_camera,
            camera_names=[render_camera],
            camera_heights=render_hw[0],
            camera_widths=render_hw[1],
            control_freq=20,
        )

        # Bootstrap obs shape by doing a reset
        raw_obs = self._env.reset()
        flat_obs = self._flatten_obs(raw_obs)
        obs_dim = flat_obs.shape[0]
        act_dim = self._env.env.action_dim

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self._last_obs = flat_obs

    # ------------------------------------------------------------------
    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        parts = []
        for key in self.obs_keys:
            val = obs_dict[key]
            if val.ndim == 0:
                val = val[None]
            parts.append(val.astype(np.float32))
        return np.concatenate(parts, axis=0)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env.seed(seed)
        raw_obs = self._env.reset()
        self._last_obs = self._flatten_obs(raw_obs)
        return self._last_obs, {}

    def step(self, action):
        raw_obs, reward, done, info = self._env.step(action)
        obs = self._flatten_obs(raw_obs)
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
        return imgs[::-1]  # flip vertically (MuJoCo convention)

    def seed(self, seed=None):
        if seed is not None:
            self._env.seed(seed)

    def close(self):
        self._env.close()
