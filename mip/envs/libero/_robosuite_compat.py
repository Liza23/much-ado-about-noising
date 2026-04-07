"""Robosuite 1.5.x compatibility shims for LIBERO.

MUST be imported before any libero.* imports. Patches the robosuite API
surface that LIBERO was written against (robosuite <=1.4) so that it works
with the 1.5.x module layout and calling conventions.

Changes in robosuite 1.5.x that break LIBERO:
  - load_controller_config  -> load_part_controller_config
  - ManipulationEnv.__init__ no longer accepts `mount_types`
  - Robot models require `arms`, `default_base`, and dict-valued
    `default_gripper` (was str-valued `default_mount` / `default_gripper`)
"""

import robosuite as _suite

# ---------------------------------------------------------------------------
# 1) Restore the old top-level name that LIBERO's ControlEnv calls.
#    In 1.5.x the old load_controller_config returned a *part* config,
#    but the env now expects a *composite* config.  We wrap the part config
#    inside a BASIC composite so that composite_controller_factory accepts it.
# ---------------------------------------------------------------------------
if not hasattr(_suite, "load_controller_config"):

    def _compat_load_controller_config(default_controller="OSC_POSE"):
        part_cfg = _suite.load_part_controller_config(
            default_controller=default_controller
        )
        part_cfg["gripper"] = {"type": "GRIP"}
        return {
            "type": "BASIC",
            "body_parts": {"right": part_cfg},
        }

    _suite.load_controller_config = _compat_load_controller_config

# ---------------------------------------------------------------------------
# 2) ManipulationEnv.__init__: silently drop `mount_types` (removed in 1.5).
# ---------------------------------------------------------------------------
from robosuite.environments.manipulation.manipulation_env import (
    ManipulationEnv as _ManipEnv,
)

_orig_manip_init = _ManipEnv.__init__


def _patched_manip_init(self, *args, **kwargs):
    kwargs.pop("mount_types", None)
    _orig_manip_init(self, *args, **kwargs)


_ManipEnv.__init__ = _patched_manip_init

# ---------------------------------------------------------------------------
# 3) Patch LIBERO custom Panda robot models.
#    In 1.5.x the model interface changed:
#      - `arms` class attr  (list of arm names, e.g. ["right"])
#      - `default_mount` (property, str) -> `default_base` (property, str)
#      - `default_gripper` must return {"right": name} instead of just name
# ---------------------------------------------------------------------------
from libero.libero.envs.robots.mounted_panda import MountedPanda as _MountedPanda
from libero.libero.envs.robots.on_the_ground_panda import (
    OnTheGroundPanda as _OnTheGroundPanda,
)


def _patch_robot_cls(cls):
    if not hasattr(cls, "arms") or cls.arms is None:
        cls.arms = ["right"]

    _mount_prop = cls.__dict__.get("default_mount")
    if _mount_prop is not None and "default_base" not in cls.__dict__:
        if isinstance(_mount_prop, property):
            cls.default_base = _mount_prop
        else:
            cls.default_base = property(lambda self, _m=_mount_prop: _m)

    _gripper_prop = cls.__dict__.get("default_gripper")
    if _gripper_prop is not None and isinstance(_gripper_prop, property):
        _orig_fget = _gripper_prop.fget

        def _dict_gripper(self, _fget=_orig_fget):
            val = _fget(self)
            if isinstance(val, str):
                return {"right": val}
            return val

        cls.default_gripper = property(_dict_gripper)


_patch_robot_cls(_MountedPanda)
_patch_robot_cls(_OnTheGroundPanda)
