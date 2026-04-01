"""Robomimic EEF teleport renderer for render-augmented training.

Restores a MuJoCo sim state from the dataset, applies Jacobian IK to move the
end-effector toward the draft action target position (no physics step), renders
a camera image, then restores the original state.

Analogous to PushTAgentOnlyRenderer but for robosuite/robomimic environments.

Author: Claude Code
Date: 2026-02-19
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger


# EEF site name for Franka Panda in robosuite
_EEF_SITE_NAME = "gripper0_right_grip_site"

# Damping factor for damped least-squares IK
_IK_DAMPING = 0.05

# Maximum delta per IK step (metres) to keep renders sensible
_IK_MAX_DELTA = 0.15


class RobomimicEEFRenderer:
    """Renders images with EEF teleported to draft action position via Jacobian IK.

    No physics step is executed: objects stay exactly where they are in the
    dataset state. Only the arm joints are updated (kinematics-only forward pass).

    Interface matches PushTAgentOnlyRenderer so RenderAugmentedNetwork works
    without modification.
    """

    def __init__(
        self,
        env,
        render_size: int = 96,
        camera_name: str = "sideview",
        max_renders_per_batch: int | None = None,
        ik_steps: int = 5,
    ):
        """
        Args:
            env: A single robomimic/robosuite env (EnvRobosuite or wrapper).
                 Must have render_offscreen=True.
            render_size: Output image side length (square).
            camera_name: Camera to render from.
            max_renders_per_batch: Optional cap; extras are resampled from renders.
            ik_steps: Number of Jacobian IK iterations to apply.
        """
        self.render_size = render_size
        self.camera_name = camera_name
        self.max_renders_per_batch = max_renders_per_batch
        self.ik_steps = ik_steps

        # Unwrap to EnvRobosuite
        self._env_robosuite = self._unwrap_to_robosuite(env)
        robosuite_env = self._env_robosuite.env          # actual robosuite Env
        self._sim = robosuite_env.sim
        self._robot = robosuite_env.robots[0]

        # Cache arm joint info (computed once)
        import mujoco
        self._mujoco = mujoco
        self._eef_site_id = mujoco.mj_name2id(
            self._sim.model._model,
            mujoco.mjtObj.mjOBJ_SITE,
            _EEF_SITE_NAME,
        )
        arm_joint_names = self._robot.robot_joints
        self._arm_qpos_addrs = np.array([
            self._sim.model.jnt_qposadr[self._sim.model.joint_name2id(jn)]
            for jn in arm_joint_names
        ], dtype=np.int32)
        self._arm_dof_addrs = np.array([
            self._sim.model.jnt_dofadr[self._sim.model.joint_name2id(jn)]
            for jn in arm_joint_names
        ], dtype=np.int32)
        self._n_arm_dof = len(self._arm_dof_addrs)

        logger.info(
            f"[RobomimicEEFRenderer] EEF site={_EEF_SITE_NAME} (id={self._eef_site_id}), "
            f"arm DOFs={self._arm_dof_addrs.tolist()}, "
            f"camera={camera_name}, render_size={render_size}"
        )

    # ------------------------------------------------------------------

    def __call__(
        self,
        render_state: torch.Tensor,
        render_action: torch.Tensor,
    ) -> torch.Tensor:
        """Render images with EEF moved to action target position.

        Args:
            render_state: [B, state_dim] flat MuJoCo states from HDF5 dataset.
            render_action: [B, act_dim] unnormalized absolute actions.
                           First 3 dims are target EEF position in world coords.

        Returns:
            [B, 3, H, W] float32 tensor in [0, 1].
        """
        batch_size = render_state.shape[0]
        device = render_state.device

        num_to_render = (
            batch_size
            if self.max_renders_per_batch is None
            else min(batch_size, self.max_renders_per_batch)
        )

        # Save current sim state so we can restore it after the batch
        saved_state = self._env_robosuite.get_state()["states"]

        future_images = []
        for i in range(num_to_render):
            state_i = render_state[i].detach().cpu().numpy().astype(np.float64)
            action_i = render_action[i].detach().cpu().numpy()
            target_pos = action_i[:3].astype(np.float64)

            img = self._render_one(state_i, target_pos)
            future_images.append(img)

        # Restore original state
        self._env_robosuite.reset_to({"states": saved_state})

        # Stack [num_to_render, C, H, W]
        rendered = torch.stack(future_images, dim=0).to(device)

        if num_to_render < batch_size:
            logger.warning(
                f"[RobomimicEEFRenderer] Replicating renders: "
                f"{num_to_render}/{batch_size} are true renders."
            )
            extra_idx = torch.randint(
                0, num_to_render, (batch_size - num_to_render,), device=device
            )
            rendered = torch.cat([rendered, rendered[extra_idx]], dim=0)

        return rendered

    # ------------------------------------------------------------------

    def _render_one(self, state_flat: np.ndarray, target_pos: np.ndarray) -> torch.Tensor:
        """Restore sim to state_flat, IK to target_pos, render.

        Returns:
            [3, H, W] float32 tensor in [0, 1].
        """
        sim = self._sim
        mujoco = self._mujoco

        # 1. Restore sim to dataset state (arm + objects in correct config)
        self._env_robosuite.reset_to({"states": state_flat})

        # 2. Jacobian IK: iteratively move EEF toward target_pos
        for _ in range(self.ik_steps):
            eef_pos = sim.data.site_xpos[self._eef_site_id].copy()
            delta = target_pos - eef_pos
            # Clamp to avoid excessive joint changes
            delta_norm = np.linalg.norm(delta)
            if delta_norm > _IK_MAX_DELTA:
                delta = delta * (_IK_MAX_DELTA / delta_norm)

            # Compute site Jacobian [3, nv]
            jacp = np.zeros((3, sim.model.nv))
            jacr = np.zeros((3, sim.model.nv))
            mujoco.mj_jacSite(
                sim.model._model, sim.data._data, jacp, jacr, self._eef_site_id
            )

            # Restrict to arm DOF columns
            J = jacp[:, self._arm_dof_addrs]  # [3, n_arm_dof]

            # Damped pseudoinverse: J^T (J J^T + λI)^{-1}
            JJT = J @ J.T  # [3, 3]
            JJT_reg = JJT + _IK_DAMPING ** 2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT_reg, delta)  # [n_arm_dof]

            # Apply joint update (kinematics only)
            sim.data.qpos[self._arm_qpos_addrs] += dq
            mujoco.mj_kinematics(sim.model._model, sim.data._data)

        # 3. Render
        img_np = self._env_robosuite.render(
            mode="rgb_array",
            height=self.render_size,
            width=self.render_size,
            camera_name=self.camera_name,
        )  # (H, W, 3) uint8

        # 4. Convert to [3, H, W] float32 in [0, 1]
        img_t = torch.from_numpy(img_np.copy()).float().permute(2, 0, 1) / 255.0
        return img_t

    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_to_robosuite(env):
        """Unwrap wrapper stack until we reach EnvRobosuite."""
        from robomimic.envs.env_robosuite import EnvRobosuite

        current = env
        while not isinstance(current, EnvRobosuite):
            if hasattr(current, "env"):
                current = current.env
            else:
                raise RuntimeError(
                    f"Could not unwrap to EnvRobosuite; stopped at {type(current)}"
                )
        return current


def create_robomimic_eef_renderer(
    env,
    render_size: int = 96,
    camera_name: str = "sideview",
    max_renders_per_batch: int | None = None,
) -> RobomimicEEFRenderer:
    """Factory function to create a RobomimicEEFRenderer."""
    return RobomimicEEFRenderer(
        env,
        render_size=render_size,
        camera_name=camera_name,
        max_renders_per_batch=max_renders_per_batch,
    )
