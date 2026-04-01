"""Image renderer for PushT that steps environment and renders future state.

This renderer takes raw state vectors and actions, sets the PushT physics engine
to the correct state per sample, steps forward, and renders the resulting image.

Author: Claude Code
Date: 2026-02-09
"""

import numpy as np
import torch
from typing import Dict
from loguru import logger
from gymnasium import Env


class PushTImageRenderer:
    """Renders future state images by stepping PushT environment."""

    def __init__(
        self,
        env: Env,
        render_size: int = 96,
        max_renders_per_batch: int | None = None,
    ):
        """Initialize the image renderer.

        Args:
            env: PushT environment (will be unwrapped to base env)
            render_size: Size of rendered images (render_size x render_size)
            max_renders_per_batch: Optional cap on number of true renders per batch.
                If None, render all samples in the batch.
        """
        self.render_size = render_size
        self.max_renders_per_batch = max_renders_per_batch

        self.wrapped_env = env
        self.env = self._unwrap_env(env)

    def __call__(
        self, render_state: torch.Tensor, render_action: torch.Tensor
    ) -> torch.Tensor:
        """Render future images by stepping environment.

        Args:
            render_state: [B, 5] raw state [agent_x, agent_y, block_x, block_y, block_angle]
            render_action: [B, 2] raw action [target_x, target_y]

        Returns:
            Rendered future images [B, C, H, W]
        """
        batch_size = render_state.shape[0]
        device = render_state.device

        if self.max_renders_per_batch is None:
            num_to_render = batch_size
        else:
            num_to_render = min(batch_size, self.max_renders_per_batch)

        future_images = []

        for i in range(num_to_render):
            # Save current environment state
            original_state = self._get_env_state()

            # Set environment to this sample's state
            state_i = render_state[i].detach().cpu().numpy()
            self._set_env_state(state_i)

            # Step environment with ground truth action
            action_i = render_action[i].detach().cpu().numpy()
            self.env.step(action_i)

            # Render the resulting state
            future_img = self.env.render(mode="rgb_array")

            # Restore original state
            self._set_env_state_dict(original_state)

            # Convert to tensor [C, H, W]
            if isinstance(future_img, np.ndarray):
                future_img = torch.from_numpy(future_img).float()
                if future_img.ndim == 3 and future_img.shape[2] == 3:
                    future_img = future_img.permute(2, 0, 1)
                if future_img.max() > 1.0:
                    future_img = future_img / 255.0

            future_images.append(future_img)

        # Stack rendered images [num_to_render, C, H, W]
        rendered_images = torch.stack(future_images, dim=0).to(device)

        # If we didn't render all, replicate random samples for the rest
        if num_to_render < batch_size:
            logger.warning(
                f"[PushTImageRenderer] Replicating rendered futures: "
                f"{num_to_render}/{batch_size} are true renders. "
                f"Increase task.max_renders_per_batch or set it to null to disable capping."
            )
            indices = torch.randint(0, num_to_render, (batch_size - num_to_render,), device=device)
            additional_images = rendered_images[indices]
            future_images_batch = torch.cat([rendered_images, additional_images], dim=0)
        else:
            future_images_batch = rendered_images

        return future_images_batch

    def _get_env_state(self) -> Dict:
        """Save current environment state."""
        return {
            'agent_pos': self.env.agent.position,
            'agent_vel': self.env.agent.velocity,
            'agent_angle': self.env.agent.angle,
            'block_pos': self.env.block.position,
            'block_vel': self.env.block.velocity,
            'block_angle': self.env.block.angle,
        }

    def _set_env_state(self, state: np.ndarray):
        """Set environment state from state vector [5].

        Args:
            state: [agent_x, agent_y, block_x, block_y, block_angle]
        """
        if len(state) >= 5:
            self.env.agent.position = (float(state[0]), float(state[1]))
            self.env.agent.velocity = (0, 0)
            self.env.block.angle = float(state[4])
            self.env.block.position = (float(state[2]), float(state[3]))
            self.env.block.velocity = (0, 0)

    def _set_env_state_dict(self, state_dict: Dict):
        """Restore environment state from state dict."""
        self.env.agent.position = state_dict['agent_pos']
        self.env.agent.velocity = state_dict['agent_vel']
        self.env.agent.angle = state_dict.get('agent_angle', 0)
        self.env.block.angle = state_dict['block_angle']
        self.env.block.position = state_dict['block_pos']
        self.env.block.velocity = state_dict['block_vel']

    def _unwrap_env(self, env):
        """Unwrap environment to get to base PushTImageEnv."""
        current_env = env
        while hasattr(current_env, 'env'):
            current_env = current_env.env
        return current_env


def create_pusht_image_renderer(
    env: Env,
    render_size: int = 96,
    max_renders_per_batch: int | None = None,
) -> PushTImageRenderer:
    """Factory function to create PushT image renderer."""
    return PushTImageRenderer(
        env,
        render_size,
        max_renders_per_batch=max_renders_per_batch,
    )
