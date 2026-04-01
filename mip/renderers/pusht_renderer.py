"""Simulator-based geometric renderer for PushT (v0).

This version uses Pymunk physics queries for accurate geometric features.
v1 will use analytical calculations for speed.

Author: Claude Code
Date: 2026-02-08
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Callable


class PushTSimRenderer(nn.Module):
    """Uses PushT's Pymunk simulator for geometric feature extraction.

    Given an observation and predicted action, queries the physics simulator
    to compute geometric plausibility features without stepping the simulation.
    """

    def __init__(self, env_fn: Callable, verbose: bool = False):
        """
        Args:
            env_fn: Callable that creates a PushT environment instance
            verbose: If True, print debug information
        """
        super().__init__()
        self.env = env_fn()  # Dedicated env for geometry queries
        self.verbose = verbose
        self.world_size = 512.0

        if self.verbose:
            print(f"[PushTSimRenderer] Initialized with env: {type(self.env)}")

    def forward(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute geometric features for predicted actions.

        Args:
            obs: Dictionary containing 'state' [B, obs_horizon, 5]
                 where state is [agent_x, agent_y, block_x, block_y, block_angle]
            action: Predicted actions [B, 2] (target pusher positions)

        Returns:
            Dictionary of geometric features:
                - dist_to_block: [B] distance from action to block center
                - will_contact: [B] binary contact prediction
                - penetration_depth: [B] overlap depth if in contact
                - contact_normal_alignment: [B] push direction quality (-1 to 1)
                - num_contact_points: [B] number of contact points
        """
        device = action.device
        batch_size = action.shape[0]

        # Initialize feature storage
        features = {
            'dist_to_block': [],
            'will_contact': [],
            'penetration_depth': [],
            'contact_normal_alignment': [],
            'num_contact_points': [],
        }

        # Process each batch element sequentially
        # (Pymunk is not batched - this is the v0 limitation)
        for i in range(batch_size):
            # Extract state for this sample
            state = obs['state'][i, -1].detach().cpu().numpy()  # [5] - use latest timestep
            action_i = action[i].detach().cpu().numpy()  # [2]

            # Query simulator
            feats = self._query_sim_geometry(state, action_i)

            # Collect features
            for k, v in feats.items():
                features[k].append(v)

        # Convert lists to tensors
        features_tensor = {}
        for k, v in features.items():
            features_tensor[k] = torch.tensor(v, device=device, dtype=torch.float32)

        return features_tensor

    def _query_sim_geometry(self, state: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        """Query Pymunk for geometric features without side effects.

        Args:
            state: [5] current state [agent_x, agent_y, block_x, block_y, block_angle]
            action: [2] target position [x, y]

        Returns:
            Dictionary of scalar geometric features
        """
        # === Save original state ===
        original_agent_pos = self.env.agent.position
        original_agent_vel = self.env.agent.velocity
        original_block_pos = self.env.block.position
        original_block_angle = self.env.block.angle

        # === Set environment state (without physics step) ===
        # Manually set positions instead of using _set_state (which steps physics)
        self.env.agent.position = tuple(state[:2].tolist())
        self.env.agent.velocity = (0, 0)  # reset velocity
        self.env.block.angle = float(state[4])
        self.env.block.position = tuple(state[2:4].tolist())

        # === Move agent to predicted action position ===
        self.env.agent.position = tuple(action.tolist())

        # === Query Pymunk geometry ===

        # 1. Distance to block (center-to-center)
        agent_pos = np.array(self.env.agent.position)
        block_pos = np.array(self.env.block.position)
        dist_to_block = float(np.linalg.norm(agent_pos - block_pos))

        # 2. Check for collisions using shape query
        agent_shape = list(self.env.agent.shapes)[0]  # shapes is a set, convert to list
        contact_set = self.env.space.shape_query(agent_shape)

        # 3. Parse contact information
        will_contact = len(contact_set) > 0
        num_contacts = len(contact_set)

        # Simplified: use distance-based heuristics instead of detailed contact info
        pusher_radius = 15.0  # from PushT env
        block_halfwidth = 50.0  # approximate from PushT
        penetration_depth = float(max(0.0, pusher_radius + block_halfwidth - dist_to_block))

        # Compute push direction quality (heuristic)
        push_vec = action - state[:2]  # direction from current to target
        push_magnitude = np.linalg.norm(push_vec)

        if push_magnitude > 1e-6 and will_contact:
            push_dir = push_vec / push_magnitude
            block_to_agent = agent_pos - block_pos
            block_to_agent_norm = block_to_agent / (np.linalg.norm(block_to_agent) + 1e-6)
            # Alignment: how well does push direction align with vector from block to agent
            normal_alignment = float(np.dot(push_dir, block_to_agent_norm))
        else:
            normal_alignment = 0.0

        # === Restore original state ===
        self.env.agent.position = original_agent_pos
        self.env.agent.velocity = original_agent_vel
        self.env.block.position = original_block_pos
        self.env.block.angle = original_block_angle

        return {
            'dist_to_block': dist_to_block,
            'will_contact': float(will_contact),
            'penetration_depth': penetration_depth,
            'contact_normal_alignment': normal_alignment,
            'num_contact_points': float(num_contacts),
        }

    def get_feature_dim(self) -> int:
        """Return dimension of flattened geometric features."""
        return 5  # dist, contact, penetration, normal_align, num_contacts


def create_pusht_renderer(task_config) -> PushTSimRenderer:
    """Factory function to create renderer from task config.

    Args:
        task_config: Task configuration object

    Returns:
        PushTSimRenderer instance
    """
    from mip.envs.pusht.pusht_env import PushTEnv

    # Create env factory function
    def env_fn():
        env = PushTEnv(
            legacy=False,
            render_size=getattr(task_config, 'render_size', 96),
        )
        # Reset to initialize the environment
        env.reset()
        return env

    return PushTSimRenderer(env_fn, verbose=False)
