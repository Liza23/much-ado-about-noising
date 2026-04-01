"""Action prediction network with geometric verification (v0).

This network architecture:
1. Encodes observations into features
2. Predicts initial action
3. Renders geometric features using predicted action
4. Combines image + geometric features
5. Outputs final action

Author: Claude Code
Date: 2026-02-08
"""

import copy
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from mip.networks.base import BaseNetwork


class GeometricActionNetwork(BaseNetwork):
    """Policy network with geometric renderer feedback.

    Architecture:
        obs → encoder → initial_action
        initial_action + obs → renderer → geometric_features
        [encoder_features, geometric_features] → final_action
    """

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        emb_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        """
        Args:
            act_dim: Action dimension
            Ta: Action horizon
            obs_dim: Observation dimension (per timestep)
            To: Observation horizon
            emb_dim: Embedding dimension for hidden layers
            n_layers: Number of layers in MLPs
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim, n_layers)

        self.dropout = dropout

        # Input dimension: action + observation + s + t
        input_dim = act_dim * Ta + obs_dim * To + 2  # +2 for s and t
        output_dim = act_dim * Ta

        # Main MLP
        layers = []
        layers.append(nn.Linear(input_dim, emb_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers):
            layers.append(nn.Linear(emb_dim, emb_dim))
            layers.append(nn.ReLU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(emb_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # Geometric feature dimension (from renderer)
        self.geom_dim = 5  # dist, contact, penetration, normal_align, num_contacts

        # Cross-attention: rendered geometry (query) attends to observation (key/value)
        self.d_model = emb_dim  # embedding dimension for attention
        self.n_heads = 4  # number of attention heads

        # Project observation to key/value tokens (To timesteps become tokens)
        self.obs_to_tokens = nn.Linear(obs_dim, self.d_model)

        # Project each geometric feature to query space
        # Each of the 5 geometric features becomes a query token
        self.geom_to_query = nn.Linear(1, self.d_model)

        # Multi-head cross-attention: geometry queries attend to observation tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=dropout,
            batch_first=True  # [B, T, D] format
        )

        # Layer norm after cross-attention
        self.attn_norm = nn.LayerNorm(self.d_model)

        # Fuse: attended geometric features + MLP features → action
        self.action_output = nn.Linear(emb_dim + self.d_model * self.geom_dim, output_dim)
        self.scalar_output = nn.Linear(emb_dim, 1)

        # Renderer will be injected from outside
        self.renderer = None

    def set_renderer(self, renderer):
        """Set the geometric renderer.

        Args:
            renderer: PushTSimRenderer instance
        """
        self.renderer = renderer

    def __deepcopy__(self, memo):
        """Custom deepcopy that doesn't copy the renderer (it's not picklable).

        Uses PyTorch's state_dict mechanism to avoid pickling issues.
        """
        print(f"[GeometricPolicy] __deepcopy__ called, using state_dict approach")

        # Get device of current network
        device = next(self.parameters()).device

        # Create new instance with same parameters
        result = self.__class__(
            act_dim=self.act_dim,
            Ta=self.Ta,
            obs_dim=self.obs_dim,
            To=self.To,
            emb_dim=self.emb_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Copy weights using state_dict (doesn't include renderer)
        result.load_state_dict(self.state_dict())

        # Move to same device as original
        result = result.to(device)

        # Renderer is intentionally NOT copied (set to None)
        result.renderer = None

        memo[id(self)] = result
        print(f"[GeometricPolicy] __deepcopy__ complete using state_dict, device={device}")
        return result

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with geometric verification.

        Args:
            x: Action tensor [B, Ta, act_dim]
            s: Start time [B]
            t: End time [B]
            condition: Observation condition [B, To, obs_dim]

        Returns:
            action: [B, Ta, act_dim] action prediction
            scalar: [B, 1] scalar output
        """
        batch_size = x.shape[0]

        # Flatten inputs for MLP
        x_flat = x.reshape(batch_size, -1)  # [B, Ta * act_dim]
        cond_flat = condition.reshape(batch_size, -1)  # [B, To * obs_dim]
        s_expanded = s.reshape(batch_size, 1)  # [B, 1]
        t_expanded = t.reshape(batch_size, 1)  # [B, 1]

        # Concatenate all inputs
        mlp_input = torch.cat([x_flat, cond_flat, s_expanded, t_expanded], dim=-1)

        # Debug: print shapes on first call
        if not hasattr(self, '_shapes_printed'):
            print(f"[GeometricPolicy Debug]")
            print(f"  x.shape: {x.shape}")
            print(f"  condition.shape: {condition.shape}")
            print(f"  s.shape: {s.shape}, t.shape: {t.shape}")
            print(f"  x_flat.shape: {x_flat.shape}")
            print(f"  cond_flat.shape: {cond_flat.shape}")
            print(f"  mlp_input.shape: {mlp_input.shape}")
            print(f"  Expected input_dim: {self.act_dim * self.Ta + self.obs_dim * self.To + 2}")
            self._shapes_printed = True

        # Forward through MLP
        features = self.mlp(mlp_input)  # [B, emb_dim]

        if not hasattr(self, '_forward_printed'):
            print(f"[GeometricPolicy] After MLP: features.shape = {features.shape}")
            self._forward_printed = True

        # Use geometric renderer if available
        if self.renderer is not None:
            # Predict initial action from MLP features only
            initial_action_flat = self.action_output(torch.cat([features, torch.zeros(batch_size, self.geom_dim, device=x.device)], dim=-1))
            initial_action = initial_action_flat.reshape(batch_size, self.Ta, self.act_dim)

            # Take first action from horizon for renderer (renderer only uses single action)
            initial_action_first = initial_action[:, 0, :]  # [B, 2]

            # Prepare observation dict for renderer (needs state with shape [B, obs_horizon, obs_dim])
            obs_dict = {'state': condition}  # condition is [B, To, obs_dim]

            # Get geometric features from renderer
            if not hasattr(self, '_renderer_called'):
                print(f"[GeometricPolicy] Calling renderer with action shape: {initial_action_first.shape}")
                self._renderer_called = True

            geom_features_dict = self.renderer(obs_dict, initial_action_first)

            # Stack geometric features into tensor
            geom_features = torch.stack([
                geom_features_dict['dist_to_block'] / 512.0,  # normalize
                geom_features_dict['will_contact'],
                geom_features_dict['penetration_depth'] / 50.0,  # normalize
                geom_features_dict['contact_normal_alignment'],
                geom_features_dict['num_contact_points'] / 10.0,  # normalize
            ], dim=-1)  # [B, 5]

            if not hasattr(self, '_geom_features_printed'):
                print(f"[GeometricPolicy] Geometric features shape: {geom_features.shape}")
                print(f"[GeometricPolicy] Sample geom features: {geom_features[0]}")
                self._geom_features_printed = True
        else:
            # No renderer available, predict action from features only
            initial_action_flat = self.action_output(torch.cat([features, torch.zeros(batch_size, self.geom_dim, device=x.device)], dim=-1))
            initial_action = initial_action_flat.reshape(batch_size, self.Ta, self.act_dim)
            # Use zero geometric features (no information)
            geom_features = torch.zeros(batch_size, self.geom_dim, device=x.device)
            geom_features_dict = None

        # === CROSS-ATTENTION: Geometric features attend to observation ===

        # 1. Create observation tokens from condition [B, To, obs_dim]
        # Each observation timestep becomes a token
        obs_tokens = self.obs_to_tokens(condition)  # [B, To, d_model]

        # 2. Create geometric queries [B, 5, d_model]
        # Each geometric feature becomes a query token
        geom_expanded = geom_features.unsqueeze(-1)  # [B, 5, 1]
        geom_queries = self.geom_to_query(geom_expanded)  # [B, 5, d_model]

        if not hasattr(self, '_cross_attn_printed'):
            print(f"[GeometricPolicy] Cross-attention shapes:")
            print(f"  obs_tokens (key/value): {obs_tokens.shape}")
            print(f"  geom_queries: {geom_queries.shape}")
            self._cross_attn_printed = True

        # 3. Cross-attention: geometric features (query) attend to observations (key/value)
        # "Given the predicted action's geometry, what in the observation is relevant?"
        attn_output, attn_weights = self.cross_attention(
            query=geom_queries,      # [B, 5, d_model] - geometric features ask
            key=obs_tokens,          # [B, To, d_model] - observation provides context
            value=obs_tokens,        # [B, To, d_model]
        )  # Output: [B, 5, d_model]

        # 4. Residual + layer norm
        geom_attended = self.attn_norm(geom_queries + attn_output)  # [B, 5, d_model]

        # 5. Flatten attended geometric features
        geom_attended_flat = geom_attended.reshape(batch_size, -1)  # [B, 5 * d_model]

        # 6. Concatenate MLP features with attended geometric features
        combined = torch.cat([features, geom_attended_flat], dim=-1)  # [B, emb_dim + 5*d_model]

        if not hasattr(self, '_combined_printed'):
            print(f"[GeometricPolicy] combined features shape: {combined.shape}")
            self._combined_printed = True

        # 7. Final action prediction
        action_output = self.action_output(combined)  # [B, Ta * act_dim]
        action_output = action_output.reshape(batch_size, self.Ta, self.act_dim)  # [B, Ta, act_dim]

        if not hasattr(self, '_action_printed'):
            print(f"[GeometricPolicy] action_output.shape = {action_output.shape}")
            self._action_printed = True

        # Scalar output
        scalar = self.scalar_output(features)

        if not hasattr(self, '_scalar_printed'):
            print(f"[GeometricPolicy] scalar.shape = {scalar.shape}")
            print(f"[GeometricPolicy] Forward pass complete!")
            self._scalar_printed = True

        # Store geometric features and attention for visualization (last batch only)
        if self.renderer is not None and geom_features_dict is not None:
            self._last_geom_features = geom_features_dict
            self._last_condition = condition
            self._last_action_pred = action_output
            self._last_attn_weights = attn_weights  # [B, 5, To] - geom features attending to obs timesteps
        else:
            self._last_geom_features = None
            self._last_attn_weights = None

        return action_output, scalar
