"""Image-based policy with cross-attention between current and future rendered states.

Architecture:
1. Current image → CNN encoder → current_features
2. Predict initial_action from current_features
3. Render future image with initial_action (no backprop)
4. Future image → CNN encoder → future_features
5. Cross-attention: future_features (query) attend to current_features (key/value)
6. Fuse attended features → final_action

Author: Claude Code
Date: 2026-02-09
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from mip.networks.base import BaseNetwork


class ImageCrossAttentionPolicy(BaseNetwork):
    """Policy with cross-attention between current and future rendered images."""

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,  # Image shape: C * H * W
        To: int,
        emb_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
        image_channels: int = 3,
        image_size: int = 96,
    ):
        """
        Args:
            act_dim: Action dimension (2 for PushT)
            Ta: Action horizon
            obs_dim: Flattened image size (C * H * W)
            To: Observation horizon (number of image frames)
            emb_dim: Embedding dimension
            n_layers: Number of CNN/MLP layers
            n_heads: Number of attention heads
            dropout: Dropout probability
            image_channels: Number of image channels (3 for RGB)
            image_size: Image height/width (assumes square images)
        """
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim, n_layers)

        self.image_channels = image_channels
        self.image_size = image_size
        self.d_model = emb_dim
        self.n_heads = n_heads

        # CNN Encoder for images [B, To, C, H, W] → [B, To, d_model]
        self.image_encoder = self._build_cnn_encoder(image_channels, emb_dim)

        # Initial action prediction from current image features
        self.action_predictor = nn.Sequential(
            nn.Linear(emb_dim * To + 2, emb_dim),  # +2 for s, t
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, act_dim * Ta),
        )

        # Cross-attention: future (query) attends to current (key/value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(self.d_model)

        # Final action prediction from fused features
        self.action_refiner = nn.Sequential(
            nn.Linear(emb_dim * (To + 1) + 2, emb_dim),  # +1 for future, +2 for s,t
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, act_dim * Ta),
        )

        # Scalar output
        self.scalar_output = nn.Linear(emb_dim * To, 1)

        # Image renderer (injected from outside)
        self.renderer = None

    def _build_cnn_encoder(self, in_channels: int, out_dim: int) -> nn.Module:
        """Build CNN encoder for images.

        Args:
            in_channels: Number of input channels (3 for RGB)
            out_dim: Output embedding dimension

        Returns:
            CNN encoder module
        """
        # Simple CNN: 96x96 → 48x48 → 24x24 → 12x12 → 6x6 → global pool → out_dim
        return nn.Sequential(
            # Conv1: 96x96 → 48x48
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # Conv2: 48x48 → 24x24
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Conv3: 24x24 → 12x12
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # Conv4: 12x12 → 6x6
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_dim),
            # Global average pooling: 6x6 → 1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # [B, out_dim]
        )

    def set_renderer(self, renderer):
        """Set the image renderer.

        Args:
            renderer: PushTImageRenderer instance
        """
        self.renderer = renderer

    def __deepcopy__(self, memo):
        """Custom deepcopy that doesn't copy the renderer (contains pygame objects).

        Uses PyTorch's state_dict mechanism to avoid pickling pygame objects.
        """
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
            n_heads=self.n_heads,
            dropout=getattr(self, 'dropout', 0.0),
            image_channels=self.image_channels,
            image_size=self.image_size,
        )

        # Copy weights using state_dict
        result.load_state_dict(self.state_dict())

        # Move to same device
        result = result.to(device)

        # Don't copy renderer (contains pygame objects)
        result.renderer = None

        return result

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with image cross-attention.

        Args:
            x: Noise actions [B, Ta, act_dim] (for flow matching)
            condition: Image observations [B, To, C, H, W]
            s: Noise level [B]
            t: Timestep [B]

        Returns:
            (action, scalar): Predicted action [B, Ta, act_dim] and scalar [B, 1]
        """
        batch_size = condition.shape[0]
        device = condition.device

        # 1. Encode current images [B, To, C, H, W] → [B, To, d_model]
        current_features = []
        for i in range(self.To):
            img_t = condition[:, i]  # [B, C, H, W]
            feat_t = self.image_encoder(img_t)  # [B, d_model]
            current_features.append(feat_t)
        current_features = torch.stack(current_features, dim=1)  # [B, To, d_model]

        # 2. Predict initial action from current features
        current_flat = current_features.reshape(batch_size, -1)  # [B, To * d_model]
        s_exp = s.reshape(batch_size, 1)
        t_exp = t.reshape(batch_size, 1)
        action_input = torch.cat([current_flat, s_exp, t_exp], dim=-1)

        initial_action_flat = self.action_predictor(action_input)
        initial_action = initial_action_flat.reshape(batch_size, self.Ta, self.act_dim)

        # 3. Render future image using initial action (first timestep only)
        if self.renderer is not None:
            # Prepare observation dict (need to reshape to [B, To, obs_dim])
            # For image renderer, we pass the full condition
            obs_dict = {'image': condition}  # [B, To, C, H, W]

            # Get first action from horizon
            initial_action_first = initial_action[:, 0, :]  # [B, act_dim]

            # Render future image (no backprop!)
            with torch.no_grad():
                future_image = self.renderer(obs_dict, initial_action_first)  # [B, C, H, W]

            # Encode future image
            future_features = self.image_encoder(future_image).unsqueeze(1)  # [B, 1, d_model]

            # 4. Cross-attention: future (query) attends to current (key/value)
            attn_output, attn_weights = self.cross_attention(
                query=future_features,      # [B, 1, d_model] - "What will happen"
                key=current_features,       # [B, To, d_model] - "Given current"
                value=current_features,     # [B, To, d_model]
            )  # Output: [B, 1, d_model]

            # Residual + norm
            future_attended = self.attn_norm(future_features + attn_output)  # [B, 1, d_model]

            # 5. Fuse current + attended future features
            future_flat = future_attended.squeeze(1)  # [B, d_model]
            fused = torch.cat([current_flat, future_flat, s_exp, t_exp], dim=-1)

            # Final action prediction
            final_action_flat = self.action_refiner(fused)
            final_action = final_action_flat.reshape(batch_size, self.Ta, self.act_dim)

            # Store for visualization
            self._last_current_images = condition
            self._last_future_image = future_image
            self._last_attn_weights = attn_weights  # [B, 1, To]
            self._last_action_pred = final_action
        else:
            # No renderer, just use initial action
            final_action = initial_action
            self._last_current_images = None
            self._last_future_image = None
            self._last_attn_weights = None

        # Scalar output
        scalar = self.scalar_output(current_flat)

        return final_action, scalar
