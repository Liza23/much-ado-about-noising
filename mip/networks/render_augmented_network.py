"""Render-Augmented Network wrapper for two-stage MIP with rendering between steps.

Wraps any base network (ChiTransformer, MLP, etc.) and intercepts forward calls
to inject rendered future image conditioning between MIP step 1 and step 2.

Architecture:
  Training (single forward call per batch step):
    1. draft_head(obs_emb) → draft_action         ← supervised by GT action (draft_loss)
    2. renderer(state, draft_action) → future_img
    3. future_encoder(future_img) → future_emb
    4. fusion(cat([obs_emb, future_emb])) → obs_emb_enhanced
    5. base_net(xs, s, t, obs_emb_enhanced) → (velocity, scalar)  ← main loss

  Inference (two forward calls per ODE step due to MIP sampler):
    Call 1 (s=0, act=zeros): draft_head → render → enhance → cache; base_net(label) → draft
    Call 2 (t=0.9, act=noisy): base_net(label_cached) → refined

The future image encoder is a separate ResNet18 (independent weights) with the same
architecture as the ResNet18 backbone inside the main MultiImageObsEncoder.

This allows standard mip_loss, mip_sampler, and FlowMap to work unchanged.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_future_encoder(obs_dim: int, use_group_norm: bool = True) -> nn.Module:
    """Build a ResNet18-based future image encoder.

    Matches the architecture used inside MultiImageObsEncoder:
      ResNet18 (fc removed) → 512-dim → Linear(512, obs_dim)

    Args:
        obs_dim: Output embedding dimension (matches main encoder output)
        use_group_norm: Replace BatchNorm with GroupNorm (matches pusht_base config)
    """
    from mip.encoders import get_resnet, replace_submodules
    resnet = get_resnet('resnet18')  # ResNet18 with fc=Identity → 512-dim output

    if use_group_norm:
        resnet = replace_submodules(
            root_module=resnet,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features // 16,
                num_channels=x.num_features,
            ),
        )

    return nn.Sequential(
        resnet,                      # [B, 3, H, W] → [B, 512]
        nn.Linear(512, obs_dim),     # [B, 512] → [B, obs_dim]
    )


class RenderAugmentedNetwork(nn.Module):
    """Wraps any base network to add render-between-steps augmentation.

    The wrapper sits at the FlowMap.net level so that all existing hooks
    (set_render_data, clear_cache, _draft_loss) in train_robomimic.py work
    automatically without any changes to the training loop.

    The future image encoder is a separate ResNet18 (same architecture as the main
    encoder's backbone) with independent weights, followed by a Linear projection
    and a fusion layer.

    A separate draft_head MLP predicts the action from obs_emb; this is supervised
    by GT action (draft_loss) and its output is used for rendering—NOT the velocity
    field output of the base network.
    """

    def __init__(self, base_network: nn.Module, obs_dim: int, act_dim: int,
                 future_image_size: int = 96, use_group_norm: bool = True):
        super().__init__()
        self.base_network = base_network
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.future_image_size = future_image_size  # stored for reference/logging

        # Draft head: obs_emb → draft action (not the velocity field!)
        # Supervised by GT action via draft_loss; output used for rendering.
        self.draft_head = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),
            nn.Linear(256, act_dim),
        )

        # Future image encoder: ResNet18 → Linear(512, obs_dim)
        # Same architecture as main MultiImageObsEncoder backbone; independent weights
        self.future_encoder = _build_future_encoder(obs_dim, use_group_norm=use_group_norm)

        # Fuse [obs_emb; future_emb] → obs_emb (same dim, so base_network is unchanged)
        self.fusion = nn.Linear(2 * obs_dim, obs_dim)

        # Non-parametric state (injected externally, not in state_dict)
        self.renderer = None
        self.action_normalizer = None

        # Per-batch data (set before each forward pass)
        self._render_state = None       # [B, To, 5] raw pixel coords
        self._render_action_gt = None   # [B, act_dim] GT action (normalized) for draft loss

        # Cache: built on call 1 during inference, consumed on call 2, cleared before each batch
        self._cache = None

        # Draft loss (picked up by update_impl)
        self._draft_loss = None

    # ------------------------------------------------------------------
    # External injection methods (called by agent.py and train_robomimic.py)
    # ------------------------------------------------------------------

    def set_renderer(self, renderer):
        """Inject the renderer (not a parameter, excluded from deepcopy)."""
        self.renderer = renderer

    def set_action_normalizer(self, normalizer):
        """Inject action normalizer for unnormalizing draft to world coords."""
        self.action_normalizer = normalizer

    def set_render_data(self, render_state, render_action_gt=None):
        """Set per-batch render data. Called before each training/eval step.

        Args:
            render_state: [B, To, 5] raw state in world/pixel coords, or None
            render_action_gt: [B, act_dim] GT action (normalized) for draft loss, or None
        """
        self._render_state = render_state
        self._render_action_gt = render_action_gt

    def clear_cache(self):
        """Clear the inter-step cache. Must be called before each new observation."""
        self._cache = None
        self._draft_loss = None
        self._render_state = None       # prevent stale state leaking if set_render_data is skipped
        self._render_action_gt = None   # hygiene: GT only valid within the step it was set

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, xs: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                label: torch.Tensor):
        """Forward pass with render augmentation.

        During training (single call per batch): always computes draft → render →
        enhance, then passes label_enhanced to base_network. This is the key fix:
        label_enhanced IS used in the main loss computation.

        During inference (two calls per ODE step via MIP sampler):
          Call 1 (cache is None): draft → render → enhance → cache; return base_net output
          Call 2 (cache exists): use cached label_enhanced; return base_net output

        Args:
            xs:    Noisy action [B, Ta, act_dim]
            s:     Source time scalar or [B] tensor
            t:     Target time scalar or [B] tensor
            label: Observation embedding [B, obs_dim] or [B, To, obs_dim]

        Returns:
            (velocity [B, Ta, act_dim], scalar [B] or scalar)
        """
        if self._cache is not None:
            # ---- Inference call 2: use cached enhanced obs_emb ----
            return self.base_network(xs, s, t, self._cache)

        # ---- Compute label_enhanced (training always lands here; inference call 1 too) ----
        label_enhanced = label  # fallback if no renderer

        if self.renderer is not None and self._render_state is not None:
            # Use the last timestep of render_state for rendering
            render_state_now = self._render_state[:, -1, :]  # [B, state_dim]

            # Draft head: obs_emb → draft action (normalized)
            # For [B, To, obs_dim] labels, use the mean over time axis for the draft head
            if label.dim() == 3:
                label_flat = label.mean(dim=1)  # [B, obs_dim]
            else:
                label_flat = label              # [B, obs_dim]

            draft_action = self.draft_head(label_flat)    # [B, act_dim]

            # Unnormalize draft to world coords for rendering
            draft_pixel = self._unnormalize(draft_action.detach())  # [B, act_dim]

            # Render: future image showing where agent will be
            with torch.no_grad():
                future_img = self.renderer(render_state_now, draft_pixel)  # [B, 3, H, W]

            # Encode future image with separate ResNet18 (same arch, independent weights)
            future_emb = self.future_encoder(future_img)     # [B, obs_dim]

            # Fuse current obs + future obs
            # label may be [B, obs_dim] (ChiTransformer) or [B, To, obs_dim] (ChiUNet)
            if label.dim() == 3:
                future_emb_exp = future_emb.unsqueeze(1).expand_as(label)
                label_enhanced = self.fusion(
                    torch.cat([label, future_emb_exp], dim=-1)
                )                                             # [B, To, obs_dim]
            else:
                label_enhanced = self.fusion(
                    torch.cat([label, future_emb], dim=-1)
                )                                             # [B, obs_dim]

            # Draft supervision: MSE(draft_action, GT_action)
            if self._render_action_gt is not None:
                self._draft_loss = F.mse_loss(
                    draft_action,
                    self._render_action_gt.float(),
                )

        # Cache enhanced obs for inference call 2
        self._cache = label_enhanced

        # Main forward with enhanced label (key fix: was using original label before)
        return self.base_network(xs, s, t, label_enhanced)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unnormalize(self, act_norm: torch.Tensor) -> torch.Tensor:
        """Unnormalize action from normalized space to world coords.

        Uses action_normalizer if available, otherwise falls back to linear scaling.
        """
        if self.action_normalizer is not None:
            try:
                return self.action_normalizer.unnormalize(act_norm)
            except Exception:
                pass
        # Fallback: assume linear mapping [-1,1] → [0,512]
        return (act_norm + 1.0) / 2.0 * 512.0

    def __deepcopy__(self, memo):
        """Custom deepcopy: copies all parametric components, excludes renderer.

        The renderer is non-parametric (just a callable) and is re-injected by
        agent.py after deepcopy, identical to VLACrossAttentionPolicy pattern.
        draft_head, future_encoder and fusion ARE deepcopied (learnable parameters, get EMA'd).
        """
        # Deep-copy base network (may have its own __deepcopy__)
        if hasattr(self.base_network, '__deepcopy__'):
            base_net_copy = self.base_network.__deepcopy__(memo)
        else:
            base_net_copy = copy.deepcopy(self.base_network, memo)

        # Create new wrapper
        new_obj = self.__class__.__new__(self.__class__)
        nn.Module.__init__(new_obj)
        memo[id(self)] = new_obj

        # Copy all parametric modules (in state_dict, get EMA'd)
        new_obj.base_network = base_net_copy
        new_obj.obs_dim = self.obs_dim
        new_obj.act_dim = self.act_dim
        new_obj.future_image_size = self.future_image_size
        new_obj.draft_head = copy.deepcopy(self.draft_head, memo)
        new_obj.future_encoder = copy.deepcopy(self.future_encoder, memo)
        new_obj.fusion = copy.deepcopy(self.fusion, memo)

        # Exclude non-parametric references (re-injected after deepcopy)
        new_obj.renderer = None
        new_obj.action_normalizer = None

        # Reset transient state
        new_obj._render_state = None
        new_obj._render_action_gt = None
        new_obj._cache = None
        new_obj._draft_loss = None

        return new_obj
