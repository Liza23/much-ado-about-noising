"""VLA-style policy with CLIP vision encoder and cross-attention decoder.

Two-stage predict-render-refine architecture:
1. CLIP ViT encodes current images → tokens
2. Initial action prediction: state (query) cross-attends to CLIP tokens (key/value) → action
3. Renderer places agent at predicted action → future image → CLIP encodes → future tokens
4. Transformer decoder cross-attends to current + future + state tokens → velocity

State is provided as temporal tokens [B, To, emb_dim] via cross-attention (not added to queries).
The initial action prediction + rendering (steps 2-3) only runs when a renderer is present.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor

from mip.networks.base import BaseNetwork


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with separate cross-attention heads."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Cross-attention to current image
        self.cross_attn_current = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_current = nn.LayerNorm(d_model)

        # Cross-attention to future image
        self.cross_attn_future = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_future = nn.LayerNorm(d_model)

        # Cross-attention to language (optional)
        self.cross_attn_lang = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_lang = nn.LayerNorm(d_model)

        # Cross-attention to state tokens (temporal proprioceptive state)
        self.cross_attn_state = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_state = nn.LayerNorm(d_model)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        current_tokens: torch.Tensor,
        future_tokens: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        state_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through decoder layer.

        Args:
            query: [B, horizon, d_model]
            current_tokens: [B, N_current, d_model]
            future_tokens: Optional [B, N_future, d_model] (None = no-render mode)
            lang_tokens: Optional [B, N_lang, d_model]
            state_tokens: Optional [B, To, d_model] (temporal proprioceptive state)

        Returns:
            query: [B, horizon, d_model]
            attn_weights: Dict of attention weights
        """
        attn_weights = {}

        # Cross-attend to current image
        attn_out, attn_w = self.cross_attn_current(
            query=query, key=current_tokens, value=current_tokens
        )
        query = self.norm_current(query + attn_out)
        attn_weights['current'] = attn_w
        attn_weights['current_out_norm'] = attn_out.norm(dim=-1).mean(dim=1)  # [B]

        # Cross-attend to future image (skip in no-render mode)
        if future_tokens is not None:
            attn_out, attn_w = self.cross_attn_future(
                query=query, key=future_tokens, value=future_tokens
            )
            query = self.norm_future(query + attn_out)
            attn_weights['future'] = attn_w
            attn_weights['future_out_norm'] = attn_out.norm(dim=-1).mean(dim=1)  # [B]

        # Cross-attend to language (if provided)
        if lang_tokens is not None:
            attn_out, attn_w = self.cross_attn_lang(
                query=query, key=lang_tokens, value=lang_tokens
            )
            query = self.norm_lang(query + attn_out)
            attn_weights['language'] = attn_w

        # Cross-attend to state tokens (temporal proprioceptive state)
        if state_tokens is not None:
            attn_out, attn_w = self.cross_attn_state(
                query=query, key=state_tokens, value=state_tokens
            )
            query = self.norm_state(query + attn_out)
            attn_weights['state'] = attn_w
            attn_weights['state_out_norm'] = attn_out.norm(dim=-1).mean(dim=1)  # [B]

        # Self-attention
        attn_out, _ = self.self_attn(query=query, key=query, value=query)
        query = self.norm_self(query + attn_out)

        # Feed-forward
        query = self.norm_ffn(query + self.ffn(query))

        return query, attn_weights


class VLACrossAttentionPolicy(BaseNetwork):
    """VLA-style policy with CLIP vision and transformer decoder."""

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        emb_dim: int = 768,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        freeze_vision: bool = True,
        use_language: bool = False,
        image_size: int = 224,
        use_state_in_decoder: bool = True,
    ):
        """
        Args:
            act_dim: Action dimension
            Ta: Action horizon
            obs_dim: Observation dimension (not used, kept for compatibility)
            To: Observation horizon
            emb_dim: Embedding dimension (768 for CLIP ViT-B)
            n_layers: Number of transformer decoder layers
            n_heads: Number of attention heads
            dropout: Dropout probability
            clip_model_name: CLIP model name
            freeze_vision: Whether to freeze CLIP vision encoder
            use_language: Whether to use language conditioning
            image_size: Input image size (CLIP expects 224)
        """
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim, n_layers)

        self.use_language = use_language
        self.image_size = image_size
        self.use_state_in_decoder = use_state_in_decoder
        self.clip_model_name = clip_model_name
        self.freeze_vision = freeze_vision

        # CLIP Vision Encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

        # Optional CLIP Text Encoder
        if use_language:
            self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()
        else:
            self.text_encoder = None

        # Learned action queries (one per timestep in horizon)
        self.action_queries = nn.Parameter(torch.randn(Ta, emb_dim) * 0.02)

        # Timestep embedding (for flow matching)
        self.time_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Noise level embedding (for flow matching, s parameter)
        self.noise_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Noisy action embedding
        self.action_embed = nn.Linear(act_dim, emb_dim)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(emb_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output head: decoder features → velocity
        self.output_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, act_dim),
        )

        # Scalar output (required by base class)
        self.scalar_output = nn.Linear(emb_dim, 1)

        # --- State token projection (temporal state → cross-attention tokens) ---
        self.state_dim = 5  # [agent_x, agent_y, block_x, block_y, block_angle]
        self.state_token_proj = nn.Sequential(
            nn.Linear(self.state_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # --- Initial action prediction (Stage 1): state cross-attends to CLIP ---
        # State (query) cross-attends to CLIP tokens (key/value).
        # This keeps state and image in separate roles so neither dominates.
        self.init_action_query_proj = nn.Linear(self.state_dim, emb_dim)
        self.init_action_cross_attn = nn.MultiheadAttention(
            emb_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.init_action_norm = nn.LayerNorm(emb_dim)
        self.init_action_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.GELU(),
            nn.Linear(256, act_dim),  # outputs raw pixel coords [0, 512]
        )
        # Initialize output bias to workspace center
        with torch.no_grad():
            self.init_action_head[-1].bias.data.fill_(256.0)

        # Image renderer (injected from outside)
        self.renderer = None

        # Render data set per batch by training loop
        self._render_state = None       # [B, To, 5] temporal state for cross-attention + renderer
        self._render_action_gt = None   # [B, 2] raw GT action for draft loss

        # Cache for sampling efficiency (computed once, reused across ODE steps)
        self._cache = None

        # Draft loss (consumed by loss function after forward)
        self._draft_loss = None
        self._draft_action = None

        # Image preprocessing (resize to CLIP size)
        self.resize = nn.Upsample(size=(image_size, image_size), mode='bilinear')

    def set_renderer(self, renderer):
        """Set the image renderer."""
        self.renderer = renderer

    def set_render_data(self, render_state, render_action_gt=None):
        """Set per-batch render data (called by training loop before forward).

        Args:
            render_state: [B, To, 5] temporal state history, or [B, 5] single frame, or None
            render_action_gt: [B, 2] raw GT action for draft loss (training only), or None
        """
        self._render_state = render_state
        self._render_action_gt = render_action_gt

    def clear_cache(self):
        """Clear cached tokens. Must be called before each new batch/sample."""
        self._cache = None

    def __deepcopy__(self, memo):
        """Custom deepcopy that doesn't copy the renderer."""
        device = next(self.parameters()).device

        result = self.__class__(
            act_dim=self.act_dim,
            Ta=self.Ta,
            obs_dim=self.obs_dim,
            To=self.To,
            emb_dim=self.emb_dim,
            n_layers=self.n_layers,
            n_heads=getattr(self, 'n_heads', 8),
            dropout=getattr(self, 'dropout', 0.1),
            clip_model_name=self.clip_model_name,
            freeze_vision=self.freeze_vision,
            use_language=self.use_language,
            image_size=self.image_size,
            use_state_in_decoder=self.use_state_in_decoder,
        )

        result.load_state_dict(self.state_dict())
        result = result.to(device)
        result.renderer = None

        return result

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with CLIP vision encoder.

        Args:
            images: [B, C, H, W] RGB images

        Returns:
            tokens: [B, N, emb_dim] vision tokens
        """
        # Resize to CLIP input size if needed
        if images.shape[-1] != self.image_size:
            images = self.resize(images)

        # CLIP expects [B, C, 224, 224] and values in [-1, 1]
        # Assuming input is already normalized to [0, 1], convert to [-1, 1]
        if images.max() <= 1.0:
            images = images * 2.0 - 1.0

        # Get vision tokens (excluding CLS token)
        if self.freeze_vision:
            with torch.no_grad():
                outputs = self.vision_encoder(pixel_values=images)
        else:
            outputs = self.vision_encoder(pixel_values=images)

        # last_hidden_state: [B, 197, 768] for ViT-B/16 (includes CLS)
        # We'll use all tokens including CLS
        vision_tokens = outputs.last_hidden_state

        return vision_tokens

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Two-stage predict-render-refine forward pass.

        Stage 1 (cached): Encode current → draft action → render → encode future
        Stage 2 (per ODE step): Build queries from noisy action → decoder → velocity

        Args:
            x: Noisy actions [B, Ta, act_dim]
            s: Noise level [B]
            t: Timestep [B]
            condition: Image observations [B, To, C, H, W]

        Returns:
            velocity: [B, Ta, act_dim]
            scalar: [B, 1]
        """
        batch_size = condition.shape[0]
        device = condition.device

        # === STAGE 1: Draft + Render (computed once, cached for ODE steps) ===
        if self._cache is None:
            # 1a. Encode current images
            current_tokens_list = []
            for i in range(self.To):
                img = condition[:, i]  # [B, C, H, W]
                tokens = self.encode_image(img)  # [B, N, emb_dim]
                current_tokens_list.append(tokens)
            current_tokens = torch.cat(current_tokens_list, dim=1)  # [B, To*N, emb_dim]

            # 1b. Normalize temporal state [B, To, 5] → state tokens [B, To, emb_dim]
            if self._render_state is not None:
                state_raw = self._render_state.to(device).clone()
                # Handle both [B, 5] (single frame) and [B, To, 5] (temporal)
                if state_raw.dim() == 2:
                    state_raw = state_raw.unsqueeze(1)  # [B, 1, 5]
                state_input = state_raw[:, :, :self.state_dim]  # [B, To, 5]
                state_input[:, :, :4] = state_input[:, :, :4] / 512.0   # positions → [0, 1]
                state_input[:, :, 4] = state_input[:, :, 4] / (2 * 3.14159)  # angle → ~[0, 1]
            else:
                state_input = torch.zeros(batch_size, 1, self.state_dim, device=device)

            # Project temporal state to tokens for cross-attention
            state_tokens = self.state_token_proj(state_input)  # [B, To, emb_dim]

            # 1c-d. Initial action prediction + rendering (only when renderer is available)
            draft_action = None
            future_image = None
            future_tokens = None

            if self.renderer is not None:
                # Use last frame for draft action prediction
                state_last = state_input[:, -1, :]  # [B, 5]
                # State as query, CLIP tokens as key/value
                state_q = self.init_action_query_proj(state_last)  # [B, emb_dim]
                state_q = state_q.unsqueeze(1)  # [B, 1, emb_dim]
                attn_out, draft_attn_w = self.init_action_cross_attn(
                    query=state_q, key=current_tokens, value=current_tokens
                )
                init_features = self.init_action_norm(state_q + attn_out)  # [B, 1, emb_dim]
                draft_action = self.init_action_head(init_features.squeeze(1))  # [B, act_dim]

                # Compute draft loss if GT available (training only)
                if self._render_action_gt is not None:
                    self._draft_loss = F.mse_loss(draft_action / 512.0, self._render_action_gt / 512.0)
                else:
                    self._draft_loss = None

                # Render future image using PREDICTED action (not GT)
                if self._render_state is not None:
                    # Renderer expects raw [B, 5] state (last frame)
                    render_state_raw = self._render_state.to(device)
                    if render_state_raw.dim() == 3:
                        render_state_raw = render_state_raw[:, -1, :]  # [B, 5]
                    with torch.no_grad():
                        future_image = self.renderer(
                            render_state_raw, draft_action.detach()
                        )  # [B, C, H, W]
                    future_tokens = self.encode_image(future_image)  # [B, N, emb_dim]

            self._cache = {
                'current_tokens': current_tokens,
                'future_tokens': future_tokens,
                'future_image': future_image,
                'draft_action': draft_action,
                'draft_attn_weights': draft_attn_w.detach() if draft_action is not None else None,
                'state_tokens': state_tokens,
            }
            self._draft_action = draft_action.detach() if draft_action is not None else None

        current_tokens = self._cache['current_tokens']
        future_tokens = self._cache['future_tokens']
        state_tokens = self._cache['state_tokens']

        # === STAGE 2: Flow matching decoder (runs each ODE step) ===

        # 2a. Build action queries (no state injection — state flows through cross-attention)
        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Ta, emb_dim]
        x_embed = self.action_embed(x)  # [B, Ta, emb_dim]
        queries = queries + x_embed
        t_embed = self.time_embed(t.reshape(batch_size, 1))  # [B, emb_dim]
        queries = queries + t_embed.unsqueeze(1)
        s_embed = self.noise_embed(s.reshape(batch_size, 1))  # [B, emb_dim]
        queries = queries + s_embed.unsqueeze(1)

        # 2b. Transformer decoder with cross-attention
        # Gate state tokens: only pass to decoder if use_state_in_decoder is True
        decoder_state_tokens = state_tokens if self.use_state_in_decoder else None
        all_attn_weights = []
        for layer in self.decoder_layers:
            queries, attn_weights = layer(
                query=queries,
                current_tokens=current_tokens,
                future_tokens=future_tokens,
                lang_tokens=None,
                state_tokens=decoder_state_tokens,
            )
            all_attn_weights.append(attn_weights)

        # 2c. Output velocity
        velocity = self.output_head(queries)  # [B, Ta, act_dim]
        scalar = self.scalar_output(queries.mean(dim=1))  # [B, 1]

        # Store for visualization / debugging
        self._last_current_images = condition
        self._last_future_image = self._cache.get('future_image', None)
        self._last_attn_weights = all_attn_weights
        self._last_velocity_pred = velocity
        self._last_draft_action = self._cache.get('draft_action', None)
        self._last_draft_attn_weights = self._cache.get('draft_attn_weights', None)
        self._last_state_tokens = state_tokens

        return velocity, scalar
