"""Utility functions for setting up, save and load networks.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import loguru
import torch
import torch.nn as nn

from mip.config import NetworkConfig, TaskConfig
from mip.encoders import (
    IdentityEncoder,
    MLPEncoder,
    MultiImageObsEncoder,
    PerStepMLPEncoder,
)


def get_network(network_config: NetworkConfig, task_config: TaskConfig):
    # Import inside function to avoid circular imports
    from mip.networks.chitfm import ChiTransformer
    from mip.networks.chiunet import ChiUNet
    from mip.networks.jannerunet import JannerUNet
    from mip.networks.mlp import MLP, VanillaMLP
    from mip.networks.rnn import RNN, VanillaRNN
    from mip.networks.sudeepdit import SudeepDiT
    from mip.networks.geometric_policy import GeometricActionNetwork
    from mip.networks.image_crossattn_policy import ImageCrossAttentionPolicy
    from mip.networks.vla_crossattn_policy import VLACrossAttentionPolicy

    # Handle geometric_policy separately (needs special setup)
    if network_config.network_type == "geometric_policy":
        net = GeometricActionNetwork(
            act_dim=task_config.act_dim,
            Ta=task_config.horizon,
            obs_dim=task_config.obs_dim,
            To=task_config.obs_steps,
            emb_dim=network_config.emb_dim,
            n_layers=network_config.num_layers,
            dropout=getattr(network_config, 'dropout', 0.0),
            use_layer_norm=getattr(network_config, 'use_layer_norm', False),
        )

        # Note: Renderer is now created in agent.py to avoid pickle issues with EMA
        # Don't inject renderer into network here
        return net

    # Handle image_crossattn_policy separately (needs special setup)
    if network_config.network_type == "image_crossattn_policy":
        net = ImageCrossAttentionPolicy(
            act_dim=task_config.act_dim,
            Ta=task_config.horizon,
            obs_dim=task_config.obs_dim,
            To=task_config.obs_steps,
            emb_dim=network_config.emb_dim,
            n_layers=network_config.num_layers,
            n_heads=getattr(network_config, 'n_heads', 4),
            dropout=getattr(network_config, 'dropout', 0.0),
            image_channels=getattr(task_config, 'image_channels', 3),
            image_size=getattr(task_config, 'render_size', 96),
        )

        # Renderer will be injected in agent.py
        return net

    # Handle vla_crossattn_policy (VLA with CLIP + cross-attention decoder)
    if network_config.network_type == "vla_crossattn_policy":
        net = VLACrossAttentionPolicy(
            act_dim=task_config.act_dim,
            Ta=task_config.horizon,
            obs_dim=task_config.obs_dim,
            To=task_config.obs_steps,
            emb_dim=getattr(network_config, 'emb_dim', 768),
            n_layers=getattr(network_config, 'num_layers', 6),
            n_heads=getattr(network_config, 'n_heads', 8),
            dropout=getattr(network_config, 'dropout', 0.1),
            clip_model_name=getattr(network_config, 'clip_model', 'openai/clip-vit-base-patch16'),
            freeze_vision=getattr(network_config, 'freeze_vision', True),
            use_language=getattr(network_config, 'use_language', False),
            image_size=getattr(network_config, 'image_size', 224),
            use_state_in_decoder=getattr(network_config, 'use_state_in_decoder', True),
        )

        # Renderer will be injected in agent.py
        return net

    network_class = {
        "mlp": MLP,
        "vanilla_mlp": VanillaMLP,
        "chitransformer": ChiTransformer,
        "chiunet": ChiUNet,
        "jannerunet": JannerUNet,
        "rnn": RNN,
        "vanilla_rnn": VanillaRNN,
        "sudeepdit": SudeepDiT,
    }[network_config.network_type]

    # Common parameters for all networks
    common_params = {
        "act_dim": task_config.act_dim,
        "Ta": task_config.horizon,
        "obs_dim": network_config.emb_dim,  # all encoder will encode obs to emb_dim
        "To": task_config.obs_steps,
    }

    if network_config.network_type == "mlp":
        net = network_class(
            **common_params,
            emb_dim=network_config.emb_dim,
            n_layers=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_dim=network_config.timestep_emb_dim,
        )
    elif network_config.network_type == "vanilla_mlp":
        net = network_class(
            **common_params,
            emb_dim=network_config.emb_dim,
            n_layers=network_config.num_layers,
            dropout=network_config.dropout,
        )
    elif network_config.network_type == "chitransformer":
        net = network_class(
            **common_params,
            d_model=network_config.emb_dim,
            nhead=network_config.n_heads,
            num_layers=network_config.num_layers,
            p_drop_emb=network_config.dropout,
            p_drop_attn=network_config.attn_dropout,
            n_cond_layers=network_config.n_cond_layers,
            timestep_emb_type=network_config.timestep_emb_type,
        )
    elif network_config.network_type == "chiunet":
        net = network_class(
            **common_params,
            model_dim=network_config.model_dim,
            emb_dim=network_config.emb_dim,
            kernel_size=network_config.kernel_size,
            cond_predict_scale=network_config.cond_predict_scale,
            obs_as_global_cond=network_config.obs_as_global_cond,
            dim_mult=network_config.dim_mult,
            timestep_emb_type=network_config.timestep_emb_type,
        )
    elif network_config.network_type == "jannerunet":
        net = network_class(
            **common_params,
            model_dim=network_config.model_dim,
            emb_dim=network_config.emb_dim,
            kernel_size=network_config.kernel_size,
            dim_mult=network_config.dim_mult,
            norm_type=network_config.norm_type,
            attention=network_config.attention,
            timestep_emb_type=network_config.timestep_emb_type,
        )
    elif network_config.network_type in ["rnn", "vanilla_rnn"]:
        rnn_params = {
            **common_params,
            "rnn_hidden_dim": network_config.emb_dim,
            "rnn_num_layers": network_config.num_layers,
            "rnn_type": network_config.rnn_type,
            "dropout": network_config.dropout,
        }
        if network_config.network_type == "rnn":
            rnn_params.update(
                {
                    "timestep_emb_dim": network_config.timestep_emb_dim,
                    "max_freq": network_config.max_freq,
                }
            )
        net = network_class(**rnn_params)
    elif network_config.network_type == "sudeepdit":
        net = network_class(
            **common_params,
            d_model=network_config.emb_dim,
            n_heads=network_config.n_heads,
            depth=network_config.num_layers,
            dropout=network_config.dropout,
            timestep_emb_type=network_config.timestep_emb_type,
        )

    # Optionally wrap with RenderAugmentedNetwork for render-augmented MIP
    # obs_dim = emb_dim because the encoder is configured to output emb_dim
    if getattr(network_config, 'use_render_augmentation', False):
        from mip.networks.render_augmented_network import RenderAugmentedNetwork
        render_size = getattr(task_config, 'render_size', 96)
        net = RenderAugmentedNetwork(
            net,
            obs_dim=network_config.emb_dim,
            act_dim=task_config.act_dim,
            future_image_size=render_size,
        )
        loguru.logger.info(
            f"[Network] Wrapped with RenderAugmentedNetwork "
            f"(obs_dim={network_config.emb_dim}, act_dim={task_config.act_dim}, "
            f"future_image_size={render_size})"
        )

    return net


def get_encoder(network_config: NetworkConfig, task_config: TaskConfig):
    # Special handling for networks with built-in encoders (e.g., VLA with CLIP)
    if network_config.network_type in ["vla_crossattn_policy", "image_crossattn_policy"]:
        # These networks have built-in encoders, use identity
        encoder_type = "identity"
    elif task_config.obs_type == "image":
        # Force image encoder for image observations (unless overridden above)
        encoder_type = "image"
    elif task_config.obs_type in ["state", "keypoint"]:
        # For state/keypoint, use configured encoder; fall back to mlp if the
        # value is not a valid state encoder (e.g. accidentally set to network_type)
        _valid_state_encoders = {"mlp", "per_step_mlp", "identity"}
        encoder_type = getattr(network_config, "encoder_type", "mlp")
        if encoder_type not in _valid_state_encoders:
            loguru.logger.warning(
                f"[Encoder] encoder_type='{encoder_type}' is not a valid state encoder "
                f"(valid: {_valid_state_encoders}). Falling back to 'mlp'."
            )
            encoder_type = "mlp"
    else:
        raise ValueError(f"Invalid observation type: {task_config.obs_type}")
    loguru.logger.info(f"Using encoder type: {encoder_type}")

    if encoder_type == "identity":
        return IdentityEncoder(dropout=network_config.encoder_dropout)
    elif encoder_type == "mlp":
        return MLPEncoder(
            obs_dim=task_config.obs_dim,
            To=task_config.obs_steps,
            emb_dim=network_config.emb_dim,
            hidden_dims=[network_config.emb_dim] * network_config.num_encoder_layers,
            dropout=network_config.encoder_dropout,
        )
    elif encoder_type == "per_step_mlp":
        return PerStepMLPEncoder(
            obs_dim=task_config.obs_dim,
            emb_dim=network_config.emb_dim,
            hidden_dims=[network_config.emb_dim] * network_config.num_encoder_layers,
            dropout=network_config.encoder_dropout,
        )
    elif encoder_type == "image":
        kwargs = {
            "shape_meta": task_config.shape_meta,
            "rgb_model_name": network_config.rgb_model_name,
            "emb_dim": network_config.emb_dim,
            "use_seq": network_config.use_seq,
            "keep_horizon_dims": network_config.keep_horizon_dims,
            "resize_shape": task_config.resize_shape,
            "crop_shape": task_config.crop_shape,
            "random_crop": task_config.random_crop,
            "use_group_norm": task_config.use_group_norm,
        }
        return MultiImageObsEncoder(**kwargs)
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")


class GroupNorm1d(nn.Module):
    def __init__(self, dim, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, dim // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x.unsqueeze(2),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x.squeeze(2)
