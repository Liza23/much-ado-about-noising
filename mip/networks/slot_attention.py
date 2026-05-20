"""Slot Attention module for object-centric intent encoding.

Reference: Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020.

SlotObjectEncoder uses a ResNet18 backbone + SlotAttention + a soft learned slot
selector to produce a single object-centric vector per image frame. Averaging
across k future frames yields the slot-attention intent vector.

Optionally, a SpatialBroadcastDecoder reconstructs each input frame from the
K slot vectors; reconstruction loss drives spatial competition between slots.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SlotAttention(nn.Module):
    """Iterative slot attention: decomposes a feature grid into K slot vectors."""

    def __init__(self, num_slots: int, slot_dim: int, num_iters: int = 3, input_dim: int = 64, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 4, slot_dim),
        )

    def forward(self, inputs: torch.Tensor, return_attn: bool = False):
        """
        Args:
            inputs: (B, N, input_dim) — flattened spatial feature grid
            return_attn: if True, also return final-iteration attention map (B, K, N)

        Returns:
            slots: (B, num_slots, slot_dim)
            attn (optional): (B, K, N) spatial attention weights per slot
        """
        B = inputs.shape[0]
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        attn = None
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # (B, K, N): each slot attends over all spatial positions
            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            # softmax over slots (dim=1) so slots compete for each position
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)  # (B, K, slot_dim)

            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)

            slots = slots + self.mlp(self.norm_mlp(slots))

        if return_attn:
            return slots, attn  # (B, K, slot_dim), (B, K, N)
        return slots  # (B, K, slot_dim)


class SpatialBroadcastDecoder(nn.Module):
    """Reconstructs an image from a set of slot vectors.

    Each slot is tiled to (H, W), concatenated with 2D coordinate channels,
    then decoded by a shared CNN into RGB + alpha logit. Masks are obtained
    by softmax over slots (they compete for each pixel), and the final image
    is the mask-weighted sum of per-slot RGB outputs.

    Loss: MSE(recon, target) where target is the normalized input frame in [0,1].
    """

    def __init__(self, slot_dim: int):
        super().__init__()
        # +2 for (x, y) positional coordinates appended after broadcast
        self.decoder = nn.Sequential(
            nn.Conv2d(slot_dim + 2, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 3, padding=1),  # 3 RGB + 1 alpha logit
        )

    def forward(self, slots: torch.Tensor, H: int, W: int):
        """
        Args:
            slots: (B, K, slot_dim)
            H, W:  spatial resolution of target frames
        Returns:
            recon: (B, 3, H, W)  reconstructed image in [0, 1]
            masks: (B, K, 1, H, W) per-slot alpha masks (sum to 1 over K)
        """
        B, K, D = slots.shape

        xs = torch.linspace(-1, 1, W, device=slots.device)
        ys = torch.linspace(-1, 1, H, device=slots.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([xx, yy], dim=0)  # (2, H, W)
        pos = pos.unsqueeze(0).expand(B * K, -1, -1, -1)  # (B*K, 2, H, W)

        s = slots.reshape(B * K, D, 1, 1).expand(-1, -1, H, W)  # (B*K, D, H, W)
        x = torch.cat([s, pos], dim=1)                           # (B*K, D+2, H, W)

        out = self.decoder(x).reshape(B, K, 4, H, W)             # (B, K, 4, H, W)
        rgb   = torch.sigmoid(out[:, :, :3])                     # (B, K, 3, H, W)
        masks = out[:, :, 3:].softmax(dim=1)                     # (B, K, 1, H, W)

        recon = (masks * rgb).sum(dim=1)                         # (B, 3, H, W)
        return recon, masks


class SlotObjectEncoder(nn.Module):
    """Encodes a stack of k image frames into an object-centric intent vector.

    Pipeline per frame:
        ResNet18 (no global pool) → spatial features → SlotAttention(K slots)
        → soft selector (learned attention over K slots) → weighted object vec
    Then: mean over k frames → intent vector.

    An auxiliary regression head predicts the object's low-dim state from the
    selected object vector; this is supervised during training to ground which
    slot corresponds to the manipulated object.
    """

    def __init__(
        self,
        num_slots: int = 4,
        slot_dim: int = 64,
        num_iters: int = 3,
        obj_state_dim: int = 10,
        slot_input_dim: int = 64,
        use_recon_decoder: bool = False,
        use_soft_selector: bool = False,
        use_layer2: bool = False,
    ):
        super().__init__()
        resnet = models.resnet18(weights=None)
        if use_layer2:
            # layer2 output: (B, 128, ~11, ~11) for 84×84 input — higher spatial resolution
            # for better object-level spatial competition between slots.
            self.cnn = nn.Sequential(*list(resnet.children())[:-4])
            self.proj = nn.Linear(128, slot_input_dim)
        else:
            # layer3 output: (B, 256, ~6, ~6) for 84×84 input.
            self.cnn = nn.Sequential(*list(resnet.children())[:-3])
            self.proj = nn.Linear(256, slot_input_dim)
        self.slot_attention = SlotAttention(num_slots, slot_dim, num_iters, input_dim=slot_input_dim)
        # Soft selector: learns which slot is task-relevant; replaces mean-pool over K
        self.selector = nn.Linear(slot_dim, 1) if use_soft_selector else None
        # Auxiliary head: predicts object low-dim state from selected/mean slot vec
        self.obj_regressor = nn.Linear(slot_dim, obj_state_dim)
        # Optional spatial broadcast decoder for reconstruction loss
        self.recon_decoder = SpatialBroadcastDecoder(slot_dim) if use_recon_decoder else None

    def forward(self, frames: torch.Tensor, return_attn: bool = False, return_recon: bool = False):
        """
        Args:
            frames:       (B, k, C, H, W) — normalized future image frames in [0, 1]
            return_attn:  if True, also return (attn, spatial_shape)
            return_recon: if True and recon_decoder is present, also return (recon, frames_target)

        Returns (always):
            intent:   (B, slot_dim)
            obj_pred: (B, k, obj_state_dim)
        Optional appended outputs (in order, if requested):
            attn:           (B*k, K, N) spatial attention weights
            spatial_shape:  (Hp, Wp)
            recon:          (B*k, 3, H, W) reconstructed frames
            frames_target:  (B*k, C, H, W) input frames (reconstruction target)
        """
        B, k, C, H, W = frames.shape
        x = frames.reshape(B * k, C, H, W)

        feat = self.cnn(x)                        # (B*k, 256, H', W')
        Hp, Wp = feat.shape[2], feat.shape[3]
        feat = feat.flatten(2).permute(0, 2, 1)   # (B*k, N, 256)
        feat = self.proj(feat)                     # (B*k, N, slot_input_dim)

        if return_attn:
            slots, attn = self.slot_attention(feat, return_attn=True)  # (B*k, K, slot_dim), (B*k, K, N)
        else:
            slots = self.slot_attention(feat)      # (B*k, K, slot_dim)
            attn = None

        # Reconstruction from per-frame slots before mean pooling
        recon = None
        if return_recon and self.recon_decoder is not None:
            recon, _ = self.recon_decoder(slots, H, W)  # (B*k, 3, H, W)

        if self.selector is not None:
            scores = self.selector(slots)           # (B*k, K, 1)
            weights = scores.softmax(dim=1)         # (B*k, K, 1)
            object_vec = (weights * slots).sum(dim=1)  # (B*k, slot_dim)
        else:
            object_vec = slots.mean(dim=1)          # (B*k, slot_dim)
        object_vec = object_vec.reshape(B, k, self.slot_attention.slot_dim)  # (B, k, slot_dim)
        obj_pred = self.obj_regressor(object_vec)  # (B, k, obj_state_dim)
        intent = object_vec.mean(dim=1)            # (B, slot_dim) — mean over k frames

        out = [intent, obj_pred]
        if return_attn:
            out += [attn, (Hp, Wp)]
        if return_recon and recon is not None:
            out += [recon, x]  # x = (B*k, C, H, W) reconstruction target
        return tuple(out) if len(out) > 2 else (intent, obj_pred)
