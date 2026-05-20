"""Simple CNN intent encoder: ResNet18 → global avg pool → Linear → intent_dim.

Used with intent_type="cnn_image". No slot attention, no auxiliary loss.
Ablation: tests whether visual conditioning itself (not slot decomposition) drives gains.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNIntentEncoder(nn.Module):
    """Encodes k future image frames into an intent vector via global average pooling.

    Pipeline:
        (B, k, C, H, W)
        → per-frame ResNet18 (includes global avg pool, no fc): (B*k, 512)
        → Linear(512, intent_dim)
        → mean over k frames
        → (B, intent_dim)
    """

    def __init__(self, intent_dim: int = 64):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool, no fc
        self.proj = nn.Linear(512, intent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, k, C, H, W) — normalized future image frames in [0, 1]

        Returns:
            intent: (B, intent_dim)
        """
        B, k, C, H, W = frames.shape
        x = frames.reshape(B * k, C, H, W)
        feat = self.cnn(x).flatten(1)       # (B*k, 512)
        intent_per_frame = self.proj(feat)  # (B*k, intent_dim)
        intent = intent_per_frame.reshape(B, k, -1).mean(dim=1)  # (B, intent_dim)
        return intent
