from typing import Any

import torch
import torch.nn as nn
import timm

class DualPathNet(nn.Module):
    """
    Parallel paths (Advice 10):
    - RGB path: standard CNN backbone (e.g. EfficientNet)
    - Mask path: small CNN on the 1-channel mask
    - Fusion: concatenate features and classify
    """
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        mask_feat_dim: int = 128,
        drop_rate: float = 0.4,
        drop_path_rate: float = 0.15,
    ):
        super().__init__()

        # ---- 1. RGB backbone ----
        # num_classes=0 -> timm returns a feature vector (no classifier)
        self.rgb_backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,             # handle mask separately
            num_classes=0,          # no final classifier
            global_pool="avg",      # output: (B, D)
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Infer RGB feature dimension by a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            rgb_feat = self.rgb_backbone(dummy)
            rgb_feat_dim = rgb_feat.shape[1]

        # ---- 2. Mask branch ----
        # Very small CNN that extracts shape/geometry features from the mask
        self.mask_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # H/2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # H/4
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # H/8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # (B, 64, 1, 1)
        )
        self.mask_fc = nn.Linear(64, mask_feat_dim)

        # ---- 3. Fusion + classifier ----
        fused_dim = rgb_feat_dim + mask_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        """
        x4: (B, 4, H, W)
        - first 3 channels: RGB
        - last channel: mask
        """
        # Split RGB and mask
        rgb = x4[:, :3, :, :]          # (B, 3, H, W)
        mask = x4[:, 3:4, :, :]        # (B, 1, H, W)

        # 1) RGB path
        rgb_feat = self.rgb_backbone(rgb)  # (B, D)

        # 2) Mask path
        mask_feat_map = self.mask_branch(mask)         # (B, 64, 1, 1)
        mask_feat = mask_feat_map.view(mask_feat_map.size(0), -1)  # (B, 64)
        mask_feat = self.mask_fc(mask_feat)            # (B, mask_feat_dim)

        # 3) Fusion
        fused = torch.cat([rgb_feat, mask_feat], dim=1)  # (B, D + mask_feat_dim)
        logits = self.classifier(fused)                  # (B, num_classes)

        return logits
