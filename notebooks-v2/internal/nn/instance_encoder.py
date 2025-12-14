import timm
import torch.nn as nn

class InstanceEncoder(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b0.ns_jft_in1k", in_chans=4, out_dim=256, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,      # no classifier head
            global_pool="",     # we’ll pool ourselves (more control)
            in_chans=in_chans,
        )

        feat_dim = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        # x: (N, C, H, W)
        f = self.backbone.forward_features(x)  # (N, feat_dim, h, w)
        f = self.pool(f).flatten(1)            # (N, feat_dim)
        return self.proj(f)                    # (N, out_dim)
