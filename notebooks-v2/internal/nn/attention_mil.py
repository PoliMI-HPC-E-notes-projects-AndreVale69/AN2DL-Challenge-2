import torch
import torch.nn as nn

from internal.nn.gated_attention import GatedAttention
from internal.nn.instance_encoder import InstanceEncoder


class AttentionMIL(nn.Module):
    def __init__(self, n_classes, encoder_dim=256, in_chans=4, enc_chunk=2):
        super().__init__()
        self.encoder = InstanceEncoder(in_chans=in_chans, out_dim=encoder_dim)
        self.attn = GatedAttention(encoder_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(encoder_dim, n_classes)
        )
        self.enc_chunk = enc_chunk

    def encode_chunked(self, xcat: torch.Tensor) -> torch.Tensor:
        # xcat: (N_total, 3, H, W)
        feats = []
        for xb in xcat.split(self.enc_chunk, dim=0):    # xb: (k,C,H,W)
            feats.append(self.encoder(xb))              # (k,D)
        return torch.cat(feats, dim=0)                  # (N_total,D)

    def forward(self, xcat, bag_sizes):
        # xcat: (N_total, 3, H, W)
        # bag_sizes: (B,)
        assert xcat.dim() == 4 and bag_sizes.dim() == 1, (
            "Invalid input dimensions, expected xcat (N_total, 3, H, W) and bag_sizes (B,)"
        )
        assert xcat.size(0) == int(bag_sizes.sum()),(
            "xcat rows must match sum(bag_sizes), got {} and {}".format(xcat.size(0), bag_sizes.sum().item())
        )

        # Encode all instances in chunks to save memory
        feats = self.encode_chunked(xcat)   # (N_total, D)

        bags = torch.split(feats, bag_sizes.tolist(), dim=0)

        logits = []
        for h in bags:
            a = self.attn(h)          # (n_i, 1)
            z = (a * h).sum(dim=0)    # (D,)
            logits.append(self.classifier(z))

        return torch.stack(logits, dim=0)  # (B, n_classes)
