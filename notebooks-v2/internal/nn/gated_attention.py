import torch
import torch.nn as nn

class GatedAttention(nn.Module):
    def __init__(self, dim, attn_dim=128):
        super().__init__()
        self.V = nn.Linear(dim, attn_dim)
        self.U = nn.Linear(dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h):
        # h: (n_i, D)
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        a = self.w(a)                    # (n_i, 1)
        a = torch.softmax(a, dim=0)      # attention over instances
        return a
