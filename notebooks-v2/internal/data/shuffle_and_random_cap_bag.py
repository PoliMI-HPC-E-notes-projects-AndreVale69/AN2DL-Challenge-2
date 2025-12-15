import torch
import random

class ShuffleAndRandomCapBag:
    def __init__(self, min_instances=8, max_instances=16):
        self.min_instances = min_instances
        self.max_instances = max_instances

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        idx = torch.randperm(n)

        k_hi = min(self.max_instances, n)
        k_lo = min(self.min_instances, k_hi)
        k = random.randint(k_lo, k_hi)  # random k in [k_lo, k_hi]

        return x[idx[:k]]
