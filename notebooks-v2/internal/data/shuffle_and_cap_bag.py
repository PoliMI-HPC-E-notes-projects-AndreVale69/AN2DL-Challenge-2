import torch

class ShuffleAndCapBag:
    def __init__(self, max_instances: int = 16):
        self.max_instances = max_instances

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_i, C, H, W)
        n = x.size(0)
        idx = torch.randperm(n)
        x = x[idx]
        if self.max_instances is not None and n > self.max_instances:
            x = x[:self.max_instances]
        return x
