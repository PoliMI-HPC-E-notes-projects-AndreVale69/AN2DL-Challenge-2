import torch

class ShuffleAndCapBag:
    def __init__(self, max_instances: int = 16):
        self.max_instances = max_instances

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        if self.max_instances is None or n <= self.max_instances:
            idx = torch.randperm(n)
            return x[idx]
        idx = torch.randperm(n)[:self.max_instances]  # random subset
        return x[idx]
