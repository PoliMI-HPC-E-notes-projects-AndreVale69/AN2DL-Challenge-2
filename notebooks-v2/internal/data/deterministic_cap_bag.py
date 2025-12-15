import torch

class DeterministicCapBag:
    def __init__(self, max_instances: int = 16):
        self.max_instances = max_instances

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # keep first max_instances (no randomness)
        if self.max_instances is None or x.size(0) <= self.max_instances:
            return x
        return x[:self.max_instances]