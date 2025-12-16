class DeterministicCapBag:
    def __init__(self, max_instances=16):
        self.max_instances = max_instances

    def __call__(self, x):
        return x[:min(x.size(0), self.max_instances)]
