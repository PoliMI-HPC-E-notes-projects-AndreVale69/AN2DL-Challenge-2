import numpy as np
import torch

class MixupCutmixWrapper:
    def __init__(self, alpha=0.4, mixup_prob=0.4, cutmix_prob=0.2):
        self.alpha = alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def _sample_lambda(self):
        lam = np.random.beta(self.alpha, self.alpha)
        return max(lam, 1.0 - lam)  # keep lam >= 0.5

    def _rand_index(self, batch_size, device):
        return torch.randperm(batch_size, device=device)

    def _rand_bbox(self, W, H, lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        return x1, y1, x2, y2

    def __call__(self, x, y):
        """
        x: [B, C, H, W]
        y: [B] long labels

        returns:
          x_aug, y1, y2, lam, mode
        """
        B, C, H, W = x.shape
        device = x.device

        # 1) maybe Mixup
        if np.random.rand() < self.mixup_prob:
            lam = self._sample_lambda()
            idx = self._rand_index(B, device)
            x_mix = lam * x + (1. - lam) * x[idx]
            y1, y2 = y, y[idx]
            return x_mix, y1, y2, lam, "mixup"

        # 2) maybe CutMix
        if np.random.rand() < self.cutmix_prob:
            lam = self._sample_lambda()
            idx = self._rand_index(B, device)
            y1, y2 = y, y[idx]

            x_cut = x.clone()
            x1, y1b, x2, y2b = self._rand_bbox(W, H, lam)
            x_cut[:, :, y1b:y2b, x1:x2] = x[idx, :, y1b:y2b, x1:x2]

            # adjust lam by actual area
            lam = 1. - ((x2 - x1) * (y2b - y1b) / (W * H))
            return x_cut, y1, y2, lam, "cutmix"

        # 3) no augmentation
        return x, y, None, None, "none"
