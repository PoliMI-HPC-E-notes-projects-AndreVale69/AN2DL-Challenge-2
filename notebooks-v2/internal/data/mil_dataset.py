import numpy as np
import torch
from torch.utils.data import Dataset

class MILDatasetMemmapRanges(Dataset):
    """
    One item = one slide (bag).
    Instances are stored contiguously in a big memmap:
      X: (total_patches, H, W, 3)
      M: (total_patches, H, W, 1) optional
    slide_to_patchidx: (n_slides, 2) with [start, end)
    y: (n_slides,)
    """
    def __init__(
        self,
        x_path: str,
        idx_path: str,
        *,
        y_path: str | None = None,
        m_path: str | None = None,
        patch_size: int = 384,
        mmap_mode: str = "r",
        bag_transform=None,     # e.g. ShuffleAndCapBag
        transform=None,         # instance transform (applied per patch tensor)
        return_meta: bool = False,
    ):
        self.patch_size = patch_size
        self.transform = transform
        self.bag_transform = bag_transform
        self.return_meta = return_meta

        if y_path is not None:
            self.y = np.load(y_path).astype(np.int64)
        else:
            self.y = None
        self.slide_to_patchidx = np.load(idx_path).astype(np.int64)

        total_patches = int(self.slide_to_patchidx[-1, 1])  # last end
        self.X = np.memmap(
            x_path, dtype=np.uint8, mode=mmap_mode,
            shape=(total_patches, patch_size, patch_size, 3)
        )

        self.M = None
        if m_path is not None:
            self.M = np.memmap(
                m_path, dtype=np.uint8, mode=mmap_mode,
                shape=(total_patches, patch_size, patch_size, 1)
            )

    def __len__(self):
        return len(self.slide_to_patchidx)

    def __getitem__(self, s: int):
        start, end = self.slide_to_patchidx[s]
        start, end = int(start), int(end)

        x_np = np.asarray(self.X[start:end])  # (n,H,W,3) non-writable
        x = torch.from_numpy(x_np).permute(0, 3, 1, 2)  # uint8 tensor (non-writable is fine)
        x = x.float().div_(255.0)  # creates new float tensor anyway

        if self.M is not None:
            m_np = np.array(self.M[start:end], copy=True)  # (n,H,W,1) writable
            m = torch.from_numpy(m_np).permute(0, 3, 1, 2).float() / 255.0  # (n,1,H,W)
            x = torch.cat([x, m], dim=1)  # (n,4,H,W)

        if self.transform is not None:
            x = torch.stack([self.transform(xi) for xi in x], dim=0)

        if self.bag_transform is not None:
            x = self.bag_transform(x)

        if self.y is not None:
            y = torch.tensor(int(self.y[s]), dtype=torch.long)
            if self.return_meta:
                meta = {"slide_index": s, "start": start, "end": end, "bag_size": x.size(0)}
                return x, y, meta
            return x, y

        meta = {"slide_index": s, "start": start, "end": end, "bag_size": x.size(0)}
        return x, meta
