import numpy as np
import torch
from torch.utils.data import Dataset

class MILDatasetMemmap(Dataset):
    """
    MIL dataset: one item = one slide (bag) of variable #patches.
    Instances stored in a memmapped numpy array (N, H, W, C) or (N, C, H, W).
    """
    def __init__(
        self,
        bags_df,
        memmap_path: str,
        *,
        x_key: str = "patch_ids",
        y_key: str = "label",
        id_key: str = "slide_id",
        mmap_shape: tuple | None = None,
        mmap_dtype=np.uint8,
        mmap_mode: str = "r",
        channels_first: bool = False,   # set True if stored as (N,C,H,W)
        transform=None,                 # instance-level transform (applied to each patch)
        bag_transform=None,             # bag-level transform (rare; e.g. shuffle, drop, etc.)
        return_meta: bool = False,
    ):
        self.df = bags_df.reset_index(drop=True)
        self.x_key = x_key
        self.y_key = y_key
        self.id_key = id_key
        self.transform = transform
        self.bag_transform = bag_transform
        self.return_meta = return_meta
        self.channels_first = channels_first

        # Load memmap
        if mmap_shape is None:
            self.X = np.load(memmap_path, mmap_mode=mmap_mode)
        else:
            self.X = np.memmap(memmap_path, dtype=mmap_dtype, mode=mmap_mode, shape=mmap_shape)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patch_ids = row[self.x_key]
        y = int(row[self.y_key])
        slide_id = row[self.id_key] if self.id_key in row else idx

        # gather instances
        arr = self.X[patch_ids]  # shape: (n_i, H,W,C) or (n_i,C,H,W)

        # convert to torch float in [0,1]
        x = torch.from_numpy(np.asarray(arr))
        if not self.channels_first:
            # (n,H,W,C) -> (n,C,H,W)
            x = x.permute(0, 3, 1, 2)
        x = x.float().div_(255.0)

        # instance transforms (apply per patch)
        if self.transform is not None:
            x = torch.stack([self.transform(xi) for xi in x], dim=0)

        # bag transforms (optional)
        if self.bag_transform is not None:
            x = self.bag_transform(x)

        y = torch.tensor(y, dtype=torch.long)

        if self.return_meta:
            meta = {"slide_id": slide_id, "patch_ids": patch_ids}
            return x, y, meta
        return x, y
