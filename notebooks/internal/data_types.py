"""
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class DictLike:
    """
    A base class that mimics dictionary behavior for dataclasses.
    """
    def to_dict(self) -> dict:
        return asdict(self)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def values(self):
        return self.to_dict().values()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.to_dict())

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def __contains__(self, key):
        return key in self.to_dict()

    def __or__(self, other):
        if isinstance(other, dict):
            return {**self.to_dict(), **other}
        return NotImplemented

@dataclass
class LabelMap(DictLike):
    """
    Stores label mappings for breast cancer subtypes.

    Attributes:
        triple_negative (str): Label for triple negative subtype.
        luminal_a (str): Label for luminal A subtype.
        luminal_b (str): Label for luminal B subtype.
        her2_enriched (str): Label for HER2-enriched subtype.
    """
    triple_negative: str
    luminal_a: str
    luminal_b: str
    her2_enriched: str

@dataclass
class DataSet(DictLike):
    """
    Stores datasets and scalers for pain intensity classification.

    Attributes:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
    """
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_loader: DataLoader
    val_loader: DataLoader
    val_test_transforms: transforms.Compose
    idx2label: dict[int, str]
    label2idx: dict[str, int]

class HistologyDataset(Dataset):
    def __init__(self, df, transforms=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _crop_using_mask(self, img: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        img  : PIL RGB image
        mask : numpy array (H, W), >0 where lesion is.
        returns a cropped PIL image
        """
        # find bounding box of mask (non-zero area); if no mask, return original image
        mask_pos = np.argwhere(mask > 0)
        if mask_pos.size == 0:
            return img  # fallback

        y_min, x_min = mask_pos.min(axis=0)
        y_max, x_max = mask_pos.max(axis=0)

        H, W = mask.shape

        # padding
        pad = 0.02
        h = y_max - y_min
        w = x_max - x_min

        y_min = max(0, int(y_min - pad * h))
        y_max = min(H, int(y_max + pad * h))
        x_min = max(0, int(x_min - pad * w))
        x_max = min(W, int(x_max + pad * w))

        # ---- NEW PART: MAKE THE CROP SQUARE ----
        crop_h = y_max - y_min
        crop_w = x_max - x_min
        side = max(crop_h, crop_w)

        # center the bbox inside the square
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2

        half = side // 2

        y1 = max(0, cy - half)
        y2 = min(H, cy + half)
        x1 = max(0, cx - half)
        x2 = min(W, cx + half)

        # adjust if at border
        if (y2 - y1) < side:
            if y1 == 0: y2 = min(H, side)
            else: y1 = max(0, y2 - side)
        if (x2 - x1) < side:
            if x1 == 0: x2 = min(W, side)
            else: x1 = max(0, x2 - side)

        cropped = img.crop((x1, y1, x2, y2))
        return cropped

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        mask_np = np.array(mask)

        img = self._crop_using_mask(img, mask_np)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.is_train:
            label = row["label_idx"]
            label = torch.tensor(label, dtype=torch.long)
            return img, label
        else:
            # for test set we return sample_index for submission
            return img, row["sample_index"]
