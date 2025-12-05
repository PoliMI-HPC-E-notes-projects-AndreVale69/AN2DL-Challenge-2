"""
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
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
        val_test_transforms (transforms.Compose): Transformations for validation and testing.
        idx2label (dict[int, str]): Mapping from index to label.
        label2idx (dict[str, int]): Mapping from label to index.
        num_K_folds (int): Number of K-folds for cross-validation. In other words, how many splits the training data has been divided into.
    """
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_transforms: transforms.Compose
    val_test_transforms: transforms.Compose
    idx2label: dict[int, str]
    label2idx: dict[str, int]
    num_K_folds: int

class HistologyDataset(Dataset):
    def __init__(self, df, transforms=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _crop_using_mask(self, img: Image.Image, mask: np.ndarray):
        """
        img  : PIL RGB image
        mask : numpy array (H, W), >0 where lesion is.
        returns (cropped_image, cropped_mask_np)
        """
        mask_pos = np.argwhere(mask > 0)
        if mask_pos.size == 0:
            # return original img + original mask if no lesion
            return img, mask

        y_min, x_min = mask_pos.min(axis=0)
        y_max, x_max = mask_pos.max(axis=0)

        H, W = mask.shape

        # -------------------------
        # Small padding
        # -------------------------
        pad = 0.02
        h = y_max - y_min
        w = x_max - x_min

        y_min = max(0, int(y_min - pad * h))
        y_max = min(H, int(y_max + pad * h))
        x_min = max(0, int(x_min - pad * w))
        x_max = min(W, int(x_max + pad * w))

        # -------------------------
        # Square crop logic
        # -------------------------
        crop_h = y_max - y_min
        crop_w = x_max - x_min
        side = max(crop_h, crop_w)

        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        half = side // 2

        y1 = max(0, cy - half)
        y2 = min(H, y1 + side)
        x1 = max(0, cx - half)
        x2 = min(W, x1 + side)

        # -------------------------
        # Produce both crops
        # -------------------------
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_mask = mask[y1:y2, x1:x2]

        return cropped_img, cropped_mask

    def _random_mask_patch(self, img, mask_np, patch_size=512):
        H, W = mask_np.shape
        if H <= patch_size or W <= patch_size:
            return img  # will be resized by transforms

        # try a few times to find a patch with tissue
        for _ in range(10):
            y1 = np.random.randint(0, H - patch_size + 1)
            x1 = np.random.randint(0, W - patch_size + 1)
            y2 = y1 + patch_size
            x2 = x1 + patch_size

            patch_mask = mask_np[y1:y2, x1:x2]
            if (patch_mask > 0).sum() > 500:  # at least some lesion
                return img.crop((x1, y1, x2, y2))

        # fallback: centered
        cy, cx = H // 2, W // 2
        y1 = max(0, cy - patch_size // 2)
        x1 = max(0, cx - patch_size // 2)
        y2 = min(H, y1 + patch_size)
        x2 = min(W, x1 + patch_size)
        return img.crop((x1, y1, x2, y2))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        mask_np = np.array(mask)

        img_c, mask_c = self._crop_using_mask(img, mask_np)

        if self.is_train:
            img_p = self._random_mask_patch(img_c, mask_c, patch_size=512)
        else:
            img_p = img_c  # single deterministic crop for val/test

        if self.transforms is not None:
            img_t = self.transforms(img_p)
        else:
            img_t = transforms.ToTensor()(img_p)

        if self.is_train:
            label = torch.tensor(row["label_idx"], dtype=torch.long)
            return img_t, label
        else:
            return img_t, row["sample_index"]