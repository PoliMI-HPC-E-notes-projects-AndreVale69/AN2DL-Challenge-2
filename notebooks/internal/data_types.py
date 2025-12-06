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
        image_size (int): Size of the images after transformations.
    """
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_transforms: transforms.Compose
    val_test_transforms: transforms.Compose
    idx2label: dict[int, str]
    label2idx: dict[str, int]
    num_K_folds: int
    image_size: int

class HistologyDataset(Dataset):
    def __init__(self, df, image_size, transforms=None, is_train=True):
        """
        df: DataFrame with columns:
            - image_path
            - mask_path
            - label_idx (for train/val)
            - sample_index (for test)
        image_size: final image size (after transforms)
        transforms: torchvision transforms to apply to final PIL image
        is_train: if True -> return (image, label), else -> (image, sample_index)
        """
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    # ---------- 1) Crop around mask (square) ----------
    def _crop_using_mask(self, img: Image.Image, mask: np.ndarray):
        """
        img  : PIL RGB image
        mask : numpy array (H, W), >0 where lesion is.
        returns (cropped_image, cropped_mask_np)
        """
        mask_pos = np.argwhere(mask > 0)
        if mask_pos.size == 0:
            # no lesion: return original img + mask
            return img, mask

        y_min, x_min = mask_pos.min(axis=0)
        y_max, x_max = mask_pos.max(axis=0)

        H, W = mask.shape

        # small padding around the mask
        pad = 0.02
        h = y_max - y_min
        w = x_max - x_min

        y_min = max(0, int(y_min - pad * h))
        y_max = min(H, int(y_max + pad * h))
        x_min = max(0, int(x_min - pad * w))
        x_max = min(W, int(x_max + pad * w))

        # make square crop
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

        # adjust if touching borders
        if (y2 - y1) < side:
            if y1 == 0:
                y2 = min(H, side)
            else:
                y1 = max(0, y2 - side)
        if (x2 - x1) < side:
            if x1 == 0:
                x2 = min(W, side)
            else:
                x1 = max(0, x2 - side)

        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_mask = mask[y1:y2, x1:x2]

        return cropped_img, cropped_mask

    # ---------- 2) Random mask-guided patch (TRAIN ONLY) ----------
    def _random_mask_patch(self, img: Image.Image, mask_np: np.ndarray, patch_size: int | None) -> Image.Image:
        """
        From a mask-cropped image, sample a random patch of size (patch_size, patch_size)
        that contains some lesion. Fallback: center patch.

        img      : PIL RGB image (mask-cropped)
        mask_np  : numpy array (H, W), >0 where lesion is.
        patch_size: size of the patch to sample (if None, use self.image_size)
        returns: cropped PIL image
        """
        patch_size = patch_size if patch_size is not None else self.image_size

        H, W = mask_np.shape

        # if the crop is already small, just use it as is (will be resized in transforms)
        if H <= patch_size or W <= patch_size:
            return img

        # try a few times to find a patch that contains lesion pixels
        for _ in range(10):
            y1 = np.random.randint(0, H - patch_size + 1)
            x1 = np.random.randint(0, W - patch_size + 1)
            y2 = y1 + patch_size
            x2 = x1 + patch_size

            patch_mask = mask_np[y1:y2, x1:x2]
            if (patch_mask > 0).sum() > 500:  # threshold: at least some lesion pixels
                return img.crop((x1, y1, x2, y2))

        # fallback: center patch (still aligned with tissue region)
        cy, cx = H // 2, W // 2
        y1 = max(0, cy - patch_size // 2)
        x1 = max(0, cx - patch_size // 2)
        y2 = min(H, y1 + patch_size)
        x2 = min(W, x1 + patch_size)
        return img.crop((x1, y1, x2, y2))

    # ---------- 3) __getitem__ ----------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        mask_np = np.array(mask)

        # 1) mask-based square crop
        img_c, mask_c = self._crop_using_mask(img, mask_np)

        # 2) for TRAIN: random lesion patch; for VAL/TEST: deterministic crop
        if self.is_train:
            img_final = self._random_mask_patch(img_c, mask_c, patch_size=self.image_size)
        else:
            img_final = img_c  # single deterministic view

        # 3) transforms (resize, normalize, etc.)
        if self.transforms is not None:
            img_t = self.transforms(img_final)
        else:
            img_t = transforms.ToTensor()(img_final)

        if self.is_train:
            label = torch.tensor(row["label_idx"], dtype=torch.long)
            return img_t, label
        else:
            # for test set (and any is_train=False dataset)
            return img_t, row["sample_index"]