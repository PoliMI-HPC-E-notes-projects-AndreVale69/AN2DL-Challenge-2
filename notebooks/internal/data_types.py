"""
"""
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
"""
Standard ImageNet mean for normalization.
"""

IMAGENET_STD  = [0.229, 0.224, 0.225]
"""
Standard ImageNet standard deviation for normalization.
"""

class HistologyDataset(Dataset):
    def __init__(self, df, image_size, is_train=True, use_mask_crop=True):
        """
        df: DataFrame with columns:
            - image_path
            - mask_path
            - label_idx     (train/val)
            - sample_index  (test when is_train=False)
        image_size: output image size (H=W=image_size)
        is_train: True -> (x4, label), False -> (x4, sample_index)
        use_mask_crop: if True, crop around mask before augmentations

        4-channel input:
        - 3 channels RGB image normalized with ImageNet stats
        - 1 channel mask normalized to [0,1]

        """
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.is_train = is_train
        self.use_mask_crop = use_mask_crop

        # check if labels are present
        self.has_labels = "label_idx" in self.df.columns

        # if is_train, add color jitter to transforms
        self.color_jitter: transforms.ColorJitter | None = transforms.ColorJitter(
            brightness=0.15, contrast=0.15,
            saturation=0.15, hue=0.03
        ) if is_train else None

        # if is_train, add random erasing to transforms
        self.random_erasing: transforms.RandomErasing | None = transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3)
        ) if is_train else None

    def __len__(self):
        return len(self.df)

    # ---------- 1) Crop around mask (square) ----------
    def _crop_using_mask(self, img_pil: Image.Image, mask_np: np.ndarray):
        """
        Returns a square crop around the lesion + corresponding cropped mask.
        If mask is empty, returns original image and mask.
        """
        mask_pos = np.argwhere(mask_np > 0)
        if mask_pos.size == 0:
            return img_pil, mask_np  # empty mask: return original

        y_min, x_min = mask_pos.min(axis=0)
        y_max, x_max = mask_pos.max(axis=0)

        H, W = mask_np.shape

        pad = 0.02  # 2% padding
        h = y_max - y_min
        w = x_max - x_min

        y_min = max(0, int(y_min - pad * h))
        y_max = min(H, int(y_max + pad * h))
        x_min = max(0, int(x_min - pad * w))
        x_max = min(W, int(x_max + pad * w))

        crop_h = y_max - y_min
        crop_w = x_max - x_min
        side = max(crop_h, crop_w)

        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        half = side // 2

        y1 = max(0, cy - half)
        x1 = max(0, cx - half)
        y2 = min(H, y1 + side)
        x2 = min(W, x1 + side)

        cropped_img = img_pil.crop((x1, y1, x2, y2))
        cropped_mask = mask_np[y1:y2, x1:x2]

        return cropped_img, cropped_mask

    # ---------- 2) Augment with joint transforms img+mask ----------
    def _apply_joint_transforms(self, img_pil: Image.Image, mask_np: np.ndarray) -> tuple:
        """
        Applies joint transformations to the image and mask.
        """
        mask_pil = Image.fromarray(mask_np.astype(np.uint8))  # mask 0/255

        if self.is_train:
            # RandomResizedCrop: 80-100% scale, 0.9-1.1 ratio
            scale = (0.6, 1.0)
            ratio = (0.8, 1.2)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img_pil, scale=scale, ratio=ratio
            )

            img_pil = F.resized_crop(
                img_pil, i, j, h, w,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR
            )
            mask_pil = F.resized_crop(
                mask_pil, i, j, h, w,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.NEAREST
            )

            # Horizontal Flip
            if random.random() < 0.5:
                img_pil = F.hflip(img_pil)
                mask_pil = F.hflip(mask_pil)

            # Vertical Flip
            if random.random() < 0.5:
                img_pil = F.vflip(img_pil)
                mask_pil = F.vflip(mask_pil)

            # Random Rotation from 0 to 360 degrees
            angle = random.uniform(0, 360)
            img_pil = F.rotate(
                img_pil, angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=(255, 255, 255)
            )
            mask_pil = F.rotate(
                mask_pil, angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0
            )
        else:
            # Validation / test: deterministic resize only
            img_pil = F.resize(
                img_pil, size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR
            )
            mask_pil = F.resize(
                mask_pil, size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.NEAREST
            )

        return img_pil, mask_pil

    # ---------- 3) __getitem__ ----------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) load image + mask
        img = Image.open(row["image_path"]).convert("RGB")
        mask_pil = Image.open(row["mask_path"]).convert("L")
        mask_np = np.array(mask_pil)

        # 2) crop using mask
        if self.use_mask_crop:
            img, mask_np = self._crop_using_mask(img, mask_np)

        # 3) joint transforms img + mask
        img, mask_pil = self._apply_joint_transforms(img, mask_np)

        # 4) color jitter apply only to img
        if self.is_train and self.color_jitter is not None:
            img = self.color_jitter(img)

        # 5) convert to tensor
        img_t = F.to_tensor(img)  # 3xHxW, [0,1]
        mask_np_final = np.array(mask_pil)  # HxW, 0-255
        mask_t = torch.from_numpy(mask_np_final).float() / 255.0  # HxW in [0,1]
        mask_t = mask_t.unsqueeze(0)  # 1xHxW

        # 6) normalize img
        img_t = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img_t)

        # 6b) random erasing on image only (train only)
        if self.is_train and self.random_erasing is not None:
            img_t = self.random_erasing(img_t)

        # 7) concatenate img + mask
        x4 = torch.cat([img_t, mask_t], dim=0)  # 4xHxW

        if self.has_labels:
            # train or val
            label = torch.tensor(row["label_idx"], dtype=torch.long)
            return x4, label
        else:
            # test (no labels, only sample_index)
            return x4, row["sample_index"]
