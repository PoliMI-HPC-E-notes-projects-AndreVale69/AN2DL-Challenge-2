"""
"""
import random
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
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
    idx2label: dict[int, str]
    label2idx: dict[str, int]
    num_K_folds: int
    train_transforms: transforms.Compose | None = None
    val_test_transforms: transforms.Compose | None = None
    image_size: int | None = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
"""
Standard ImageNet mean for normalization.
"""

IMAGENET_STD  = [0.229, 0.224, 0.225]
"""
Standard ImageNet standard deviation for normalization.
"""

class HistologyDataset(Dataset):
    def __init__(
            self,
            df,
            image_size: int,
            is_train: bool = True,
            use_mask_crop: bool = True,
            patch_mode: bool = False,
            patches_per_image: int = 1,
            patch_size: Optional[int] = None,
    ):
        """
        df: DataFrame with columns:
            - image_path
            - mask_path
            - label_idx     (train/val)
            - sample_index  (test when is_train=False)

        image_size: final output size (H=W=image_size)
        is_train: True -> (x4, label), False -> (x4, sample_index)
        use_mask_crop: if True, crop around mask before other ops

        patch_mode (train only):
            - if True, we sample smaller patches around the lesion
            - dataset length becomes len(df) * patches_per_image
        patch_size:
            - size (in pixels) of the patch BEFORE resizing to image_size
            - if None, we will auto-select based on current crop size

        4-channel input:
        - 3 channels RGB image normalized with ImageNet stats
        - 1 channel mask normalized to [0,1]
        """
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.is_train = is_train
        self.use_mask_crop = use_mask_crop

        self.patch_mode = patch_mode if is_train else False
        self.patches_per_image = patches_per_image if self.patch_mode else 1
        self.patch_size = patch_size

        # check if labels are present
        self.has_labels = "label_idx" in self.df.columns

        # if is_train, add color jitter to transforms
        self.color_jitter: Optional[transforms.ColorJitter] = (
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
            )
            if is_train
            else None
        )

        # if is_train, add random erasing to transforms
        # self.random_erasing: Optional[transforms.RandomErasing] = transforms.RandomErasing(
        #     p=0.25,
        #     scale=(0.02, 0.15),
        #     ratio=(0.3, 3.3)
        # ) if is_train else None
        self.random_erasing = None

    def __len__(self):
        return len(self.df)

    # ---------- 1) Crop around mask (square) ----------
    def _crop_using_mask(self, img_pil: Image.Image, mask_np: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        """
        Returns a square crop around the lesion + corresponding cropped mask.
        If mask is empty or extremely small, fall back to the original image.
        """
        H, W = mask_np.shape

        # 1) Where is the mask > 0?
        mask_pos = np.argwhere(mask_np > 0)

        if mask_pos.size == 0:
            # Completely empty mask -> trust the original framing
            return img_pil, mask_np

        # 2) Compute mask area ratio
        mask_area = mask_pos.shape[0]
        area_ratio = mask_area / float(H * W)

        # If the mask is too tiny (e.g. < 1% of the image), it's probably not reliable
        # -> keep the original image and mask instead of zooming like crazy.
        if area_ratio < 0.01:
            return img_pil, mask_np

        # 3) Basic bounding box
        y_min, x_min = mask_pos.min(axis=0)
        y_max, x_max = mask_pos.max(axis=0)

        # Ensure valid bounds
        y_min = max(0, y_min)
        x_min = max(0, x_min)
        y_max = min(H - 1, y_max)
        x_max = min(W - 1, x_max)

        # 4) Expand bbox a bit to include context
        if self.is_train:
            # Random expansion between 10% and 30%
            expand_factor = np.random.uniform(1.1, 1.3)
        else:
            # Deterministic, mild expansion for validation/test
            expand_factor = 1.2

        bbox_h = y_max - y_min + 1
        bbox_w = x_max - x_min + 1
        cy = (y_min + y_max) / 2.0
        cx = (x_min + x_max) / 2.0

        half_h = (bbox_h * expand_factor) / 2.0
        half_w = (bbox_w * expand_factor) / 2.0

        # Make square by taking the max half-size
        half_side = max(half_h, half_w)

        y1 = int(round(cy - half_side))
        y2 = int(round(cy + half_side))
        x1 = int(round(cx - half_side))
        x2 = int(round(cx + half_side))

        # Clip to image borders
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(H, y2)
        x2 = min(W, x2)

        # Final safety: if something went wrong, just return original
        if (y2 <= y1) or (x2 <= x1):
            return img_pil, mask_np

        cropped_img = img_pil.crop((x1, y1, x2, y2))
        cropped_mask = mask_np[y1:y2, x1:x2]

        return cropped_img, cropped_mask

    # ---------- 1b) Patch sampling around mask (for training only) ----------
    def _sample_mask_patch(
            self,
            img_pil: Image.Image,
            mask_np: np.ndarray,
            patch_size: Optional[int] = None,
    ) -> tuple[Image.Image, np.ndarray]:
        """
        From the (possibly mask-cropped) image + mask, take a smaller
        square patch around the lesion.

        - If mask has no pixels, fallback to a central patch.
        - If patch_size is None or too large, we adapt it to the image size.
        """
        H, W = mask_np.shape

        # pick patch size
        if patch_size is None:
            # e.g. 60% of the smaller side
            base = min(H, W)
            patch_size = int(0.6 * base)
        patch_size = max(32, patch_size)  # min safety

        # if patch size is bigger than the image, just return original
        if patch_size >= min(H, W):
            return img_pil, mask_np

        half = patch_size // 2

        mask_pos = np.argwhere(mask_np > 0)

        if mask_pos.size == 0:
            # no mask: take central patch
            cy, cx = H // 2, W // 2
        else:
            # choose a random positive pixel
            idx = np.random.randint(0, mask_pos.shape[0])
            cy, cx = mask_pos[idx]

            # add small jitter for variety
            jitter_y = np.random.randint(-patch_size // 4, patch_size // 4 + 1)
            jitter_x = np.random.randint(-patch_size // 4, patch_size // 4 + 1)
            cy = int(np.clip(cy + jitter_y, 0, H - 1))
            cx = int(np.clip(cx + jitter_x, 0, W - 1))

        y1 = cy - half
        y2 = cy + half
        x1 = cx - half
        x2 = cx + half

        # Clip to bounds
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y2 > H:
            y1 -= (y2 - H)
            y2 = H
        if x2 > W:
            x1 -= (x2 - W)
            x2 = W

        # safety
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(H, y2)
        x2 = min(W, x2)
        if (y2 <= y1) or (x2 <= x1):
            return img_pil, mask_np

        patch_img: Image.Image = img_pil.crop((x1, y1, x2, y2))
        patch_mask: np.ndarray = mask_np[y1:y2, x1:x2]

        return patch_img, patch_mask

    # ---------- 2) Joint geometric transforms img+mask ----------
    def _apply_joint_transforms(self, img_pil: Image.Image, mask_np: np.ndarray) -> tuple:
        """
        Joint geometric transforms for img + mask.
        We assume we've already applied mask-based cropping.
        """
        mask_pil = Image.fromarray(mask_np.astype(np.uint8))

        if self.is_train:
            # 1) Just resize to target size first
            img_pil = F.resize(
                img_pil,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR
            )
            mask_pil = F.resize(
                mask_pil,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.NEAREST
            )

            # 2) Horizontal / vertical flips
            if random.random() < 0.5:
                img_pil = F.hflip(img_pil)
                mask_pil = F.hflip(mask_pil)

            if random.random() < 0.5:
                img_pil = F.vflip(img_pil)
                mask_pil = F.vflip(mask_pil)

            # 3) Small rotation
            angle = random.uniform(-15.0, 15.0)
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
                img_pil,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR
            )
            mask_pil = F.resize(
                mask_pil,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.NEAREST
            )

        return img_pil, mask_pil

    def _apply_marker_occlusion(self, img_pil: Image.Image) -> Image.Image:
        """
        Simulate marker-pen artifacts by drawing random irregular blobs.
        """
        if random.random() < 0.05:  # 5% chance
            img_np = np.array(img_pil).copy()
            H, W, _ = img_np.shape

            # Random center
            cx = random.randint(0, W - 1)
            cy = random.randint(0, H - 1)

            # Random blob size
            radius = random.randint(int(0.1 * min(W, H)), int(0.25 * min(W, H)))

            # Blob color (green-ish like marker pen)
            color = np.array([0, random.randint(180, 255), random.randint(0, 80)])

            # Draw circular / irregular blob
            Y, X = np.ogrid[:H, :W]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2

            # Blend: semi-transparent
            alpha = 0.6
            img_np[mask] = (alpha * img_np[mask] + (1 - alpha) * color).astype(np.uint8)

            img_pil = Image.fromarray(img_np)

        return img_pil

    # ---------- 3) Artifact-style appearance augmentations ----------
    def _apply_artifact_augs(self, img_pil: Image.Image) -> Image.Image:
        """
        Simulate blur, illumination changes, and small noise
        to make the model robust to slide / scanner artefacts.
        Train only.
        """
        # 1) Gaussian blur (out-of-focus / smear) – LOW probability
        if random.random() < 0.15:  # 15%
            radius = random.uniform(0.5, 1.2)
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

        # 2) Gamma / brightness variation (old / uneven slides)
        if random.random() < 0.15:  # 15%
            gamma = random.uniform(0.8, 1.2)
            img_pil = F.adjust_gamma(img_pil, gamma=gamma)

        # 3) Small Gaussian noise (dust / scanner noise)
        if random.random() < 0.15:  # 15%
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            noise = np.random.normal(0.0, 0.02, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0.0, 1.0)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

        # 4) Marker occlusion – keep probability small, artefact is strong
        return self._apply_marker_occlusion(img_pil)

    # ---------- 4) __getitem__ ----------
    def __getitem__(self, idx):
        # in patch_mode, map idx to base image index
        if self.is_train and self.patch_mode:
            base_idx = idx // self.patches_per_image
        else:
            base_idx = idx

        row = self.df.iloc[base_idx]

        # 1) load image + mask
        img = Image.open(row["image_path"]).convert("RGB")
        mask_pil = Image.open(row["mask_path"]).convert("L")
        mask_np = np.array(mask_pil)

        # 2) optional: big mask-based crop (coarse ROI)
        if self.use_mask_crop:
            img, mask_np = self._crop_using_mask(img, mask_np)

        # 3) optional: smaller patch around mask (fine-grained patch)
        if self.is_train and self.patch_mode:
            img, mask_np = self._sample_mask_patch(
                img,
                mask_np,
                patch_size=self.patch_size,
            )

        # 4) joint spatial transforms (img + mask)
        img, mask_pil = self._apply_joint_transforms(img, mask_np)

        # 5) appearance / artifact augs (img only)
        if self.is_train:
            img = self._apply_artifact_augs(img)
            if self.color_jitter is not None:
                img = self.color_jitter(img)

        # 6) to tensor
        img_t = F.to_tensor(img)  # 3xHxW, [0,1]
        mask_np_final = np.array(mask_pil)  # HxW, 0-255
        mask_t = torch.from_numpy(mask_np_final).float() / 255.0  # HxW in [0,1]
        mask_t = mask_t.unsqueeze(0)  # 1xHxW

        # Zero out background in the image using the mask
        mask_bin = (mask_t > 0.5).float()  # 1xHxW binary
        # Broadcast to 3 channels for normalization
        rgb_mask = mask_bin.expand_as(img_t)  # 3xHxW
        # Zero out background
        img_t = img_t * rgb_mask

        # 7) normalize image
        img_t = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img_t)

        # 8) random erasing
        if self.is_train and self.random_erasing is not None:
            img_t = self.random_erasing(img_t)

        # 9) concatenate img + mask -> 4 channels
        x4 = torch.cat([img_t, mask_t], dim=0)  # 4xHxW

        # 10) return label or sample_index
        if self.has_labels:
            label = torch.tensor(row["label_idx"], dtype=torch.long)
            return x4, label
        else:
            return x4, row["sample_index"]