import torch
import torchvision.transforms.functional as TF

from internal.nn.get_mask_based_crops_4ch import get_mask_based_crops_4ch


def apply_tta(img_tensor) -> list[torch.Tensor]:
    """
    img_tensor: [3, H, W] torch tensor (after ToTensor+Normalize)
    returns list of augmented tensors (each [3,H,W])
    """
    tta_list = []

    # identity
    tta_list.append(img_tensor)

    # horizontal flip
    tta_list.append(TF.hflip(img_tensor))

    # vertical flip
    tta_list.append(TF.vflip(img_tensor))

    # rotations
    tta_list.append(TF.rotate(img_tensor, 90))
    tta_list.append(TF.rotate(img_tensor, -90))
    tta_list.append(TF.rotate(img_tensor, 180))

    return tta_list

def apply_tta_4ch_safe(img4):
    """
    img4: [4,H,W]
    """
    return [
        img4,
        torch.flip(img4, [2])  # horizontal flip only
    ]

# def get_fixed_multi_crops(img: torch.Tensor,
#                           base_size: int = 640,
#                           inner_ratio: float = 0.8):
#     """
#     img: [C, H, W] tensor, already mask-centered and resized to base_size.
#     We take several overlapping crops of size (inner_ratio * base_size),
#     then resize them back to base_size so the model can ingest them.
#     """
#     C, H, W = img.shape
#     assert H == W == base_size, f"Expected square {base_size}x{base_size}, got {H}x{W}"
#
#     crop_size = int(base_size * inner_ratio)  # e.g. 512 if base_size=640
#     step = base_size - crop_size             # how much margin we have
#
#     # coordinates for crops: center + 4 corners
#     coords = []
#
#     # top-left
#     coords.append((0, 0))
#     # top-right
#     coords.append((0, step))
#     # bottom-left
#     coords.append((step, 0))
#     # bottom-right
#     coords.append((step, step))
#     # center
#     center_y = step // 2
#     center_x = step // 2
#     coords.append((center_y, center_x))
#
#     crops = []
#     for y, x in coords:
#         patch = img[:, y:y + crop_size, x:x + crop_size]  # [C, crop, crop]
#         patch = TF.resize(patch, [base_size, base_size])
#         crops.append(patch)
#
#     return crops  # list of [C, base_size, base_size]


# def apply_multicrop_tta(img_tensor: torch.Tensor,
#                         base_size: int = 640,
#                         inner_ratio: float = 0.8):
#     """
#     img_tensor: [3,H,W] from your test dataset (mask-centered).
#     Returns a list of TTA views from several spatial crops.
#     """
#     all_views = []
#     spatial_crops = get_fixed_multi_crops(
#         img_tensor, base_size=base_size, inner_ratio=inner_ratio
#     )
#
#     for crop in spatial_crops:
#         tta_views = apply_tta(crop)
#         all_views.extend(tta_views)
#
#     return all_views  # list of [3,H,W]

def apply_mask_multicrop_tta(
    img4: torch.Tensor,
    crop_size: int,
    n_crops: int = 4
):
    """
    1) Get n_crops mask-centered crops.
    2) For each crop, add a horizontal flip.
    Total views = n_crops * 2.
    """
    base_crops = get_mask_based_crops_4ch(
        img4,
        crop_size=crop_size,
        n_crops=n_crops
    )

    tta_tensors = []
    for c in base_crops:
        tta_tensors.append(c)                    # original
        tta_tensors.append(torch.flip(c, [2]))   # horizontal flip (W axis)

    return tta_tensors
