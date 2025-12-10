import torch


def get_mask_based_crops_4ch(
    img4: torch.Tensor,      # [4, H, W]  (RGB + mask)
    crop_size: int = 512,
    n_crops: int = 4,
    mask_channel: int = 3,   # 0,1,2 = RGB; 3 = mask
) -> list[torch.Tensor]:
    """
    Generate n_crops crops focused around the lesion region using the mask
    contained in the channel `mask_channel` of img4.

    If the mask is empty, falls back to a single central crop.
    Returns: list of tensors [4, crop_size, crop_size].
    """
    C, H, W = img4.shape
    assert mask_channel < C, f"mask_channel={mask_channel} but img has {C} channels"

    # mask: [1, H, W] -> [H, W]
    mask = img4[mask_channel:mask_channel+1]  # [1, H, W]
    mask_2d = mask.squeeze(0)                # [H, W]

    # --- 1) Locate non-zero (tumor) pixels ---
    ys, xs = torch.where(mask_2d > 0.1)  # now only 2 outputs

    # Fallback: no mask -> central crop
    if ys.numel() == 0:
        y1 = max(0, (H - crop_size) // 2)
        x1 = max(0, (W - crop_size) // 2)
        y1 = min(y1, H - crop_size)
        x1 = min(x1, W - crop_size)
        central = img4[:, y1:y1+crop_size, x1:x1+crop_size]
        return [central]

    # --- 2) Tumor centroid ---
    cy = int(ys.float().mean())
    cx = int(xs.float().mean())

    crops = []
    for _ in range(n_crops):
        # small random jitter around the centroid
        dy = int(torch.randint(-crop_size // 4, crop_size // 4 + 1, (1,)))
        dx = int(torch.randint(-crop_size // 4, crop_size // 4 + 1, (1,)))

        y1 = cy + dy - crop_size // 2
        x1 = cx + dx - crop_size // 2

        # clip to valid range
        y1 = max(0, min(H - crop_size, y1))
        x1 = max(0, min(W - crop_size, x1))

        crop = img4[:, y1:y1+crop_size, x1:x1+crop_size]
        crops.append(crop)

    return crops
