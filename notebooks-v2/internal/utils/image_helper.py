"""
Image processing helper functions.
"""
import cv2
import imagehash
import numpy as np
from PIL import Image


# -------------------------
# IO helpers
# -------------------------
def read_rgb(path: str) -> np.ndarray:
    """
    Read an RGB image from disk.
    :param path: Path to image file.
    :return: RGB image as a NumPy array.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def read_mask(path: str) -> np.ndarray:
    """
    Read a binary mask image from disk.
    :param path: Path to mask file.
    :return: Binary mask as a NumPy array (uint8, 0/1).
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8)


# -------------------------
# Geometry helpers
# -------------------------
def clamp_center(cx: int, cy: int, w: int, h: int, half: int) -> tuple[int, int]:
    """
    Clamp center coordinates to ensure a patch of size (2*half x 2*half) fits inside image of size (w x h).
    :param cx: Center x-coordinate.
    :param cy: Center y-coordinate.
    :param w: Image width.
    :param h: Image height.
    :param half: Half patch size.
    :return: Clamped (cx, cy).
    """
    cx = int(np.clip(cx, half, w - half - 1))
    cy = int(np.clip(cy, half, h - half - 1))
    return cx, cy

def crop_patch(img: np.ndarray, cx: int, cy: int, ps: int) -> np.ndarray:
    """
    Crop a square patch of size (ps x ps) from image centered at (cx, cy).
    :param img: Input image as a NumPy array.
    :param cx: Center x-coordinate.
    :param cy: Center y-coordinate.
    :param ps: Patch size.
    :return: Cropped patch as a NumPy array.
    """
    half = ps // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    return img[y0:y1, x0:x1]

def far_enough(cx: int, cy: int, centers: list[tuple[int, int]], min_dist_sq: int) -> bool:
    """
    Check if point (cx, cy) is at least min_dist away from all points in centers.
    :param cx: x-coordinate of point.
    :param cy: y-coordinate of point.
    :param centers: List of (x, y) center points.
    :param min_dist_sq: Minimum squared distance.
    :return: True if far enough, False otherwise.
    """
    for x, y in centers:
        if (cx - x) * (cx - x) + (cy - y) * (cy - y) < min_dist_sq:
            return False
    return True

def box_from_center(cx: int, cy: int, half: int) -> tuple[int, int, int, int]:
    """
    Create a bounding box from center coordinates and half-size.
    :param cx: Center x-coordinate.
    :param cy: Center y-coordinate.
    :param half: Half-size of the box.
    :return: Bounding box as (x0, y0, x1, y1).
    """
    return cx - half, cy - half, cx + half, cy + half

def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    See: https://en.wikipedia.org/wiki/Jaccard_index
    :param a: First bounding box as (x0, y0, x1, y1).
    :param b: Second bounding box as (x0, y0, x1, y1).
    :return: IoU value.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (ax1-ax0) * (ay1-ay0)
    area_b = (bx1-bx0) * (by1-by0)
    return inter / (area_a + area_b - inter)

def too_much_overlap(new_box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], max_iou=0.7) -> bool:
    """
    Check if new_box has IoU > max_iou with any box in boxes.
    In other words, check if new_box overlaps too much with existing boxes.
    :param new_box: New bounding box as (x0, y0, x1, y1).
    :param boxes: List of existing bounding boxes.
    :param max_iou: Maximum allowed IoU.
    :return: True if too much overlap, False otherwise.
    """
    return any(iou(new_box, b) > max_iou for b in boxes)

# -------------------------
# Robust tissue detector (LAB distance from background)
# -------------------------
def tissue_mask_from_rgb_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Compute a tissue mask from an RGB image using LAB color space distance from background.
    A tissue mask is a binary mask where tissue pixels are marked as 1 and background pixels as 0.
    Tissue pixels are identified as those that differ significantly in color from the border pixels,
    which are assumed to represent the background.

    What is LAB color space?
    The LAB (L*a*b*) color space is a color representation that separates lightness (L*)
    from color information (a* and b* channels).
    
    It is designed to be more perceptually uniform than RGB,
    meaning that a given numerical change corresponds to a similar perceived change in color.
    
    This makes LAB useful for tasks like color-based segmentation,
    as it can better capture how humans perceive color differences.
    :param rgb: Input RGB image as a NumPy array.
    :return: Binary tissue mask as a NumPy array.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(8, min(h, w) // 50)  # adaptive border thickness

    border = np.concatenate([
        lab[:b, :, :].reshape(-1, 3),
        lab[-b:, :, :].reshape(-1, 3),
        lab[:, :b, :].reshape(-1, 3),
        lab[:, -b:, :].reshape(-1, 3),
    ], axis=0)

    bg_mean = border.mean(axis=0)
    bg_std  = border.std(axis=0) + 1e-6

    d = ((lab - bg_mean) / bg_std) ** 2
    d = d.sum(axis=2)

    # keep top ~10% most "non-background" pixels
    thr = np.percentile(d, 90)
    t = (d > thr).astype(np.uint8)

    top_h = detect_top_shadow_by_step(rgb)
    if top_h > 0:
        t[:top_h, :] = 0

    top2, bot_h = detect_shadow_bands(rgb)
    if bot_h > 0:
        t[h-bot_h:, :] = 0

    t = cv2.medianBlur(t, 5)
    kernel = np.ones((5, 5), np.uint8)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel, iterations=1)

    return t

def tissue_fraction(rgb_patch: np.ndarray) -> float:
    """
    Compute the fraction of tissue pixels in an RGB patch.
    :param rgb_patch: Input RGB patch as a NumPy array.
    :return: Fraction of tissue pixels (between 0 and 1).
    """
    return float(tissue_mask_from_rgb_lab(rgb_patch).mean())

def detect_shadow_bands(rgb: np.ndarray, max_frac: float = 0.12) -> tuple[int, int]:
    """
    Detect scanner shadow bands at top and bottom of the image.
    Returns the heights of the top and bottom shadow bands in pixels.
    :param rgb: Input RGB image as a NumPy array.
    :param max_frac: Maximum fraction of image height to consider for shadow bands.
    :return: Tuple of (top_shadow_height, bottom_shadow_height).
    """
    gray = rgb.mean(axis=2).astype(np.float32)
    h = gray.shape[0]
    k = max(10, int(h * max_frac))

    row_mean = gray.mean(axis=1)
    row_std  = gray.std(axis=1)

    # Use interior as reference
    interior = slice(k, h - k) if h > 2*k else slice(h//4, 3*h//4)
    ref_mean = np.median(row_mean[interior])
    ref_std  = np.median(row_std[interior])

    # Shadow rows: very uniform AND significantly darker than interior
    std_thr  = max(3.0, 0.35 * ref_std)
    mean_thr = ref_mean - 0.10 * ref_mean  # 10% darker than interior median

    top_h = 0
    for i in range(k):
        if (row_std[i] < std_thr) and (row_mean[i] < mean_thr):
            top_h += 1
        else:
            break

    bot_h = 0
    for i in range(h-1, h-k-1, -1):
        if (row_std[i] < std_thr) and (row_mean[i] < mean_thr):
            bot_h += 1
        else:
            break

    return top_h, bot_h

def has_top_shadow(patch_rgb: np.ndarray, band: int = 40, mid: int = 80, mean_drop: float = 8.0, std_max: float = 6.0) -> bool:
    """
    Heuristic to detect if the top of the patch has a shadow band.
    :param patch_rgb: Input RGB patch as a NumPy array.
    :param band: Height of the top band to analyze.
    :param mid: Height of the middle band to analyze.
    :param mean_drop: Minimum mean drop from mid to top to consider as shadow.
    :param std_max: Maximum standard deviation in top band to consider as shadow.
    :return: True if top shadow is detected, False otherwise.
    """
    gray = patch_rgb.mean(axis=2).astype(np.float32)

    h = gray.shape[0]
    band = min(band, h//3)
    mid0 = h//2 - mid//2
    mid1 = h//2 + mid//2

    top = gray[:band, :]
    midp = gray[mid0:mid1, :]

    top_mean = float(top.mean())
    top_std  = float(top.std())
    mid_mean = float(midp.mean())

    # shadow = darker than center AND uniform/smooth
    return (mid_mean - top_mean) > mean_drop and top_std < std_max

def detect_top_shadow_by_step(rgb: np.ndarray, max_frac: float = 0.15) -> int:
    """
    Detect top shadow band by looking for a strong upward step in row mean intensity.
    Returns the height of the top shadow band in pixels.
    :param rgb: Input RGB image as a NumPy array.
    :param max_frac: Maximum fraction of image height to consider for shadow band.
    :return: Height of top shadow band in pixels.
    """
    gray = rgb.mean(axis=2).astype(np.float32)
    h = gray.shape[0]
    k = max(20, int(h * max_frac))

    row_mean = gray.mean(axis=1)
    row_mean_s = cv2.GaussianBlur(row_mean.reshape(-1,1), (1, 31), 0).ravel()

    # derivative: big positive jump means leaving dark band
    d = np.diff(row_mean_s[:k])
    j = int(np.argmax(d))  # location of strongest upward step

    # validate: require meaningful jump
    if d[j] < 2.5:   # tweak (2-6) depending on slides
        return 0

    # take a small safety margin
    return min(k, j + 5)

# -------------------------
# Mask component sampling
# -------------------------
def get_mask_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Get connected components from a binary mask.
    :param mask: Input binary mask as a NumPy array.
    :return: List of components, each as a tuple of (xs, ys) coordinates.
    """
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    comps = []
    for k in range(1, num):
        ys, xs = np.where(labels == k)
        if len(xs) > 0:
            comps.append((xs, ys))
    return comps

def sample_from_components(components: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> tuple[int, int]:
    """
    Sample a random point from the connected components, weighted by component area.
    :param components: List of components, each as a tuple of (xs, ys) coordinates.
    :param rng: NumPy random generator.
    :return: Sampled point as (x, y).
    """
    areas = np.array([len(c[0]) for c in components], dtype=np.float64)
    probs = areas / areas.sum()
    idx = rng.choice(len(components), p=probs)
    xs, ys = components[idx]
    j = rng.integers(0, len(xs))
    return int(xs[j]), int(ys[j])

def sample_from_binary(binmask: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    """
    Sample a random point from a binary mask.
    :param binmask: Input binary mask as a NumPy array.
    :param rng: NumPy random generator.
    :return: Sampled point as (x, y), or None if no points available.
    """
    ys, xs = np.where(binmask > 0)
    if len(xs) == 0:
        return None
    j = rng.integers(0, len(xs))
    return int(xs[j]), int(ys[j])

# -------------------------
# Patch hashing (to avoid duplicates)
# -------------------------
def quick_patch_hash(rgb: np.ndarray, size: int = 32) -> int:
    """
    Compute a quick hash for an RGB patch.
    :param rgb: Input RGB patch as a NumPy array.
    :param size: Size to downsample to for hashing.
    :return: Hash value as an integer.
    """
    # very cheap perceptual-ish hash
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return hash((small.mean(axis=2) > 128).astype(np.uint8).tobytes())

def ahash_u64(rgb_patch: np.ndarray, hash_size: int = 8) -> int:
    """
    Compute average hash (aHash) for an RGB patch, returning a uint64 integer.

    See: https://erdogant.github.io/undouble/pages/html/hash_functions.html#average-hash
    :param rgb_patch: Input RGB patch as a NumPy array.
    :param hash_size: Hash size (hash will be hash_size x hash_size).
    :return: Hash value as a uint64 integer.
    """
    # grayscale + downsample to (hash_size x hash_size)
    gray = (0.299 * rgb_patch[...,0] + 0.587 * rgb_patch[...,1] + 0.114 * rgb_patch[...,2]).astype(np.float32)
    small = gray.reshape(hash_size, rgb_patch.shape[0]//hash_size, hash_size, rgb_patch.shape[1]//hash_size).mean(axis=(1,3))
    bits = small > small.mean()
    # pack into uint64
    h = 0
    for b in bits.flatten():
        h = (h << 1) | int(b)
    return h

def patch_key(patch_rgb: np.ndarray, patch_msk: np.ndarray, hash_size: int = 16) -> tuple[imagehash.ImageHash, imagehash.ImageHash]:
    """
    Compute perceptual hashes for RGB patch and mask patch.
    :param patch_rgb: Input RGB patch as a NumPy array.
    :param patch_msk: Input mask patch as a NumPy array.
    :param hash_size: Hash size for perceptual hash.
    :return: Tuple of (RGB hash, Mask hash).
    """
    # RGB hash
    h_img = imagehash.phash(Image.fromarray(patch_rgb), hash_size=hash_size)

    # Mask hash (squeeze + uint8)
    m = (patch_msk > 0).astype(np.uint8) * 255
    h_msk = imagehash.phash(Image.fromarray(m), hash_size=hash_size)

    return h_img, h_msk
