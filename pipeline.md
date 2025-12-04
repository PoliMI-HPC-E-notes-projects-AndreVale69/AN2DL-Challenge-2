# Pipeline for this project

<!-- This is a markdown file that describes the pipeline for the project. It is a sort of TODOs to implement -->

This document outlines the pipeline for the project, detailing the steps and processes involved in its implementation.

## 🧱 0. High-level idea

* **Task**: 4-class image classification
* **Key weapons**:

  * Use **pretrained CNN** (EfficientNet / ConvNeXt)
  * Use the **masks to crop the tumor region**
  * **5-fold stratified CV**
  * **Class-weighted loss**
  * Strong but reasonable **augmentations**
  * **Ensemble** the 5 folds at test time

This is standard winning medicine on small medical image datasets.

---

## 1️⃣ Data organization

1. Unzip:

   * `train_data/` → 2,824 PNGs (1,412 images + 1,412 masks)
   * `test_data/` → 1,908 PNGs (954 images + 954 masks)
2. Inspect filenames: usually something like
   `xxx.png` and `xxx_mask.png` or similar.
3. Build a table `df` with columns:

   * `sample_index`
   * `image_path`
   * `mask_path`
   * `label` (from `train_labels.csv`)

You’ll use this `df` for splits and for your custom Dataset.

---

## 2️⃣ Validation: 5-fold Stratified CV

Use **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)** on `label`:

* For each fold `k`:

  * `train_idx`, `val_idx`
  * Train a model on `train_idx`, validate on `val_idx`
  * Save best weights by **val macro F1** or accuracy

Later, you’ll use the **5 models as an ensemble**.

---

## 3️⃣ How to use the masks (recommended variant)

### ✅ Strategy: **Crop around the mask + padding**

For each image:

1. Load RGB image and its binary mask.
2. If the mask has any positive pixels:

   * Find mask **bounding box** (min/max row/col of nonzero pixels).
   * Add some padding (e.g. 10–15% of bbox size on each side).
   * Clip to image boundaries.
   * Crop both image and mask using that bbox.
3. If the mask is empty (no positive pixels – rare but possible):

   * Fallback: use the **full image** or a central crop.
4. Resize the crop to a **fixed size**, e.g. **512×512**.

**Why this is good**:
You keep only relevant tissue, keep good resolution, and reduce background noise.

You do this **inside your Dataset.**getitem****, so it’s all on-the-fly.

---

## 4️⃣ Transformations & Augmentation

I’d use **albumentations** (but torchvision is fine too).

**Train transforms (on the cropped image):**

* Resize → 512×512 (if not already)
* Horizontal flip, vertical flip
* Small rotation (±15° / ±20°)
* Random scale / random crop
* Color jitter:

  * brightness, contrast, saturation, hue
* Maybe **Gaussian noise** or slight blur
* Normalize with ImageNet stats
  (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

Optional but strong:

* **Mixup** or **CutMix** implemented in the training loop / collate function.

**Validation/Test transforms:**

* Just resize → 512×512
* Normalize with ImageNet stats
* (Optional later) TTA: horizontal/vertical flips at inference and average.

---

## 5️⃣ Custom PyTorch Dataset

`__getitem__(idx)` should:

1. Look up row in `df`.
2. Load image (RGB) and mask (grayscale or binary).
3. Compute bbox from mask.
4. Crop image (and mask if needed).
5. Apply augmentations (using only the image for classification).
6. Return:

   * `image` (tensor, shape [3, 512, 512])
   * `label` (int in {0..3}) for train/val, or `sample_index` for test.

---

## 6️⃣ Model architecture

Use **timm** if you can; otherwise vanilla torchvision.

Recommended main architecture:

* `convnext_tiny` or `tf_efficientnetv2_s` (ImageNet pretrained)

Modifications:

* Replace the final classifier head with:

  * `Linear(in_features, 4)`
* Add dropout in the head (e.g. 0.3–0.5).

This gives a **strong yet not too heavy** model.

---

## 7️⃣ Loss, optimizer, scheduler

### **Loss**

Because of class imbalance (Triple Negative only 11%):

* Compute **class weights** = inverse frequency (normalized), and use

  * `nn.CrossEntropyLoss(weight=class_weights)`
    OR
* Focal Loss (γ = 2) with class weights.

Start with **weighted cross-entropy**; it’s simple and works well.

### **Optimizer**

* `AdamW`

  * lr = 1e-4 (for fine-tuning the whole network)
  * weight_decay = 1e-4

### **Scheduler**

Two good options:

* CosineAnnealingLR with warmup
* OneCycleLR (fast and stable)

You can keep it simple initially:
`CosineAnnealingLR(optimizer, T_max=epochs)`.

---

## 8️⃣ Training loop (per fold)

For each fold:

* Train for, say, **15–25 epochs** with early stopping:

  * Monitor **val macro F1** or accuracy.
  * Save weights when val score improves.
* Use a reasonably large batch size:

  * 16 or 32 (depending on GPU memory with 512×512 images).
* Mixup/CutMix: if you use them, reduce lr marginally or train a bit longer.

Track:

* Train loss
* Val loss
* Val accuracy
* Val macro F1 (more informative with imbalance)

---

## 9️⃣ Inference & Kaggle submission

### On test set:

1. Build a test Dataset that:

   * Loads image and mask
   * Crops around mask (same as train/val)
   * Applies test transforms
2. For each fold model:

   * Load best weights
   * Predict **probabilities** for each class (softmax outputs)
3. **Average probabilities across all 5 folds**.
4. Choose `argmax` for final class.
5. Map {0..3} → {Luminal A, Luminal B, HER2(+), Triple negative}.
6. Create `submission.csv` with columns:

   * `sample_index`
   * `label`

Upload to Kaggle → check leaderboard.

---

## 🔟 Suggested “phases” so you don’t get lost

1. **Baseline**:

   * Ignore masks; full image, resize 512×512, single split, ResNet50, plain CE.
2. **Better backbone + masks**:

   * Switch to ConvNeXt/EfficientNet.
   * Implement mask-based cropping.
3. **Proper CV + class weights**:

   * 5-fold stratified CV.
   * Weighted CE.
4. **Augmentation boost**:

   * Strong color + geometric aug.
   * (Optional) mixup / cutmix.
5. **TTA & small tweaks**:

   * Test time flips.
   * Maybe slight LR tuning or extra epochs.

At each phase you should see a **Kaggle private score improvement**.
