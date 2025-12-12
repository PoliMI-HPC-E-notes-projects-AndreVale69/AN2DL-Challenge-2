import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from timm.utils import ModelEmaV2
from tqdm import tqdm

from internal.nn.mixup_cutmix_wrapper import MixupCutmixWrapper


def train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        grad_accum_steps: int = 1,
        mixup_fn: MixupCutmixWrapper | None=None,
        ema_model: ModelEmaV2 | None = None
):
    """
    Train the model for one epoch.

    Args:
        model: The neural network model to train.
        loader: DataLoader providing the training data.
        optimizer: The optimizer for updating model weights.
        criterion: Loss function.
        device: Device to run the training on (CPU or GPU).
        grad_accum_steps: Number of steps to accumulate gradients before updating weights.
        mixup_fn: Optional mixup function for data augmentation.
        ema_model: Optional ModelEmaV2 instance for EMA tracking.
    """
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    optimizer.zero_grad()
    num_batches = len(loader)

    for step, (imgs, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Keep the original labels for metrics (not the mixed ones)
        targets_for_metrics = labels.clone()

        # ----- forward + loss (with or without mixup) -----
        if mixup_fn is not None:
            imgs_aug, y1, y2, lam, mode = mixup_fn(imgs, labels)
            logits = model(imgs_aug)

            if mode == "none":
                loss = criterion(logits, y1)
            else:
                loss = lam * criterion(logits, y1) + (1.0 - lam) * criterion(logits, y2)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)

        # store *true* loss value for logging (before scaling)
        loss_value = loss.item()

        # ----- gradient accumulation: scale loss before backward -----
        loss = loss / grad_accum_steps
        loss.backward()

        # update every grad_accum_steps or at the very last batch
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
            optimizer.step()

            if ema_model is not None:
                ema_model.update(model)

            optimizer.zero_grad()

        # ----- metrics / logging -----
        running_loss += loss_value * imgs.size(0)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets_for_metrics.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')

    print(f"    t_loss={epoch_loss:.4f} | F1(macro)={f1:.4f} | Acc={acc:.4f}")

    return epoch_loss, acc, f1

@torch.no_grad()
def validate(model, loader, criterion, device, idx2label=None, print_report=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)

    print("Confusion matrix:\n", cm)

    if print_report:
        if idx2label is None:
            # fallback: class names are just indices
            target_names = [str(i) for i in sorted(set(all_targets.tolist() + all_preds.tolist()))]
        else:
            target_names = [idx2label[i] for i in range(len(idx2label))]

        print("\nPer-class report (VAL):")
        print(classification_report(all_targets, all_preds, target_names=target_names, digits=3))

    print("Pred distribution:", np.bincount(all_preds, minlength=4))
    print("True distribution:", np.bincount(all_targets, minlength=4))

    return epoch_loss, acc, f1
