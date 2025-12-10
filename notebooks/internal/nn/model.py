import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        grad_accum_steps: int = 1,
        mixup_fn=None
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
def validate(model, loader, criterion, device):
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
    f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)

    print("Confusion matrix:\n", cm)
    return epoch_loss, acc, f1
