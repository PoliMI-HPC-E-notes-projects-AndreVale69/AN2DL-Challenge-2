import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, mixup_fn=None):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Keep the original labels for metrics
        targets_for_metrics = labels.clone()

        optimizer.zero_grad()

        if mixup_fn is not None:
            imgs_aug, y1, y2, lam, mode = mixup_fn(imgs, labels)
            logits = model(imgs_aug)

            if mode == "none":
                loss = criterion(logits, y1)
            else:
                # mix the losses of the two label sets
                loss = lam * criterion(logits, y1) + (1.0 - lam) * criterion(logits, y2)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

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
