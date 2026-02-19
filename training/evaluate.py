import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Make project root importable (for helper)
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper import accuracy_np


def run_one_epoch(model, loader, optimizer, device, train_mode=True):
    """
    Run one epoch (train or validation).
    Returns (avg_loss, avg_acc, y_true_np, y_pred_np).
    """
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true_all = []
    y_pred_all = []
    n_batches = 0

    if train_mode:
        model.train()
        pbar = tqdm(loader, desc="Train", dynamic_ncols=True)
    else:
        model.eval()
        pbar = tqdm(loader, desc="Val  ", dynamic_ncols=True)

    for X, y in pbar:
        X = X.to(device)
        y = y.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(X)
            loss = loss_fn(logits, y)

            if train_mode:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.item())
        n_batches += 1

        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())

        if n_batches % 20 == 0:
            y_true_np = np.concatenate(y_true_all)
            y_pred_np = np.concatenate(y_pred_all)
            pbar.set_postfix(loss=total_loss / n_batches, acc=accuracy_np(y_true_np, y_pred_np))

    y_true_np = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
    y_pred_np = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=int)

    avg_loss = total_loss / max(1, n_batches)
    avg_acc = accuracy_np(y_true_np, y_pred_np) if len(y_true_np) else 0.0
    return avg_loss, avg_acc, y_true_np, y_pred_np

