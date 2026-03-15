"""
Training module — full training loop with AdamW, CosineAnnealingLR,
gradient clipping, and best-checkpoint saving.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs       : int   = 30,
    lr           : float = 3e-4,
    weight_decay : float = 1e-4,
    max_grad_norm: float = 1.0,
    checkpoint_dir: str  = "checkpoints",
):
    """
    Full training loop with:
      • AdamW optimiser          — decoupled weight decay
      • CosineAnnealingLR        — smooth LR decay from lr → ~0 over `epochs`
      • clip_grad_norm_          — prevents exploding gradients
      • Gradient norm tracking   — useful for diagnosing training stability
      • Best-checkpoint saving   — saves best_model.pt on val loss improvement

    Returns:
        history (dict): train_loss, val_loss, grad_norms, lr_history per epoch
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = dict(train_loss=[], val_loss=[], grad_norms=[], lr_history=[])

    header = f"{'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>10} │ {'Grad Norm':>10} │ {'LR':>10} │ {'Saved':>6}"
    sep    = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for epoch in range(1, epochs + 1):

        # ── Training Phase ────────────────────────────────────────────────────
        model.train()
        epoch_loss  = 0.0
        epoch_gnorm = 0.0

        for x_batch, lengths, targets in train_loader:
            x_batch = x_batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(x_batch, lengths)
            loss  = criterion(preds, targets)
            loss.backward()

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            epoch_loss  += loss.item()
            epoch_gnorm += gnorm.item()

        avg_train = epoch_loss  / len(train_loader)
        avg_gnorm = epoch_gnorm / len(train_loader)

        # ── Validation Phase ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, lengths, targets in val_loader:
                x_batch = x_batch.to(device)
                targets = targets.to(device)
                preds   = model(x_batch, lengths)
                val_loss += criterion(preds, targets).item()

        avg_val    = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["grad_norms"].append(avg_gnorm)
        history["lr_history"].append(current_lr)

        # ── Checkpoint ────────────────────────────────────────────────────────
        saved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "epoch"     : epoch,
                    "model_state": model.state_dict(),
                    "val_loss"  : avg_val,
                },
                best_ckpt_path,
            )
            saved = "✓"

        print(
            f"{epoch:>5} │ {avg_train:>10.5f} │ {avg_val:>10.5f} │"
            f" {avg_gnorm:>10.5f} │ {current_lr:>10.2e} │ {saved:>6}"
        )

    print(sep + "\n")
    print(f"   Best checkpoint saved → {best_ckpt_path}  (val loss: {best_val_loss:.5f})")
    return history
