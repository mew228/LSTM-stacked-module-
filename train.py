"""
train.py — Entry point for the Stacked BiLSTM project.

Usage:
    python train.py                          # uses configs/base.yaml
    python train.py --config path/to/cfg.yaml
"""

import argparse
import os
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from models.stacked_bilstm import StackedBiLSTM, count_parameters
from data.dataset import SyntheticSeqDataset, collate_fn
from training.config import TrainConfig
from training.trainer import train
from utils.visualize import plot_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stacked BiLSTM")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "base.yaml"),
        help="Path to YAML config file (default: configs/base.yaml)",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = TrainConfig.from_yaml(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("      Stacked BiLSTM — Architecture & Configuration")
    print("=" * 60)
    print(f"\n{cfg}")
    print(f"\n   Device : {DEVICE}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\n▶  Building synthetic dataset (z-score normalised)...")
    dataset = SyntheticSeqDataset(
        n_samples  = cfg.n_samples,
        input_size = cfg.input_size,
        min_len    = cfg.min_len,
        max_len    = cfg.max_len,
        seed       = cfg.seed,
    )

    n_val   = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"   Train samples : {n_train}  |  Val samples : {n_val}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = StackedBiLSTM(
        input_size  = cfg.input_size,
        hidden_size = cfg.hidden_size,
        num_layers  = cfg.num_layers,
        output_size = cfg.output_size,
        dropout     = cfg.dropout,
    ).to(DEVICE)

    print(f"\n{model}")
    print(f"\n   Trainable parameters : {count_parameters(model):,}")
    print("=" * 60)

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n▶  Training with AdamW + CosineAnnealingLR + Gradient Clipping")
    print(f"   LR={cfg.lr}  |  Weight Decay={cfg.weight_decay}  |  Max Grad Norm={cfg.max_grad_norm}")

    root_dir       = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(root_dir, "checkpoints")

    history = train(
        model, train_loader, val_loader,
        device         = DEVICE,
        epochs         = cfg.epochs,
        lr             = cfg.lr,
        weight_decay   = cfg.weight_decay,
        max_grad_norm  = cfg.max_grad_norm,
        checkpoint_dir = checkpoint_dir,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    print("┌──────────────────────────────────┐")
    print("│         Training Summary          │")
    print("├──────────────────────────────────┤")
    print(f"│  Best Val Epoch   : {best_epoch:<14}│")
    print(f"│  Best Val Loss    : {min(history['val_loss']):<14.5f}│")
    print(f"│  Final Train Loss : {history['train_loss'][-1]:<14.5f}│")
    print(f"│  Final Grad Norm  : {history['grad_norms'][-1]:<14.5f}│")
    print(f"│  Final LR        : {history['lr_history'][-1]:<14.2e}│")
    print("└──────────────────────────────────┘")

    # ── Visualisation ─────────────────────────────────────────────────────────
    print("\n▶  Generating 4-panel diagnostic figure...")
    out_path = os.path.join(root_dir, "bilstm_diagnostics.png")
    saved    = plot_diagnostics(model, history, val_loader, DEVICE, save_path=out_path)
    print(f"\n✓  Diagnostics saved → {saved}")
    print("   Panels: [Loss Curve] [Gradient Norm] [Output Histogram] [Weight Heatmap]")


if __name__ == "__main__":
    main()
