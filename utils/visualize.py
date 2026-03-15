"""
Utils — Visualisation module. Generates the 4-panel diagnostic figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


def plot_diagnostics(model, history, val_loader, device, save_path="bilstm_diagnostics.png"):
    """
    4-panel figure:
      ┌─────────────────────┬─────────────────────┐
      │  Train vs Val Loss  │  Gradient Norm       │
      ├─────────────────────┼─────────────────────┤
      │  Output Histogram   │  LSTM Weight Heatmap │
      └─────────────────────┴─────────────────────┘
    """

    DARK   = "#0d1117"
    ACCENT = "#58a6ff"
    GREEN  = "#3fb950"
    ORANGE = "#f78166"
    PURPLE = "#bc8cff"
    TEXT   = "#e6edf3"
    GRID   = "#30363d"

    fig = plt.figure(figsize=(14, 10), facecolor=DARK)
    fig.suptitle(
        "Stacked BiLSTM — Training Diagnostics",
        fontsize=16, fontweight="bold", color=TEXT, y=0.97
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linestyle="--", linewidth=0.5, alpha=0.7)

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Panel 1: Loss Curves ──────────────────────────────────────────────────
    ax = axes[0]
    style_ax(ax, "Train vs Validation Loss", "Epoch", "MSE Loss")
    ax.plot(epochs, history["train_loss"], color=ACCENT,  lw=2,   label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color=ORANGE,  lw=2,   label="Val Loss", linestyle="--")
    ax.fill_between(epochs, history["train_loss"], history["val_loss"],
                    alpha=0.08, color=PURPLE)
    ax.legend(facecolor=DARK, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # ── Panel 2: Gradient Norm ────────────────────────────────────────────────
    ax = axes[1]
    style_ax(ax, "Gradient Norm per Epoch (post-clip)", "Epoch", "‖∇‖₂")
    ax.plot(epochs, history["grad_norms"], color=GREEN, lw=2)
    ax.axhline(y=1.0, color=ORANGE, linestyle=":", lw=1.2, label="clip threshold")
    ax.fill_between(epochs, 0, history["grad_norms"], alpha=0.15, color=GREEN)
    ax.legend(facecolor=DARK, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # ── Panel 3: Output Distribution Histogram ────────────────────────────────
    ax = axes[2]
    style_ax(ax, "Model Output Distribution (Val Set)", "Predicted Value", "Count")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, lengths, targets in val_loader:
            preds = model(x_batch.to(device), lengths)
            all_preds.extend(preds.cpu().squeeze().tolist())
            all_targets.extend(targets.squeeze().tolist())

    ax.hist(all_preds,   bins=25, color=ACCENT,  alpha=0.75, label="Predictions", edgecolor=DARK)
    ax.hist(all_targets, bins=25, color=ORANGE,  alpha=0.55, label="Targets",     edgecolor=DARK)
    ax.legend(facecolor=DARK, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # ── Panel 4: LSTM Weight Heatmap ──────────────────────────────────────────
    ax = axes[3]
    last_layer_idx = model.num_layers - 1
    weight_name    = f"weight_ih_l{last_layer_idx}"
    weight_matrix  = model.lstm.state_dict()[weight_name].cpu().numpy()

    w_slice = weight_matrix[:64, :]
    im = ax.imshow(w_slice, aspect="auto", cmap="RdBu_r",
                   vmin=-weight_matrix.std() * 2, vmax=weight_matrix.std() * 2)

    ax.set_facecolor("#161b22")
    ax.set_title(f"LSTM {weight_name} — Weight Heatmap\n(first 64 rows shown)",
                 color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Input Features", color=TEXT, fontsize=9)
    ax.set_ylabel("Gate Units (ih)", color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT, labelsize=7)
    cbar.outline.set_edgecolor(GRID)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    return save_path
