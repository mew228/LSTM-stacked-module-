"""
================================================================================
Stacked Bidirectional LSTM — Sequence-to-One Task
================================================================================
Architecture:
    Input → Normalised Data → PackedSequence → BiLSTM (N layers)
          → Hidden State Concat → FC Head → Output

Techniques:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Optimizer   : AdamW  (decoupled weight decay)                       │
    │  LR Schedule : CosineAnnealingLR  (smooth decay → T_max epochs)     │
    │  Grad Clip   : clip_grad_norm_  (max_norm = 1.0)                     │
    │  Data Format : z-score normalisation  +  variable-length packing     │
    │  Visualise   : 4-panel diagnostic figure  (loss / grad / hist / hm)  │
    └──────────────────────────────────────────────────────────────────────┘

Forget Gate & Cell State (Vanishing Gradient Prevention):
    ─────────────────────────────────────────────────────
    Within each LSTM cell, the *Cell State* (C_t) acts as a long-term memory
    highway that flows through time with only elementwise operations, avoiding
    repeated matrix multiplications that cause gradients to vanish.

    The *Forget Gate* (f_t = σ(W_f·[h_{t-1}, x_t] + b_f)) controls *how much*
    of the previous cell state to retain. A gate value near 1.0 means "remember
    everything"; near 0.0 means "forget". The gradient of the loss with respect
    to earlier time steps flows *directly* through this gate, kept alive as long
    as f_t ≈ 1.

    AdamW + CosineAnnealingLR ensures stable, adaptive learning rate decay,
    while gradient clipping prevents exploding gradients that could destabilise
    the training dynamics of deep stacked BiLSTMs.
================================================================================
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — saves to file without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model Definition
# ──────────────────────────────────────────────────────────────────────────────

class StackedBiLSTM(nn.Module):
    """
    Stacked Bidirectional LSTM for sequence-to-one tasks.

    Args:
        input_size  (int)  : Number of input features per time step.
        hidden_size (int)  : Number of units in each LSTM direction per layer.
                             Effective hidden dim after bidir concat = 2×hidden_size.
        num_layers  (int)  : Number of stacked LSTM layers. Default: 2.
        output_size (int)  : Number of output units. Default: 1.
        dropout     (float): Dropout between LSTM layers and inside FC head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.num_dirs    = 2  # bidirectional

        # ── LSTM Core ─────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        # ── Fully Connected Head ──────────────────────────────────────────────
        fc_input_dim = self.num_dirs * hidden_size  # 2 × hidden_size

        self.fc_head = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_input_dim // 2, output_size),
        )

        self._init_weights()

    # ── Weight Initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Orthogonal init for recurrent weights, Xavier for input weights,
        zeros for biases — with forget gate bias set to 1.0.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)   # forget gate → 1

        for layer in self.fc_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ── Forward Pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, (h_n, _c_n) = self.lstm(packed_input)

        _output, _ = pad_packed_sequence(packed_out, batch_first=True)

        # h_n: (num_layers × num_dirs, batch, hidden_size)
        h_n = h_n.view(self.num_layers, self.num_dirs, batch_size, self.hidden_size)
        last_layer_h = h_n[-1]                                    # (2, B, H)
        h_combined   = torch.cat([last_layer_h[0], last_layer_h[1]], dim=-1)  # (B, 2H)

        return self.fc_head(h_combined)                           # (B, output_size)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Dataset with Normalised Formatting
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticSeqDataset(Dataset):
    """
    Variable-length sine + Gaussian-noise sequences.

    Each sample:
        x       : (seq_len, input_size) — z-score normalised
        length  : actual sequence length (int)
        target  : mean of the raw signal (scalar regression target)

    Args:
        n_samples   : number of sequences
        input_size  : feature dimension per time step
        min_len     : minimum sequence length
        max_len     : maximum sequence length
        seed        : random seed for reproducibility
    """

    def __init__(
        self,
        n_samples : int  = 512,
        input_size: int  = 32,
        min_len   : int  = 10,
        max_len   : int  = 50,
        seed      : int  = 42,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)

        self.samples = []
        all_raw = []

        # ── Generate raw sequences ────────────────────────────────────────────
        lengths = rng.integers(min_len, max_len + 1, size=n_samples)
        raw_seqs = []

        for length in lengths:
            t    = np.linspace(0, 2 * math.pi, length)               # time axis
            freq = rng.uniform(0.5, 3.0)                              # random freq
            amp  = rng.uniform(0.5, 2.0)                              # random amp
            sig  = amp * np.sin(freq * t)                             # base signal
            # Expand to input_size features with slight variation per feature
            noise   = rng.normal(0, 0.1, size=(length, input_size))
            feat    = sig[:, None] + noise                            # (L, F)
            raw_seqs.append(feat)
            all_raw.append(feat)

        # ── Compute global z-score stats across ALL data ──────────────────────
        all_concat  = np.concatenate(all_raw, axis=0)                 # (N_total, F)
        self.mean_  = all_concat.mean(axis=0, keepdims=True)          # (1, F)
        self.std_   = all_concat.std(axis=0, keepdims=True) + 1e-8    # (1, F)

        # ── Normalise & build dataset ─────────────────────────────────────────
        for i, feat in enumerate(raw_seqs):
            x_norm  = (feat - self.mean_) / self.std_                 # z-score
            target  = float(feat.mean())                              # regression label
            self.samples.append(
                (torch.tensor(x_norm, dtype=torch.float32),
                 int(lengths[i]),
                 torch.tensor([target], dtype=torch.float32))
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch):
    """
    Pad variable-length sequences in a batch to the same length.
    Returns:
        x_padded : (B, max_len, F)
        lengths  : (B,)  — CPU tensor
        targets  : (B, 1)
    """
    xs, lengths, targets = zip(*batch)
    max_len = max(lengths)

    padded = torch.zeros(len(xs), max_len, xs[0].size(-1))
    for i, (x, l) in enumerate(zip(xs, lengths)):
        padded[i, :l] = x

    return padded, torch.tensor(lengths), torch.stack(targets)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Gradient-Based Optimization Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs     : int   = 30,
    lr         : float = 3e-4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
):
    """
    Full training loop with:
      • AdamW optimiser          — decoupled weight decay for better generalisation
      • CosineAnnealingLR        — smooth LR decay from lr → ~0 over `epochs` steps
      • clip_grad_norm_          — prevents exploding gradients (max_norm = 1.0)
      • Gradient norm tracking   — useful for diagnosing training stability

    Returns:
        history (dict): train_loss, val_loss, grad_norms, lr_history per epoch
    """
    criterion = nn.MSELoss()

    # AdamW: Adam with correctly decoupled weight decay (Loshchilov & Hutter, 2019)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Cosine Annealing: lr(t) = lr_min + 0.5*(lr_max-lr_min)*(1+cos(π·t/T_max))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = dict(train_loss=[], val_loss=[], grad_norms=[], lr_history=[])

    header = f"{'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>10} │ {'Grad Norm':>10} │ {'LR':>10}"
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

            # Gradient clipping — prevents exploding gradients
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

        avg_val = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        scheduler.step()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["grad_norms"].append(avg_gnorm)
        history["lr_history"].append(current_lr)

        print(f"{epoch:>5} │ {avg_train:>10.5f} │ {avg_val:>10.5f} │ {avg_gnorm:>10.5f} │ {current_lr:>10.2e}")

    print(sep + "\n")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# 4. Visualisation — 4-Panel Diagnostic Figure
# ──────────────────────────────────────────────────────────────────────────────

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
    # Extract weight_ih of the last layer (forward direction)
    last_layer_idx = model.num_layers - 1
    weight_name    = f"weight_ih_l{last_layer_idx}"
    weight_matrix  = model.lstm.state_dict()[weight_name].cpu().numpy()

    # Show a 64×32 slice for readability
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


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main — Smoke Test + Full Training + Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE   = 32
    INPUT_SIZE   = 32
    HIDDEN_SIZE  = 128
    NUM_LAYERS   = 3
    OUTPUT_SIZE  = 1
    DROPOUT      = 0.3
    EPOCHS       = 30
    LR           = 3e-4

    # ── Print header ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("      Stacked BiLSTM — Architecture & Configuration")
    print("=" * 60)

    # ── Build Dataset ─────────────────────────────────────────────────────────
    print("\n▶  Building synthetic dataset (z-score normalised)...")
    dataset = SyntheticSeqDataset(
        n_samples  = 512,
        input_size = INPUT_SIZE,
        min_len    = 10,
        max_len    = 50,
        seed       = 42,
    )

    n_val  = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"   Train samples : {n_train}  |  Val samples : {n_val}")
    print(f"   Input z-score :  μ ≈ 0.0   σ ≈ 1.0  (across {INPUT_SIZE} features)")
    print(f"   Device        : {DEVICE}")

    # ── Build Model ───────────────────────────────────────────────────────────
    model = StackedBiLSTM(
        input_size  = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        output_size = OUTPUT_SIZE,
        dropout     = DROPOUT,
    ).to(DEVICE)

    print(f"\n{model}")
    print(f"\n   Trainable parameters : {count_parameters(model):,}")
    print("=" * 60)

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n▶  Training with AdamW + CosineAnnealingLR + Gradient Clipping")
    print(f"   LR = {LR}  |  Weight Decay = 1e-4  |  Max Grad Norm = 1.0")

    history = train(
        model, train_loader, val_loader,
        device=DEVICE, epochs=EPOCHS, lr=LR
    )

    # ── Summary Table ─────────────────────────────────────────────────────────
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
    import os
    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "bilstm_diagnostics.png")

    saved = plot_diagnostics(model, history, val_loader, DEVICE, save_path=out_path)
    print(f"\n✓  Diagnostics saved → {saved}")
    print("   Panels: [Loss Curve] [Gradient Norm] [Output Histogram] [Weight Heatmap]")
