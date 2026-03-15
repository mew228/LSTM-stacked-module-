"""
data/jena_climate.py
────────────────────
Loads the Jena Climate 2009-2016 dataset and produces sliding-window
sequence samples suitable for the StackedBiLSTM model.

Dataset: 420,551 rows × 14 numerical features, recorded every 10 minutes.
Task   : Predict air temperature (T (degC)) one step ahead from a window
         of past observations.

Usage:
    from data.jena_climate import load_jena_dataset
    train_loader, val_loader, meta = load_jena_dataset(
        csv_path="jena_climate_2009_2016.csv",
        window_size=72,      # 72 × 10min = 12 hours of context
        step=6,              # sliding step (reduces dataset to ~1/6)
        val_fraction=0.2,
        batch_size=32,
    )
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ── Column definitions ────────────────────────────────────────────────────────

# Target column we want to predict
TARGET_COL = "T (degC)"

# All numerical feature columns (drop 'Date Time' and 'wd (deg)' which is cyclic)
FEATURE_COLS = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
]

# Human-readable labels for the Streamlit UI
FEATURE_LABELS = {
    "p (mbar)"        : "Pressure (mbar)",
    "T (degC)"        : "Temperature (°C)  ← target",
    "Tpot (K)"        : "Potential Temp (K)",
    "Tdew (degC)"     : "Dew Point (°C)",
    "rh (%)"          : "Relative Humidity (%)",
    "VPmax (mbar)"    : "Sat. Vapour Pressure (mbar)",
    "VPact (mbar)"    : "Actual Vapour Pressure (mbar)",
    "VPdef (mbar)"    : "Vapour Pressure Deficit (mbar)",
    "sh (g/kg)"       : "Specific Humidity (g/kg)",
    "H2OC (mmol/mol)" : "Water Vapour Conc. (mmol/mol)",
    "rho (g/m**3)"    : "Air Density (g/m³)",
    "wv (m/s)"        : "Wind Velocity (m/s)",
    "max. wv (m/s)"   : "Max Wind Velocity (m/s)",
}

INPUT_SIZE = len(FEATURE_COLS)   # 13


# ── Dataset class ─────────────────────────────────────────────────────────────

class JenaClimateDataset(Dataset):
    """
    Sliding-window dataset over the Jena Climate CSV.

    Each sample:
        x      : (window_size, INPUT_SIZE) — z-score normalised
        length : window_size (fixed-length, so lengths are all equal)
        target : temperature at the *next* step after the window (scalar)
    """

    def __init__(
        self,
        data     : np.ndarray,   # (N, INPUT_SIZE)  — already normalised
        targets  : np.ndarray,   # (N,)              — raw temp values (normalised)
        window_size: int = 72,
        step       : int = 6,
    ):
        self.X       = data
        self.y       = targets
        self.window  = window_size
        self.step    = step
        # Valid start indices
        self.indices = list(range(0, len(data) - window_size - 1, step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.window
        x     = torch.tensor(self.X[start:end], dtype=torch.float32)
        t     = torch.tensor([self.y[end]], dtype=torch.float32)
        return x, self.window, t


def collate_fn(batch):
    """Fixed-length collate — no padding needed but kept compatible."""
    xs, lengths, targets = zip(*batch)
    return torch.stack(xs), torch.tensor(lengths), torch.stack(targets)


# ── Public loader ─────────────────────────────────────────────────────────────

def load_jena_dataset(
    csv_path    : str,
    window_size : int   = 72,
    step        : int   = 6,
    val_fraction: float = 0.2,
    batch_size  : int   = 32,
    max_rows    : int   = None,   # set e.g. 50_000 for faster experiments
    seed        : int   = 42,
):
    """
    Load, preprocess, and split the Jena Climate dataset.

    Returns:
        train_loader : DataLoader
        val_loader   : DataLoader
        meta         : dict with keys:
                         input_size, window_size, feature_cols,
                         mean_, std_, n_train, n_val, target_col
    """
    df = pd.read_csv(csv_path, parse_dates=["Date Time"])

    # Keep only numerical feature columns
    avail = [c for c in FEATURE_COLS if c in df.columns]
    data  = df[avail].values.astype(np.float32)

    if max_rows:
        data = data[:max_rows]

    # ── Global z-score normalisation ──────────────────────────────────────────
    mean_ = data.mean(axis=0, keepdims=True)
    std_  = data.std(axis=0, keepdims=True) + 1e-8
    data_norm = (data - mean_) / std_

    # Target: normalised temperature column
    t_idx   = avail.index(TARGET_COL)
    targets = data_norm[:, t_idx]

    # ── Build dataset (chronological split — NO random shuffle) ───────────────
    n_total  = len(data_norm) - window_size - 1
    n_val_pt = int(n_total * val_fraction)
    n_trn_pt = n_total - n_val_pt

    train_ds = JenaClimateDataset(
        data_norm[:n_trn_pt + window_size + 1],
        targets  [:n_trn_pt + window_size + 1],
        window_size=window_size, step=step,
    )
    val_ds = JenaClimateDataset(
        data_norm[n_trn_pt:],
        targets  [n_trn_pt:],
        window_size=window_size, step=step,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    meta = dict(
        input_size   = len(avail),
        window_size  = window_size,
        feature_cols = avail,
        mean_        = mean_,
        std_         = std_,
        n_train      = len(train_ds),
        n_val        = len(val_ds),
        target_col   = TARGET_COL,
        target_idx   = t_idx,
    )

    return train_loader, val_loader, meta


def denormalise_temp(val_norm: float, meta: dict) -> float:
    """Convert a normalised temperature prediction back to °C."""
    t_idx = meta["target_idx"]
    return float(val_norm * meta["std_"][0, t_idx] + meta["mean_"][0, t_idx])


def get_sample_window(csv_path: str, meta: dict, start_idx: int = 10_000):
    """
    Return a single window of real data for the Inference page.
    Returns: x_norm (window_size, F), raw_temps (window_size,)
    """
    df   = pd.read_csv(csv_path, parse_dates=["Date Time"])
    avail = meta["feature_cols"]
    data  = df[avail].values.astype(np.float32)

    window = meta["window_size"]
    chunk  = data[start_idx : start_idx + window]
    chunk_norm = (chunk - meta["mean_"]) / meta["std_"]

    raw_temps = chunk[:, meta["target_idx"]]   # actual °C values
    return chunk_norm, raw_temps
