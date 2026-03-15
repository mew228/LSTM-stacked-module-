"""
Data module — Synthetic sequence dataset and batching utilities.
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset


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
