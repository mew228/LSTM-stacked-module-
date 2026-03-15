"""
Models package — Stacked Bidirectional LSTM.

Exports:
    StackedBiLSTM   : The main model class.
    count_parameters: Helper to count trainable parameters.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
