"""
Training config — hyperparameters loaded from YAML.
"""

from dataclasses import dataclass, field
import yaml


@dataclass
class TrainConfig:
    # ── Model ──────────────────────────────────────────────────────────────────
    input_size  : int   = 32
    hidden_size : int   = 128
    num_layers  : int   = 3
    output_size : int   = 1
    dropout     : float = 0.3

    # ── Optimiser ──────────────────────────────────────────────────────────────
    epochs        : int   = 30
    lr            : float = 3e-4
    weight_decay  : float = 1e-4
    max_grad_norm : float = 1.0
    batch_size    : int   = 32

    # ── Dataset ────────────────────────────────────────────────────────────────
    n_samples : int = 512
    min_len   : int = 10
    max_len   : int = 50
    seed      : int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load a TrainConfig from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __str__(self) -> str:
        lines = ["TrainConfig:"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k:<16}: {v}")
        return "\n".join(lines)
