# 🧠 Stacked Bidirectional LSTM

A clean, modular PyTorch implementation of a **Stacked Bidirectional LSTM** for sequence-to-one prediction tasks — with gradient-based optimization, z-score data normalisation, checkpoint saving, config-driven training, and visual diagnostics.

---

## 📁 Project Structure

```
LSTM structure/
├── models/
│   └── stacked_bilstm.py       ← StackedBiLSTM class
├── data/
│   └── dataset.py              ← SyntheticSeqDataset + collate_fn
├── training/
│   ├── config.py               ← TrainConfig dataclass (YAML loader)
│   └── trainer.py              ← Training loop + checkpoint saving
├── utils/
│   └── visualize.py            ← 4-panel diagnostic figure
├── configs/
│   └── base.yaml               ← Default hyperparameters
├── checkpoints/                ← Best model saved here (auto-created)
│   └── best_model.pt
├── train.py                    ← Entry point (argparse)
├── requirements.txt
├── bilstm_diagnostics.png      ← Auto-generated after training
└── stacked_bilstm.py           ← Original monolithic file (reference)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train with default config

```bash
python train.py
```

### 3. Train with a custom config

```bash
python train.py --config configs/base.yaml
```

---

## ⚙️ Configuration (`configs/base.yaml`)

| Key | Default | Description |
|---|---|---|
| `input_size` | 32 | Features per time step |
| `hidden_size` | 128 | LSTM hidden units per direction |
| `num_layers` | 3 | Number of stacked BiLSTM layers |
| `dropout` | 0.3 | Dropout rate |
| `epochs` | 30 | Training epochs |
| `lr` | 3e-4 | Learning rate (AdamW) |
| `batch_size` | 32 | Batch size |
| `n_samples` | 512 | Synthetic dataset size |

---

## 🔖 Checkpointing

The trainer automatically saves `checkpoints/best_model.pt` whenever validation loss improves. To reload:

```python
import torch
from models.stacked_bilstm import StackedBiLSTM

model = StackedBiLSTM(input_size=32, hidden_size=128, num_layers=3)
ckpt  = torch.load("checkpoints/best_model.pt")
model.load_state_dict(ckpt["model_state"])
print(f"Best val loss: {ckpt['val_loss']:.5f}  (epoch {ckpt['epoch']})")
```

---

## 🤔 What is a Stacked BiLSTM?

| Feature | Explanation |
|---|---|
| **Bidirectional** | Reads the sequence forwards AND backwards |
| **Stacked (3 layers)** | Each layer learns more abstract patterns |
| **Forget Gate** | Controls how much memory to keep at each step |

### Training pipeline

```
Raw sequences → Z-score Normalise → Pack → 3-layer BiLSTM
    → Final hidden state concat → FC Head → MSE Loss
    → AdamW + CosineAnnealingLR + Gradient Clip → update
```

---

## 📊 Diagnostic Chart (`bilstm_diagnostics.png`)

4-panel figure auto-generated after training:

```
┌──────────────────────┬──────────────────────┐
│  Loss Curve          │  Gradient Norm        │
│  (train vs val)      │  (per epoch)          │
├──────────────────────┼──────────────────────┤
│  Output Histogram    │  LSTM Weight Heatmap  │
│  (pred vs target)    │  (colour map)         │
└──────────────────────┴──────────────────────┘
```

---

## 📈 Expected Results (~30 epochs)

| Metric | Value |
|---|---|
| Final Train Loss | ~0.001 |
| Final Val Loss | ~0.0003 |
| Gradient Norm | < 0.1 |
| Trainable Parameters | ~1.2 million |

---

## 🏗️ Architecture

```
Input (32 features/step)
    │
    ▼
┌─────────────────────────────┐
│  BiLSTM Layer 1             │  hidden: 128 × 2 = 256 dim
└────────────────┬────────────┘
                 │ dropout 0.3
┌────────────────▼────────────┐
│  BiLSTM Layer 2             │
└────────────────┬────────────┘
                 │ dropout 0.3
┌────────────────▼────────────┐
│  BiLSTM Layer 3             │
└────────────────┬────────────┘
                 │  concat fwd + bwd → 256 dim
┌────────────────▼────────────┐
│  FC: 256 → 128 → ReLU → 1  │
└─────────────────────────────┘
```

---

## 🌐 Web App (Streamlit)

An interactive multi-page web app is included. Run locally with:

```bash
python -m streamlit run app.py
```

Then open **http://localhost:8501**

| Page | What it does |
|---|---|
| 🏠 Home | Architecture overview & stats |
| 🏋️ Train | Live hyperparameter tuning + loss chart |
| 🔮 Inference | Generate a sequence → get a prediction |
| 📐 Architecture | Interactive weight heatmaps & explanations |

### Deploy to Streamlit Cloud (free)
1. Push to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point to `app.py` → **Deploy**

---

*Built with PyTorch 2.x · Python 3.11 · Streamlit*
