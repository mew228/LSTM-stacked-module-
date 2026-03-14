# 🧠 Stacked Bidirectional LSTM — `LSTM structure`

A clean, well-commented PyTorch implementation of a **Stacked Bidirectional LSTM** for sequence-to-one prediction tasks — with gradient-based optimization, z-score data formatting, and visual diagnostics.

---

## 📁 What's in this folder?

| File | What it does |
|---|---|
| `stacked_bilstm.py` | The full model + training script |
| `bilstm_diagnostics.png` | Auto-generated 4-panel training chart |
| `README.md` | This file — explains everything! |

---

## 🤔 What is an LSTM?

Think of reading a sentence:

> *"The cat sat on the ___"*

To predict the missing word, you need to **remember** what came before. That's exactly what an **LSTM (Long Short-Term Memory)** does — it reads a sequence step-by-step and keeps a memory of what it has seen.

### What makes this one special?

| Feature | Simple Explanation |
|---|---|
| **Bidirectional** | Reads the sequence both forwards AND backwards — like reading a sentence left-to-right and right-to-left to understand it better |
| **Stacked (3 layers)** | Like stacking 3 readers on top of each other — each one learns more abstract patterns from the one below |
| **Forget Gate** | A "memory filter" inside each cell that decides what to remember and what to forget |

---

## ⚙️ How does training work?

```
Raw sequences
    ↓
Z-score Normalise (μ=0, σ=1)
    ↓
Pack variable-length sequences
    ↓
3-layer BiLSTM → hidden state
    ↓
Fully Connected Head → prediction
    ↓
MSE Loss → AdamW Optimizer + Cosine LR → update weights
```

### Optimization techniques used

| Technique | Why? |
|---|---|
| **AdamW** | Smarter than plain SGD — adapts the learning rate per weight |
| **CosineAnnealingLR** | Smoothly shrinks the learning rate over time (like slowing down as you approach the destination) |
| **Gradient Clipping** | Stops the model from taking huge, wild update steps |

---

## 📊 What does the diagnostic chart show?

`bilstm_diagnostics.png` has 4 panels:

```
┌──────────────────────┬──────────────────────┐
│  Loss Curve          │  Gradient Norm        │
│  (train vs val)      │  (per epoch)          │
├──────────────────────┼──────────────────────┤
│  Output Histogram    │  LSTM Weight Heatmap  │
│  (pred vs target)    │  (colour map)         │
└──────────────────────┴──────────────────────┘
```

- **Loss Curve** — Both lines should go down → the model is learning ✅
- **Gradient Norm** — Should stay below 1.0 (clipping keeps it safe) ✅
- **Output Histogram** — Predictions (blue) should overlap targets (orange) ✅
- **Weight Heatmap** — Colourful = the model learned varied patterns ✅

---

## 🚀 How to run it

### Requirements

```bash
pip install torch numpy matplotlib
```

### Run

```bash
python stacked_bilstm.py
```

You'll see a live training table and a `bilstm_diagnostics.png` will be saved automatically.

---

## 🧪 Simple Example — What does one training step look like?

```python
# 1) A batch of 32 sequences, each up to 50 steps long, with 32 features
x       = torch.randn(32, 50, 32)   # (batch, time, features)
lengths = torch.randint(10, 51, (32,))  # variable lengths

# 2) Forward pass
predictions = model(x, lengths)     # shape → (32, 1)

# 3) Compute loss and update
loss = criterion(predictions, targets)
loss.backward()
optimizer.step()
```

Every step, the model gets a little bit better at predicting the target value from the sequence! 🎯

---

## 📈 Training Results (30 epochs)

| Metric | Value |
|---|---|
| Final Train Loss | ~0.001 |
| Final Val Loss | ~0.0003 |
| Gradient Norm | < 0.1 (well below clip threshold) |
| Trainable Parameters | ~1.2 million |

---

## 🏗️ Architecture at a Glance

```
Input (32 features/step)
    │
    ▼
┌─────────────────────────────┐
│  BiLSTM Layer 1             │  ← reads sequence forward + backward
│  hidden: 128 × 2 = 256 dim  │
└────────────────┬────────────┘
                 │ dropout 0.3
┌────────────────▼────────────┐
│  BiLSTM Layer 2             │
└────────────────┬────────────┘
                 │ dropout 0.3
┌────────────────▼────────────┐
│  BiLSTM Layer 3             │
└────────────────┬────────────┘
                 │  concat fwd + bwd final hidden → 256 dim
┌────────────────▼────────────┐
│  FC: 256 → 128 → ReLU → 1  │  ← single output per sequence
└─────────────────────────────┘
```

---

*Built with PyTorch 2.x · Python 3.11*
