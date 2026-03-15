"""
Train page — configure hyperparameters, select dataset, and run live training.
"""

import os
import time
import torch
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.stacked_bilstm import StackedBiLSTM, count_parameters
from data.dataset import SyntheticSeqDataset, collate_fn as synth_collate
from data.jena_climate import (
    load_jena_dataset,
    FEATURE_COLS,
    INPUT_SIZE as JENA_INPUT_SIZE,
    TARGET_COL,
    FEATURE_LABELS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Train · BiLSTM", page_icon="🏋️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .ds-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .ds-title { font-weight: 700; color: #e6edf3; font-size: 1rem; }
    .ds-desc  { color: #8b949e; font-size: 0.83rem; margin-top: 0.2rem; }
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

JENA_CSV = os.path.join(os.path.dirname(__file__), "..", "jena_climate_2009_2016.csv")
JENA_CSV = os.path.abspath(JENA_CSV)
JENA_AVAILABLE = os.path.isfile(JENA_CSV)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Dataset")
    ds_options = ["🧪 Synthetic (sine waves)"]
    if JENA_AVAILABLE:
        ds_options.insert(0, "🌦️ Jena Climate (real-world)")
    dataset_choice = st.radio("Choose dataset", ds_options, index=0)
    use_jena = dataset_choice.startswith("🌦️")

    st.markdown("---")
    st.markdown("## ⚙️ Hyperparameters")

    if use_jena:
        window_size = st.slider("Window size (timesteps)", 24, 144, 72, step=12,
                                help="Each timestep = 10 min → 72 steps = 12 hours")
        step = st.select_slider("Sliding step", options=[1, 3, 6, 12], value=6,
                                help="Subsample factor — larger = fewer samples, faster training")
        max_rows = st.select_slider("Rows to use",
                                    options=[50_000, 100_000, 200_000, 420_551],
                                    value=100_000,
                                    format_func=lambda x: f"{x:,}")
        input_size = JENA_INPUT_SIZE
    else:
        n_samples  = st.slider("Dataset size", 128, 1024, 512, step=128)
        input_size = 32

    hidden_size  = st.slider("Hidden size (per direction)", 32, 256, 128, step=32)
    num_layers   = st.slider("BiLSTM layers", 1, 4, 3)
    dropout      = st.slider("Dropout", 0.0, 0.5, 0.3, step=0.05)
    epochs       = st.slider("Epochs", 5, 50, 20, step=5)
    lr           = st.select_slider("Learning rate",
                                    options=[1e-4, 3e-4, 1e-3, 3e-3],
                                    value=3e-4,
                                    format_func=lambda x: f"{x:.0e}")
    batch_size   = st.select_slider("Batch size", options=[16, 32, 64, 128], value=32)

    st.markdown("---")
    train_btn = st.button("🚀 Start Training", use_container_width=True, type="primary")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🏋️ Train the Model")

# Dataset info cards
if use_jena:
    st.markdown(
        f"""
        <div class="ds-card">
            <div class="ds-title">🌦️ Jena Climate Dataset  <span style="color:#3fb950;font-size:0.8rem">▸ REAL DATA</span></div>
            <div class="ds-desc">
                420,551 rows · 13 meteorological features · recorded every 10 minutes (2009–2016)<br>
                <b>Task:</b> Predict air temperature (<code>T (degC)</code>) one step ahead
                from a sliding window of past observations.<br>
                <b>Using:</b> first {max_rows:,} rows · window = {window_size} steps
                ({window_size * 10 // 60}h {window_size * 10 % 60}min context)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("📋 Feature columns (13 inputs)"):
        for col in FEATURE_COLS:
            color = "#58a6ff" if col == TARGET_COL else "#8b949e"
            st.markdown(
                f'<span style="color:{color}">{"→ " if col==TARGET_COL else "  "}'
                f'{FEATURE_LABELS.get(col, col)}</span>',
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        """
        <div class="ds-card">
            <div class="ds-title">🧪 Synthetic Dataset</div>
            <div class="ds-desc">Variable-length sine + Gaussian-noise sequences · 32 features · regression target = sequence mean</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(f"Device: `{DEVICE}` &nbsp;|&nbsp; Input size: `{input_size}` &nbsp;|&nbsp; Output: `1 (regression)`")

# ── Live training ─────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Loading dataset..."):
        if use_jena:
            train_loader, val_loader, jena_meta = load_jena_dataset(
                csv_path    = JENA_CSV,
                window_size = window_size,
                step        = step,
                val_fraction= 0.2,
                batch_size  = batch_size,
                max_rows    = max_rows,
            )
            st.info(
                f"✅ Jena Climate loaded — "
                f"**{jena_meta['n_train']:,}** train / **{jena_meta['n_val']:,}** val windows"
            )
            st.session_state["jena_meta"] = jena_meta
            st.session_state["jena_csv"]  = JENA_CSV
        else:
            dataset = SyntheticSeqDataset(n_samples=n_samples, input_size=32,
                                          min_len=10, max_len=50, seed=42)
            n_val   = int(0.2 * len(dataset))
            n_train = len(dataset) - n_val
            train_ds, val_ds = random_split(
                dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True,  collate_fn=synth_collate)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                      shuffle=False, collate_fn=synth_collate)
            st.info(f"✅ Synthetic dataset — **{n_train}** train / **{n_val}** val samples")

    model = StackedBiLSTM(
        input_size=input_size, hidden_size=hidden_size,
        num_layers=num_layers, output_size=1, dropout=dropout
    ).to(DEVICE)

    st.success(f"Model built — **{count_parameters(model):,}** trainable parameters")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    progress_bar = st.progress(0, text=f"Epoch 0 / {epochs}")
    chart_ph     = st.empty()
    metrics_ph   = st.empty()

    train_losses, val_losses, grad_norms = [], [], []
    best_val = float("inf")

    def make_chart(tl, vl):
        fig = go.Figure()
        xs  = list(range(1, len(tl) + 1))
        fig.add_trace(go.Scatter(x=xs, y=tl, mode="lines+markers",
                                  name="Train Loss",
                                  line=dict(color="#58a6ff", width=2), marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=xs, y=vl, mode="lines+markers",
                                  name="Val Loss",
                                  line=dict(color="#f78166", width=2, dash="dot"), marker=dict(size=4)))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", family="Inter"),
            xaxis=dict(title="Epoch", gridcolor="#30363d"),
            yaxis=dict(title="MSE Loss", gridcolor="#30363d"),
            legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1),
            margin=dict(l=50, r=20, t=30, b=50), height=320,
        )
        return fig

    for epoch in range(1, epochs + 1):
        model.train()
        el, eg = 0.0, 0.0
        for x_batch, lengths, targets in train_loader:
            x_batch, targets = x_batch.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x_batch, lengths)
            loss  = criterion(preds, targets)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            el += loss.item()
            eg += gnorm.item()

        avg_train = el / len(train_loader)
        avg_gnorm = eg / len(train_loader)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x_batch, lengths, targets in val_loader:
                preds = model(x_batch.to(DEVICE), lengths)
                vl += criterion(preds, targets.to(DEVICE)).item()
        avg_val    = vl / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        grad_norms.append(avg_gnorm)

        progress_bar.progress(epoch / epochs, text=f"Epoch {epoch} / {epochs}")
        chart_ph.plotly_chart(make_chart(train_losses, val_losses), use_container_width=True)
        with metrics_ph.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train MSE", f"{avg_train:.5f}")
            m2.metric("Val MSE",   f"{avg_val:.5f}")
            m3.metric("Grad Norm", f"{avg_gnorm:.4f}")
            m4.metric("LR",        f"{current_lr:.2e}")

        if avg_val < best_val:
            best_val = avg_val
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : avg_val,
                "config"     : dict(hidden_size=hidden_size, num_layers=num_layers,
                                    dropout=dropout, input_size=input_size, output_size=1),
                "dataset"    : "jena" if use_jena else "synthetic",
            }, "checkpoints/best_model.pt")

    progress_bar.progress(1.0, text="✅ Training complete!")

    st.session_state["model"]         = model
    st.session_state["model_cfg"]     = dict(hidden_size=hidden_size, num_layers=num_layers,
                                              dropout=dropout, input_size=input_size, output_size=1)
    st.session_state["train_history"] = dict(train_loss=train_losses, val_loss=val_losses,
                                              grad_norms=grad_norms)
    st.session_state["dataset_used"]  = "jena" if use_jena else "synthetic"

    best_epoch = int(np.argmin(val_losses)) + 1
    st.success(
        f"🎉 Best val MSE: **{min(val_losses):.5f}** at epoch **{best_epoch}** — "
        f"checkpoint saved to `checkpoints/best_model.pt`"
    )
    st.info("👉 Head to **🔮 Inference** to predict on real weather data!")

elif "model" in st.session_state:
    ds_used = st.session_state.get("dataset_used", "synthetic")
    st.success(f"✅ Trained model loaded from session ({'Jena Climate' if ds_used=='jena' else 'Synthetic'})")
    hist = st.session_state.get("train_history", {})
    if hist:
        fig = go.Figure()
        xs  = list(range(1, len(hist["train_loss"]) + 1))
        fig.add_trace(go.Scatter(x=xs, y=hist["train_loss"], name="Train", line=dict(color="#58a6ff", width=2)))
        fig.add_trace(go.Scatter(x=xs, y=hist["val_loss"],   name="Val",   line=dict(color="#f78166", width=2, dash="dot")))
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                          font=dict(color="#e6edf3"), height=300,
                          xaxis=dict(title="Epoch", gridcolor="#30363d"),
                          yaxis=dict(title="MSE Loss", gridcolor="#30363d"),
                          margin=dict(l=50, r=20, t=20, b=50))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown(
        """
        <div style="background:#161b22;border:1px dashed #30363d;border-radius:12px;
                    padding:3rem;text-align:center;color:#8b949e;">
            <h3 style="color:#e6edf3">👈 Choose a dataset & click Start Training</h3>
            <p>Select <b>Jena Climate</b> for real-world weather prediction, or <b>Synthetic</b> for a quick smoke test.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
