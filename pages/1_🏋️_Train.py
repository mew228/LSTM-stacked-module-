"""
Train page — configure hyperparameters and run live training.
"""

import time
import torch
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from torch.utils.data import DataLoader, random_split

from models.stacked_bilstm import StackedBiLSTM, count_parameters
from data.dataset import SyntheticSeqDataset, collate_fn
from training.config import TrainConfig

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Train · BiLSTM", page_icon="🏋️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-row {
        display: flex; gap: 1rem; margin-bottom: 1rem;
    }
    .epoch-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px;
        padding: 0.8rem 1.2rem; text-align: center;
    }
    .epoch-val { font-size: 1.6rem; font-weight: 700; color: #58a6ff;
                 font-family: 'JetBrains Mono', monospace; }
    .epoch-lbl { color: #8b949e; font-size: 0.78rem; }
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Hyperparameters")
    st.markdown("---")

    hidden_size  = st.slider("Hidden size (per direction)", 32, 256, 128, step=32)
    num_layers   = st.slider("BiLSTM layers", 1, 4, 3)
    dropout      = st.slider("Dropout", 0.0, 0.5, 0.3, step=0.05)
    epochs       = st.slider("Epochs", 5, 60, 30, step=5)
    lr           = st.select_slider("Learning rate",
                                    options=[1e-4, 3e-4, 1e-3, 3e-3],
                                    value=3e-4,
                                    format_func=lambda x: f"{x:.0e}")
    batch_size   = st.select_slider("Batch size", options=[8, 16, 32, 64], value=32)
    n_samples    = st.slider("Dataset size", 128, 1024, 512, step=128)

    st.markdown("---")
    train_btn = st.button("🚀 Start Training", use_container_width=True, type="primary")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🏋️ Train the Model")
st.markdown(
    "Adjust hyperparameters in the sidebar, then hit **Start Training**. "
    "The loss chart updates live after each epoch."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.caption(f"Device: `{DEVICE}` &nbsp;|&nbsp; Input size: `32 features` &nbsp;|&nbsp; Output size: `1`")

# ── Live training ─────────────────────────────────────────────────────────────
if train_btn:
    # Build dataset
    dataset = SyntheticSeqDataset(
        n_samples=n_samples, input_size=32, min_len=10, max_len=50, seed=42
    )
    n_val   = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Build model
    model = StackedBiLSTM(
        input_size=32, hidden_size=hidden_size,
        num_layers=num_layers, output_size=1, dropout=dropout
    ).to(DEVICE)

    n_params = count_parameters(model)
    st.info(f"Model built — **{n_params:,}** trainable parameters")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Live UI elements
    progress_bar  = st.progress(0, text="Epoch 0 / " + str(epochs))
    chart_ph      = st.empty()
    metrics_ph    = st.empty()

    train_losses, val_losses, grad_norms = [], [], []

    def make_chart(train_losses, val_losses):
        fig = go.Figure()
        xs = list(range(1, len(train_losses) + 1))
        fig.add_trace(go.Scatter(x=xs, y=train_losses, mode="lines+markers",
                                  name="Train Loss", line=dict(color="#58a6ff", width=2),
                                  marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=xs, y=val_losses, mode="lines+markers",
                                  name="Val Loss", line=dict(color="#f78166", width=2, dash="dot"),
                                  marker=dict(size=5)))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", family="Inter"),
            xaxis=dict(title="Epoch", gridcolor="#30363d", showgrid=True),
            yaxis=dict(title="MSE Loss", gridcolor="#30363d", showgrid=True),
            legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1),
            margin=dict(l=50, r=20, t=30, b=50),
            height=320,
        )
        return fig

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss, epoch_gnorm = 0.0, 0.0
        for x_batch, lengths, targets in train_loader:
            x_batch, targets = x_batch.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x_batch, lengths)
            loss  = criterion(preds, targets)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss  += loss.item()
            epoch_gnorm += gnorm.item()

        avg_train = epoch_loss / len(train_loader)
        avg_gnorm = epoch_gnorm / len(train_loader)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, lengths, targets in val_loader:
                preds = model(x_batch.to(DEVICE), lengths)
                val_loss += criterion(preds, targets.to(DEVICE)).item()
        avg_val = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        grad_norms.append(avg_gnorm)

        # Update UI
        progress_bar.progress(epoch / epochs, text=f"Epoch {epoch} / {epochs}")
        chart_ph.plotly_chart(make_chart(train_losses, val_losses), use_container_width=True)

        with metrics_ph.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train Loss", f"{avg_train:.5f}")
            m2.metric("Val Loss",   f"{avg_val:.5f}")
            m3.metric("Grad Norm",  f"{avg_gnorm:.4f}")
            m4.metric("LR",         f"{current_lr:.2e}")

    progress_bar.progress(1.0, text="✅ Training complete!")

    # Save model to session state
    st.session_state["model"]       = model
    st.session_state["model_cfg"]   = dict(
        hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, input_size=32, output_size=1
    )
    st.session_state["train_history"] = dict(
        train_loss=train_losses, val_loss=val_losses, grad_norms=grad_norms
    )

    # Save checkpoint
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "epoch": epochs,
        "model_state": model.state_dict(),
        "val_loss": min(val_losses),
        "config": st.session_state["model_cfg"],
    }, "checkpoints/best_model.pt")

    # Summary
    best_epoch = int(np.argmin(val_losses)) + 1
    st.success(
        f"🎉 Training done!  Best val loss **{min(val_losses):.5f}** at epoch **{best_epoch}**.  "
        f"Checkpoint saved → `checkpoints/best_model.pt`"
    )
    st.info("👉 Head to **🔮 Inference** in the sidebar to run predictions!")

elif "model" in st.session_state:
    st.success("✅ A trained model is already loaded in session — head to 🔮 Inference!")
    hist = st.session_state.get("train_history", {})
    if hist:
        train_losses = hist["train_loss"]
        val_losses   = hist["val_loss"]
        fig = go.Figure()
        xs  = list(range(1, len(train_losses) + 1))
        fig.add_trace(go.Scatter(x=xs, y=train_losses, name="Train", line=dict(color="#58a6ff", width=2)))
        fig.add_trace(go.Scatter(x=xs, y=val_losses,   name="Val",   line=dict(color="#f78166", width=2, dash="dot")))
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
            <h3 style="color:#e6edf3">👈 Set hyperparameters & click Start Training</h3>
            <p>The loss chart will update live as the model trains.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
