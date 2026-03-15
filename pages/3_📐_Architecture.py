"""
Architecture page — explore model structure, weight heatmaps, and explanations.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from models.stacked_bilstm import StackedBiLSTM, count_parameters

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Architecture · BiLSTM", page_icon="📐", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .explain-card {
        background: #161b22; border-left: 4px solid #58a6ff;
        border-radius: 0 10px 10px 0; padding: 1rem 1.2rem;
        margin-bottom: 0.8rem; color: #e6edf3;
    }
    .explain-title { font-weight: 600; color: #58a6ff; margin-bottom: 0.3rem; }
    .explain-body  { color: #8b949e; font-size: 0.88rem; line-height: 1.6; }
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Resolve model ─────────────────────────────────────────────────────────────
st.markdown("# 📐 Architecture Explorer")
st.markdown("Inspect the model structure, parameter count, and learned weight heatmaps.")

model = st.session_state.get("model", None)
cfg   = st.session_state.get("model_cfg", None)

if model is None:
    try:
        ckpt  = torch.load("checkpoints/best_model.pt", map_location=DEVICE)
        cfg   = ckpt.get("config", dict(hidden_size=128, num_layers=3,
                                        dropout=0.3, input_size=32, output_size=1))
        model = StackedBiLSTM(**cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        st.caption("_Loaded from `checkpoints/best_model.pt`_")
    except Exception:
        # Fallback: default model with random weights for exploration
        cfg   = dict(hidden_size=128, num_layers=3, dropout=0.3, input_size=32, output_size=1)
        model = StackedBiLSTM(**cfg).to(DEVICE)
        st.info("ℹ️ No checkpoint found — showing a randomly-initialised model for exploration.")

model.eval()

# ── Section 1: Parameter summary ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Parameter Summary")

rows = []
total = 0
for name, param in model.named_parameters():
    n = param.numel()
    total += n
    rows.append({
        "Layer": name,
        "Shape": str(list(param.shape)),
        "Parameters": f"{n:,}",
        "Trainable": "✅" if param.requires_grad else "❌",
    })

col1, col2, col3 = st.columns(3)
col1.metric("Total Trainable Params", f"{count_parameters(model):,}")
col2.metric("LSTM Layers",            cfg.get("num_layers", 3) if cfg else 3)
col3.metric("Hidden Size per Dir",    cfg.get("hidden_size", 128) if cfg else 128)

st.dataframe(rows, use_container_width=True, hide_index=True)

# ── Section 2: Weight heatmaps ────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🗺️ Weight Heatmaps")
st.markdown(
    "Use the selector to explore `weight_ih` (input→hidden) and `weight_hh` (hidden→hidden) "
    "matrices for each layer."
)

state  = model.lstm.state_dict()
wnames = sorted(state.keys())

tabs = st.tabs([n.replace("_", " ") for n in wnames])
for tab, wname in zip(tabs, wnames):
    with tab:
        w = state[wname].cpu().numpy()
        # Limit to 128 rows for performance
        w_show = w[:min(128, w.shape[0]), :]
        vmax = w.std() * 2.5
        fig = go.Figure(go.Heatmap(
            z=w_show.tolist(), colorscale="RdBu_r",
            zmid=0, zmin=-vmax, zmax=vmax,
            colorbar=dict(thickness=12, tickfont=dict(color="#8b949e")),
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", family="Inter"),
            xaxis=dict(title="Columns (features)", gridcolor="#30363d"),
            yaxis=dict(title="Rows (gate units)", gridcolor="#30363d"),
            height=340, margin=dict(l=50, r=20, t=20, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Shape: `{list(w.shape)}` &nbsp;·&nbsp; Mean: `{w.mean():.4f}` &nbsp;·&nbsp; Std: `{w.std():.4f}`")

# ── Section 3: Forget gate bias ───────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🚪 Forget Gate Bias per Layer")
st.markdown(
    "A forget gate bias initialised to **1.0** prevents early forgetting at epoch 0. "
    "Values near 1 post-training → model is still 'remembering'."
)

fig2 = go.Figure()
for layer_idx in range(model.num_layers):
    for direction, suffix in [(0, ""), (1, "_reverse")]:
        bias_key = f"bias_ih_l{layer_idx}{suffix}"
        if bias_key not in state:
            continue
        bias = state[bias_key].cpu().numpy()
        n = len(bias)
        forget_gate_bias = bias[n // 4 : n // 2]
        label = f"Layer {layer_idx+1} {'←' if direction else '→'}"
        fig2.add_trace(go.Box(
            y=forget_gate_bias.tolist(), name=label,
            marker_color=["#58a6ff", "#3fb950", "#bc8cff", "#f78166"][layer_idx % 4],
            boxmean=True,
        ))

fig2.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="Inter"),
    yaxis=dict(title="Bias value", gridcolor="#30363d"),
    showlegend=True,
    legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1),
    height=320, margin=dict(l=50, r=20, t=20, b=50),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Section 4: Concept cards ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 💡 How It Works")

concepts = [
    ("🔁 Bidirectional LSTM",
     "Two LSTM chains run in parallel — one left-to-right, one right-to-left. "
     "Their hidden states are concatenated, giving the model full context from both directions. "
     "Effective hidden size = 2 × hidden_size = 256."),
    ("🚪 Forget Gate & Cell State",
     "f_t = σ(W_f·[h_{t-1}, x_t] + b_f). When f_t ≈ 1, the cell remembers everything; "
     "when f_t ≈ 0, it forgets. The cell state flows through time with only elementwise ops, "
     "keeping gradients alive over long sequences."),
    ("✂️ Gradient Clipping",
     "After backprop, all gradients are scaled down so their L2-norm never exceeds 1.0. "
     "This prevents a single large gradient update from destabilising training."),
    ("📐 Orthogonal Init",
     "Recurrent weight matrices (W_hh) are initialised as orthogonal — preserving the L2-norm "
     "of activations during the forward pass, which greatly reduces vanishing/exploding gradients "
     "at epoch 0."),
]

for title, body in concepts:
    st.markdown(
        f'<div class="explain-card"><div class="explain-title">{title}</div>'
        f'<div class="explain-body">{body}</div></div>',
        unsafe_allow_html=True,
    )
