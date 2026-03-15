"""
Inference page — generate a sequence and predict with the trained model.
"""

import math
import torch
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from models.stacked_bilstm import StackedBiLSTM

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Inference · BiLSTM", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .pred-box {
        background: linear-gradient(135deg, #1c2128, #161b22);
        border: 2px solid #58a6ff; border-radius: 16px;
        padding: 2rem; text-align: center;
    }
    .pred-value { font-size: 3rem; font-weight: 700; color: #58a6ff;
                  font-family: 'JetBrains Mono', monospace; }
    .pred-label { color: #8b949e; font-size: 0.9rem; margin-top: 0.4rem; }
    .info-card  { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
                  padding: 1rem 1.2rem; margin-bottom: 0.5rem; }
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint_model(path="checkpoints/best_model.pt"):
    """Try loading a model from checkpoint."""
    try:
        ckpt = torch.load(path, map_location=DEVICE)
        cfg  = ckpt.get("config", dict(hidden_size=128, num_layers=3,
                                       dropout=0.3, input_size=32, output_size=1))
        model = StackedBiLSTM(**cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt.get("val_loss", None)
    except Exception:
        return None, None


def generate_sequence(seq_len, freq, amp, noise_std, input_size=32, seed=0):
    """Generate a single normalised sine sequence (L, F)."""
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 2 * math.pi, seq_len)
    sig = amp * np.sin(freq * t)
    noise = rng.normal(0, noise_std, size=(seq_len, input_size))
    feat  = sig[:, None] + noise
    mean  = feat.mean(axis=0, keepdims=True)
    std   = feat.std(axis=0, keepdims=True) + 1e-8
    return (feat - mean) / std, sig


# ── Load model ────────────────────────────────────────────────────────────────
st.markdown("# 🔮 Inference")
st.markdown("Configure a test sequence below, then hit **Predict** to see the model's output.")

model = st.session_state.get("model", None)
source_label = ""
if model is not None:
    source_label = "_(using model trained in this session)_"
else:
    model, best_loss = load_checkpoint_model()
    if model is not None:
        best_loss_str = f"{best_loss:.5f}" if best_loss else "unknown"
        source_label  = f"_(loaded from `checkpoints/best_model.pt`, best val loss: {best_loss_str})_"

if model is None:
    st.warning(
        "⚠️ No trained model found. Please train the model first on the **🏋️ Train** page, "
        "or ensure `checkpoints/best_model.pt` exists."
    )
    st.stop()

st.success(f"✅ Model ready {source_label}")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Sequence Config")
    st.markdown("---")
    seq_len   = st.slider("Sequence length", 10, 80, 30)
    freq      = st.slider("Signal frequency", 0.5, 3.0, 1.5, step=0.1)
    amp       = st.slider("Signal amplitude", 0.5, 2.0, 1.0, step=0.1)
    noise_std = st.slider("Noise std", 0.0, 0.5, 0.1, step=0.05)
    seed      = st.number_input("Random seed", value=7, step=1)
    st.markdown("---")
    predict_btn = st.button("⚡ Predict", use_container_width=True, type="primary")

# ── Inference ─────────────────────────────────────────────────────────────────
if predict_btn:
    feat_norm, raw_sig = generate_sequence(seq_len, freq, amp, noise_std, seed=int(seed))

    x       = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([seq_len])

    model.eval()
    with torch.no_grad():
        pred = model(x, lengths).item()

    true_mean = float(raw_sig.mean())

    # ── Layout ────────────────────────────────────────────────────────────────
    left, right = st.columns([2, 1], gap="large")

    with left:
        # Build subplot: raw signal + normalised feature heatmap
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Raw Signal (feature 0)", "Normalised Input (all 32 features)"),
            row_heights=[0.45, 0.55],
            vertical_spacing=0.14,
        )

        t_axis = list(range(seq_len))
        fig.add_trace(
            go.Scatter(x=t_axis, y=raw_sig.tolist(), mode="lines+markers",
                       name="Signal", line=dict(color="#58a6ff", width=2),
                       marker=dict(size=4)),
            row=1, col=1,
        )
        fig.add_hline(y=true_mean, line_dash="dot", line_color="#f78166",
                      annotation_text=f"True mean: {true_mean:.3f}",
                      annotation_font_color="#f78166", row=1, col=1)

        fig.add_trace(
            go.Heatmap(z=feat_norm.T.tolist(), colorscale="RdBu_r",
                       zmid=0, showscale=True,
                       colorbar=dict(len=0.4, y=0.15, thickness=10,
                                     tickfont=dict(color="#8b949e"))),
            row=2, col=1,
        )

        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", family="Inter"),
            height=430,
            showlegend=False,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        for i in [1, 2]:
            fig.update_xaxes(gridcolor="#30363d", row=i, col=1)
            fig.update_yaxes(gridcolor="#30363d", row=i, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### 🎯 Prediction")
        st.markdown(
            f'<div class="pred-box">'
            f'<div class="pred-value">{pred:.4f}</div>'
            f'<div class="pred-label">Predicted sequence mean</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        error = abs(pred - true_mean)
        rel_error = (error / (abs(true_mean) + 1e-8)) * 100

        col_a, col_b = st.columns(2)
        col_a.metric("True Mean",      f"{true_mean:.4f}")
        col_b.metric("Absolute Error", f"{error:.4f}")
        st.metric("Relative Error", f"{rel_error:.1f}%",
                  delta=f"{'good' if rel_error < 10 else 'check training'}",
                  delta_color="normal" if rel_error < 10 else "inverse")

        st.markdown("---")
        st.markdown("**Sequence Info**")
        for label, val in [
            ("Length", f"{seq_len} steps"),
            ("Frequency", f"{freq:.1f} rad/step"),
            ("Amplitude", f"{amp:.1f}"),
            ("Noise std", f"{noise_std:.2f}"),
            ("Input features", "32"),
        ]:
            st.markdown(
                f'<div class="info-card" style="padding:0.5rem 1rem;">'
                f'<span style="color:#58a6ff">{label}</span> &nbsp;'
                f'<span style="color:#e6edf3;float:right">{val}</span></div>',
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        """
        <div style="background:#161b22;border:1px dashed #30363d;border-radius:12px;
                    padding:3rem;text-align:center;color:#8b949e;">
            <h3 style="color:#e6edf3">👈 Configure a sequence & click Predict</h3>
            <p>The model will predict the mean value of your generated sequence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
