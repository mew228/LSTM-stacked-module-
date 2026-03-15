"""
Inference page — predict with the trained model on real or synthetic sequences.
"""

import math
import os
import torch
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from models.stacked_bilstm import StackedBiLSTM
from data.jena_climate import (
    get_sample_window, denormalise_temp,
    FEATURE_LABELS, FEATURE_COLS, TARGET_COL,
)

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
    .pred-unit  { font-size: 1.1rem; color: #8b949e; }
    .pred-label { color: #8b949e; font-size: 0.9rem; margin-top: 0.4rem; }
    .info-chip  { display:inline-block; background:#21262d; border:1px solid #30363d;
                  border-radius:6px; padding:0.2rem 0.6rem; color:#8b949e;
                  font-size:0.78rem; margin:0.15rem; }
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JENA_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "jena_climate_2009_2016.csv")
)
JENA_AVAILABLE = os.path.isfile(JENA_CSV)

# ── Load model ────────────────────────────────────────────────────────────────
st.markdown("# 🔮 Inference")

model    = st.session_state.get("model", None)
jena_meta = st.session_state.get("jena_meta", None)
ds_used   = st.session_state.get("dataset_used", "synthetic")

if model is None:
    try:
        ckpt     = torch.load("checkpoints/best_model.pt", map_location=DEVICE)
        cfg      = ckpt.get("config", dict(hidden_size=128, num_layers=3,
                                           dropout=0.3, input_size=13, output_size=1))
        model    = StackedBiLSTM(**cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        ds_used  = ckpt.get("dataset", "synthetic")
        st.caption(f"_Loaded from `checkpoints/best_model.pt` · best val loss: {ckpt.get('val_loss', '?'):.5f}_")
    except Exception:
        st.warning(
            "⚠️ No trained model found. Please train on the **🏋️ Train** page first."
        )
        st.stop()

model.eval()
use_jena = (ds_used == "jena") and JENA_AVAILABLE and (jena_meta is not None)

st.success(
    f"✅ Model ready — "
    f"{'🌦️ Jena Climate (real weather data)' if use_jena else '🧪 Synthetic dataset'}"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if use_jena:
        st.markdown("## 🌦️ Jena Climate Sample")
        st.markdown("---")
        max_start = 400_000 - jena_meta["window_size"]
        start_idx = st.slider("Start index in dataset",
                              0, min(max_start, 400_000), 10_000, step=1000,
                              help="Row index in the CSV to start the window from")
        st.markdown(
            f"<span class='info-chip'>Window: {jena_meta['window_size']} steps</span>"
            f"<span class='info-chip'>{jena_meta['window_size'] * 10 // 60}h "
            f"{jena_meta['window_size'] * 10 % 60}min context</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## 🎛️ Sequence Config")
        st.markdown("---")
        seq_len   = st.slider("Sequence length", 10, 80, 30)
        freq      = st.slider("Signal frequency", 0.5, 3.0, 1.5, step=0.1)
        amp       = st.slider("Signal amplitude",  0.5, 2.0, 1.0, step=0.1)
        noise_std = st.slider("Noise std", 0.0, 0.5, 0.1, step=0.05)
        seed      = st.number_input("Random seed", value=7, step=1)

    st.markdown("---")
    predict_btn = st.button("⚡ Predict", use_container_width=True, type="primary")

st.markdown(
    "Use the sidebar to configure the input window, then hit **Predict**. "
    + ("The model will predict the **next air temperature (°C)**." if use_jena
       else "The model will predict the **mean value** of the generated sequence.")
)

# ── Inference ─────────────────────────────────────────────────────────────────
if predict_btn:

    if use_jena:
        # ── Real Jena Climate window ───────────────────────────────────────────
        chunk_norm, raw_temps = get_sample_window(JENA_CSV, jena_meta, start_idx)
        window_size = jena_meta["window_size"]

        x       = torch.tensor(chunk_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([window_size])

        with torch.no_grad():
            pred_norm = model(x, lengths).item()

        pred_c = denormalise_temp(pred_norm, jena_meta)
        true_c = float(raw_temps[-1])   # last temp in window ≈ current temp

        left, right = st.columns([2, 1], gap="large")

        with left:
            # Multi-panel: temperature over window + all features heatmap
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f"Temperature (°C) — {window_size} step window",
                    "All 13 normalised features (heatmap)",
                ),
                row_heights=[0.48, 0.52],
                vertical_spacing=0.14,
            )
            t_axis = list(range(window_size))
            fig.add_trace(
                go.Scatter(
                    x=t_axis, y=raw_temps.tolist(), mode="lines",
                    name="T (°C)",
                    line=dict(color="#58a6ff", width=2),
                    fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[window_size], y=[pred_c],
                    mode="markers", marker=dict(color="#f78166", size=12, symbol="star"),
                    name=f"Predicted next temp",
                ),
                row=1, col=1,
            )
            fig.add_hline(
                y=pred_c, line_dash="dot", line_color="#f78166",
                annotation_text=f"Prediction: {pred_c:.2f}°C",
                annotation_font_color="#f78166",
                row=1, col=1,
            )

            feat_labels_short = [FEATURE_LABELS.get(c, c).split("(")[0].strip()
                                  for c in jena_meta["feature_cols"]]
            fig.add_trace(
                go.Heatmap(
                    z=chunk_norm.T.tolist(),
                    y=feat_labels_short,
                    colorscale="RdBu_r", zmid=0, showscale=True,
                    colorbar=dict(len=0.4, y=0.15, thickness=10,
                                  tickfont=dict(color="#8b949e")),
                ),
                row=2, col=1,
            )
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#e6edf3", family="Inter"),
                height=460,
                legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1),
                margin=dict(l=60, r=20, t=50, b=40),
            )
            for i in [1, 2]:
                fig.update_xaxes(gridcolor="#30363d", row=i, col=1,
                                 title_text="Timestep (×10 min)" if i == 1 else "Timestep")
                fig.update_yaxes(gridcolor="#30363d", row=i, col=1)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("### 🌡️ Temperature Prediction")
            st.markdown(
                f'<div class="pred-box">'
                f'<div class="pred-value">{pred_c:.2f}<span class="pred-unit"> °C</span></div>'
                f'<div class="pred-label">Predicted next air temperature</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            delta_c = pred_c - true_c
            col_a, col_b = st.columns(2)
            col_a.metric("Window End Temp", f"{true_c:.2f} °C")
            col_b.metric("Predicted Next", f"{pred_c:.2f} °C",
                         delta=f"{delta_c:+.2f} °C",
                         delta_color="normal")

            st.markdown("---")
            st.markdown("**Window Info**")
            winfo = {
                "Start index"   : f"{start_idx:,}",
                "Window steps"  : f"{window_size}",
                "Context"       : f"{window_size * 10 // 60}h {window_size * 10 % 60}min",
                "Input features": str(len(jena_meta["feature_cols"])),
                "Target"        : "T (degC) — next step",
            }
            for k, v in winfo.items():
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
                    f'padding:0.4rem 0.9rem;margin-bottom:0.4rem;">'
                    f'<span style="color:#58a6ff">{k}</span>'
                    f'<span style="color:#e6edf3;float:right">{v}</span></div>',
                    unsafe_allow_html=True,
                )

    else:
        # ── Synthetic sequence ─────────────────────────────────────────────────
        rng   = np.random.default_rng(int(seed))
        t_ax  = np.linspace(0, 2 * math.pi, seq_len)
        sig   = amp * np.sin(freq * t_ax)
        noise = rng.normal(0, noise_std, size=(seq_len, 32))
        feat  = sig[:, None] + noise
        mean_ = feat.mean(axis=0, keepdims=True)
        std_  = feat.std(axis=0, keepdims=True) + 1e-8
        feat_norm = (feat - mean_) / std_
        true_mean = float(sig.mean())

        x       = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([seq_len])
        with torch.no_grad():
            pred = model(x, lengths).item()

        left, right = st.columns([2, 1], gap="large")
        with left:
            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("Raw Signal (feature 0)", "Normalised features (heatmap)"),
                                row_heights=[0.45, 0.55], vertical_spacing=0.14)
            fig.add_trace(go.Scatter(x=list(range(seq_len)), y=sig.tolist(), mode="lines+markers",
                                      name="Signal", line=dict(color="#58a6ff", width=2), marker=dict(size=4)),
                          row=1, col=1)
            fig.add_hline(y=true_mean, line_dash="dot", line_color="#f78166",
                          annotation_text=f"True mean: {true_mean:.3f}",
                          annotation_font_color="#f78166", row=1, col=1)
            fig.add_trace(go.Heatmap(z=feat_norm.T.tolist(), colorscale="RdBu_r",
                                      zmid=0, showscale=True,
                                      colorbar=dict(len=0.4, y=0.15, thickness=10,
                                                    tickfont=dict(color="#8b949e"))),
                          row=2, col=1)
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                              font=dict(color="#e6edf3", family="Inter"),
                              height=430, showlegend=False,
                              margin=dict(l=50, r=20, t=50, b=40))
            for i in [1, 2]:
                fig.update_xaxes(gridcolor="#30363d", row=i, col=1)
                fig.update_yaxes(gridcolor="#30363d", row=i, col=1)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("### 🎯 Prediction")
            st.markdown(f'<div class="pred-box"><div class="pred-value">{pred:.4f}</div>'
                        f'<div class="pred-label">Predicted sequence mean</div></div>',
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            err = abs(pred - true_mean)
            rel = (err / (abs(true_mean) + 1e-8)) * 100
            col_a, col_b = st.columns(2)
            col_a.metric("True Mean",      f"{true_mean:.4f}")
            col_b.metric("Absolute Error", f"{err:.4f}")
            st.metric("Relative Error", f"{rel:.1f}%")

else:
    st.markdown(
        f"""
        <div style="background:#161b22;border:1px dashed #30363d;border-radius:12px;
                    padding:3rem;text-align:center;color:#8b949e;">
            <h3 style="color:#e6edf3">👈 Configure the window & click Predict</h3>
            <p>{'The model will predict the next air temperature in °C from real weather data.' if use_jena
                else 'The model will predict the mean value of your generated sequence.'}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
