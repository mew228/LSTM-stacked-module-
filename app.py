"""
app.py — Home page of the Stacked BiLSTM Streamlit app.
Run with:  streamlit run app.py
"""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stacked BiLSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #bc8cff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #8b949e;
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto;
    }
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .stat-card:hover { border-color: #58a6ff; }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-label { color: #8b949e; font-size: 0.85rem; margin-top: 0.25rem; }
    .arch-block {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #e6edf3;
        line-height: 1.7;
    }
    .feature-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .feature-title { color: #58a6ff; font-weight: 600; font-size: 0.95rem; }
    .feature-desc  { color: #8b949e; font-size: 0.85rem; margin-top: 0.2rem; }
    .nav-hint {
        background: linear-gradient(90deg, #1c2128, #161b22);
        border: 1px solid #3fb950;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #3fb950;
        font-size: 0.9rem;
        text-align: center;
    }
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #30363d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 BiLSTM Explorer")
    st.markdown("---")
    st.markdown("**Navigate:**")
    st.markdown("🏋️ **Train** — Configure & train the model live")
    st.markdown("🔮 **Inference** — Predict on a custom sequence")
    st.markdown("📐 **Architecture** — Explore weights & structure")
    st.markdown("---")
    st.markdown(
        "<span style='color:#8b949e;font-size:0.8rem'>Built with PyTorch 2.x · Streamlit</span>",
        unsafe_allow_html=True,
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🧠 Stacked BiLSTM</h1>
        <p>An interactive explorer for a Stacked Bidirectional LSTM — train it live,
        tune hyperparameters, and inspect learned weights, all in your browser.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Stats ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
stats = [
    (c1, "~1.2M",  "Trainable Parameters"),
    (c2, "3",      "BiLSTM Layers"),
    (c3, "256",    "Hidden Dim (2×128)"),
    (c4, "AdamW",  "Optimizer"),
]
for col, val, label in stats:
    with col:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{val}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Architecture + Features ───────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown("### 🏗️ Architecture")
    st.markdown(
        """
        <div class="arch-block">
Input (32 features / step)<br>
&nbsp;&nbsp;&nbsp;&nbsp;│<br>
&nbsp;&nbsp;&nbsp;&nbsp;▼<br>
┌──────────────────────────┐<br>
│  BiLSTM Layer 1          │  ← fwd + bwd, 256-dim<br>
└───────────┬──────────────┘<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  dropout 0.3<br>
┌───────────▼──────────────┐<br>
│  BiLSTM Layer 2          │<br>
└───────────┬──────────────┘<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  dropout 0.3<br>
┌───────────▼──────────────┐<br>
│  BiLSTM Layer 3          │<br>
└───────────┬──────────────┘<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  concat fwd+bwd → 256-dim<br>
┌───────────▼──────────────┐<br>
│  FC: 256→128→ReLU→1      │  ← single output<br>
└──────────────────────────┘
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("### ⚡ Key Techniques")
    features = [
        ("🔁 Bidirectional", "Reads sequences forwards AND backwards for full context"),
        ("🧱 3 Stacked Layers", "Each layer learns increasingly abstract temporal patterns"),
        ("🚪 Forget Gate bias=1", "Initialised to remember everything — prevents early forgetting"),
        ("📐 Orthogonal Init", "Recurrent weights initialised for stable gradient flow"),
        ("📉 CosineAnnealingLR", "Learning rate decays smoothly from 3e-4 → ~0"),
        ("✂️ Gradient Clipping", "Clips norm to 1.0 — prevents exploding gradients"),
        ("⚖️ AdamW", "Decoupled weight decay for better regularisation"),
        ("📦 PackedSequence", "Variable-length batches with no wasted computation"),
    ]
    for title, desc in features:
        st.markdown(
            f'<div class="feature-card"><div class="feature-title">{title}</div>'
            f'<div class="feature-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

# ── Navigation hint ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div class="nav-hint">👈 Use the sidebar to navigate · '
    'Start with <strong>🏋️ Train</strong> to train the model, '
    'then try <strong>🔮 Inference</strong></div>',
    unsafe_allow_html=True,
)
