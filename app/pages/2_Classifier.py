"""
classifier.py — Live ticket classification and explainability page.
Run: streamlit run app/pages/classifier.py  (or via multi-page navigation)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
from explainer import explain_classification

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Support Log Analyzer", page_icon="📋", layout="wide")

# ── Dark theme CSS (matches overview + category pages) ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root palette ── */
:root {
    --bg:          #0d0f14;
    --surface:     #13161e;
    --surface-2:   #1a1e2a;
    --border:      #242836;
    --accent:      #378ADD;
    --accent-glow: rgba(55,138,221,0.18);
    --accent-dim:  rgba(55,138,221,0.08);
    --text:        #e8eaf0;
    --text-muted:  #6b7280;
    --green:       #3dd68c;
    --font-head:   'Syne', sans-serif;
    --font-mono:   'DM Mono', monospace;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}

[data-testid="stSidebar"] { background-color: var(--surface) !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Main container padding ── */
.main .block-container {
    padding: 2.5rem 3rem 3rem !important;
    max-width: 1400px;
}

/* ── Page header ── */
.page-title {
    font-family: var(--font-head);
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text);
    line-height: 1;
}
.page-title span { color: var(--accent); }
.page-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid rgba(55,138,221,0.25);
    padding: 3px 10px 4px;
    border-radius: 4px;
    margin-left: 0.75rem;
    vertical-align: middle;
}
.page-caption {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    margin-bottom: 2rem;
    letter-spacing: 0.01em;
}

/* ── Text area ── */
[data-testid="stTextArea"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stTextArea"] textarea {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    caret-color: var(--accent) !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* ── Primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background-color: var(--accent) !important;
    border: none !important;
    border-radius: 6px !important;
    color: #fff !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.5rem !important;
    transition: opacity 0.15s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.85 !important;
}

/* ── Prediction result card ── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.25rem;
}
.result-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
}
.result-value {
    font-family: var(--font-head);
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.01em;
}

/* ── Explanation block ── */
.explain-wrap {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    font-family: var(--font-mono);
    font-size: 0.83rem;
    color: var(--text);
    line-height: 1.65;
}
.explain-label {
    font-size: 0.63rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:0.25rem">
    <span class="page-title">Support Log <span>Analyzer</span></span>
    <span class="page-badge">Classifier</span>
</div>
<div class="page-caption">Run a support ticket through the ML model and inspect the classification explanation</div>
""", unsafe_allow_html=True)

# ── Data / model paths ────────────────────────────────────────────────────────
PROCESSED = Path(__file__).parent.parent / "data" / "processed" / "service_tickets_cleaned.csv"

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED)

@st.cache_resource
def load_model():
    return joblib.load(Path("models/classifier.pkl"))

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()

# ── Input + prediction ────────────────────────────────────────────────────────
user_input = st.text_area("Describe your issue")
if st.button("Predict", type="primary"):
    prediction = model.predict([user_input])[0]

    # ── Styled prediction result ──
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Category</div>
        <div class="result-value">{prediction}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Styled explanation output ──
    explanation = explain_classification(user_input, prediction)
    st.markdown(f"""
    <div class="explain-wrap">
        <div class="explain-label">Explanation</div>
        {explanation}
    </div>
    """, unsafe_allow_html=True)