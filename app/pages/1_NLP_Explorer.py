"""
category.py — Category drill-down page for support log analysis.
Run: streamlit run app/pages/category.py  (or via multi-page navigation)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Support Log Analyzer", page_icon="📋", layout="wide")

# ── Dark theme CSS (matches overview dashboard) ───────────────────────────────
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
    --amber:       #f59e0b;
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
.page-header {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    margin-bottom: 0.25rem;
}
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
    margin-bottom: 6px;
}
.page-caption {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    margin-bottom: 2rem;
    letter-spacing: 0.01em;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stSelectbox"] > div > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* ── Section labels ── */
h3, .stSubheader {
    font-family: var(--font-head) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    margin-bottom: 0.75rem !important;
}

/* ── Chart containers ── */
[data-testid="stPlotlyChart"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
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
<div class="page-header">
    <div class="page-title">Support Log <span>Analyzer</span></div>
    <div class="page-badge">Category View</div>
</div>
<div class="page-caption">Drill down into chunk distributions and sample tickets by topic group</div>
""", unsafe_allow_html=True)

# ── Shared Plotly layout ──────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=16, r=16, t=16, b=16),
    xaxis=dict(
        gridcolor="#242836", linecolor="#242836",
        tickfont=dict(color="#6b7280", size=10)
    ),
    yaxis=dict(
        gridcolor="#242836", linecolor="#242836",
        tickfont=dict(color="#6b7280", size=10)
    ),
)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA  = Path(__file__).parent.parent.parent / "data" / "processed" / "service_tickets_cleaned.csv"
CHUNK = Path(__file__).parent.parent.parent / "data" / "processed" / "aggregated_chunks.csv"

# ── Loaders (cached) ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA)

if not DATA.exists():
    st.warning("No processed data found. Run `python src/ingest.py` first, then refresh.")
    st.stop()

@st.cache_data
def load_chunks():
    return pd.read_csv(CHUNK)

if not CHUNK.exists():
    st.warning("No processed data found. Run `python src/ingest.py` first, then refresh.")
    st.stop()

# ── Load dataframes ───────────────────────────────────────────────────────────
df     = load_data()
chunks = load_chunks().rename(columns={"chunk": "Chunk", "count": "Count"})

# ── Category selector ─────────────────────────────────────────────────────────
category    = st.selectbox('Select Category', options=df['Topic_group'].unique(), index=0)
filtered_df = chunks[chunks["category"] == category][["Chunk", "Count"]]

# ── Chunk distribution bar chart ──────────────────────────────────────────────
st.subheader("Chunk Distribution")
fig = px.bar(
    filtered_df, x="Chunk", y="Count",
    color_discrete_sequence=["#378ADD"]
)
fig.update_traces(marker_line_width=0, opacity=0.9)
fig.update_layout(**CHART_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

# ── Sample tickets table ──────────────────────────────────────────────────────
st.subheader("Sample Tickets")
tickets = df.loc[df["Topic_group"] == category].iloc[0:9]
columns = ["Category", "Document", "Character Length"]

fig = go.Figure(data=[go.Table(
    columnwidth=[90, 420, 90],
    header=dict(
        values=[f"<b>{c}</b>" for c in columns],
        align="left",
        fill_color="#1a1e2a",
        font=dict(family="DM Mono, monospace", size=11, color="#6b7280"),
        line_color="#242836",
        height=36,
    ),
    cells=dict(
        values=[tickets.Topic_group, tickets.Document, tickets.char_length],
        align="left",
        fill_color=["#13161e", "#13161e", "#13161e"],
        font=dict(family="DM Mono, monospace", size=11, color="#e8eaf0"),
        line_color="#242836",
        height=32,
    )
)])
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
)
st.plotly_chart(fig, use_container_width=True)