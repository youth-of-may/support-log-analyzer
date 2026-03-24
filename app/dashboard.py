"""
dashboard.py — Streamlit dashboard for support log analysis.
Run: streamlit run app/dashboard.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
from insights import (
    volume_over_time,
    category_breakdown,
    avg_resolution_by_category,
    priority_distribution,
    flag_slow_tickets,
)

st.set_page_config(page_title="Support Log Analyzer", layout="wide")
st.title("Support Log Analyzer")
st.caption("ML-powered pipeline for customer support ticket analysis")

PROCESSED = Path(__file__).parent.parent / "data" / "processed" / "logs_clean.csv"


@st.cache_data
def load():
    return pd.read_csv(PROCESSED)


if not PROCESSED.exists():
    st.warning("No processed data found. Run `python src/ingest.py` first, then refresh.")
    st.stop()

df = load()

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total tickets", len(df))
col2.metric("Avg resolution (hrs)", round(df["resolution_time_hrs"].mean(), 1))
col3.metric("Categories", df["category"].nunique())
col4.metric("High priority", len(df[df["priority"] == "high"]))

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Ticket volume over time")
    vol = volume_over_time(df)
    vol["month"] = vol["month"].astype(str)
    fig = px.bar(vol, x="month", y="ticket_count", color_discrete_sequence=["#378ADD"])
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Category breakdown")
    cats = category_breakdown(df)
    fig2 = px.pie(cats, names="category", values="count", hole=0.4)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Avg resolution time by category (hrs)")
res = avg_resolution_by_category(df)
fig3 = px.bar(
    res, x="resolution_time_hrs", y="category",
    orientation="h", color_discrete_sequence=["#D85A30"]
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Slow tickets (> 24 hrs to resolve)")
slow = flag_slow_tickets(df)
st.dataframe(slow, use_container_width=True)
