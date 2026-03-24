"""
insights.py — Aggregate cleaned data into insight-ready summaries.
These feed directly into the Streamlit dashboard.
"""
import pandas as pd


def volume_over_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("month").size().reset_index(name="ticket_count")


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    return counts


def avg_resolution_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category")["resolution_time_hrs"]
        .mean()
        .round(1)
        .reset_index()
        .sort_values("resolution_time_hrs", ascending=False)
    )


def priority_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["priority"].value_counts().reset_index()
    counts.columns = ["priority", "count"]
    return counts


def flag_slow_tickets(df: pd.DataFrame, threshold_hrs: float = 24.0) -> pd.DataFrame:
    return (
        df[df["resolution_time_hrs"] > threshold_hrs][
            ["ticket_id", "category", "priority", "resolution_time_hrs", "description"]
        ]
        .sort_values("resolution_time_hrs", ascending=False)
    )
