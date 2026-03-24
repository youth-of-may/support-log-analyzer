"""
ingest.py — Load and clean raw support log CSVs.
Expected columns: ticket_id, created_at, category, priority, description, resolution_time_hrs
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_raw(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(path, parse_dates=["created_at"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["description"])
    df["description"] = df["description"].str.strip().str.lower()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["month"] = df["created_at"].dt.to_period("M")
    df["resolution_time_hrs"] = pd.to_numeric(df["resolution_time_hrs"], errors="coerce")
    return df


def save_processed(df: pd.DataFrame, filename: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)
    print(f"Saved {len(df)} rows to {PROCESSED_DIR / filename}")


if __name__ == "__main__":
    df = load_raw("logs_sample.csv")
    df = clean(df)
    save_processed(df, "logs_clean.csv")
    print(df.head())
