"""
ingest.py — Load and clean raw support log CSVs.
Expected columns: ticket_id, created_at, category, priority, description, resolution_time_hrs
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def load_raw(file: str) -> pd.DataFrame:
    path = RAW_DIR / file
    df = pd.read_csv(path, parse_dates=["Date of Purchase", "First Response Time"])
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = ["Customer Name", "Customer Email", "Customer Gender", "Customer Age"]
    df.drop(columns=to_drop, inplace=True)
    df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')
    df['Ticket Turnover'] = (
    (df['Time to Resolution'] - df['First Response Time'])
    .dt.total_seconds() / 3600
)
    df.loc[df['Ticket Turnover'] < 0, 'Ticket Turnover'] = pd.NA
    return df

def save_processed(df: pd.DataFrame, filename: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)


if __name__ == "__main__":
    df = load_raw("customer_support_tickets.csv")
    df = clean(df)
    save_processed(df, "tickets_cleaned.csv")
    print(df.head())
    print(df.columns)
