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
    df = pd.read_csv(path)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['document_clean'] = df['Document'].fillna('')
    df['char_length'] = df['document_clean'].str.len()
    df['word_count'] = df['document_clean'].str.split().str.len()
    df.drop(columns=['document_clean'], inplace=True)
    return df

def save_processed(df: pd.DataFrame, filename: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)


if __name__ == "__main__":
    df = load_raw("it_service_tickets.csv")
    df = engineer_features(df)
    save_processed(df, "service_tickets_cleaned.csv")
