"""
ner.py — Extract named entities from support log text using spaCy.
Useful for spotting product names, error codes, feature mentions, etc.
For each ticket, extract noun chunks with doc.noun_chunks, then aggregate by category to find the most common phrases per topic group. The output answers a real question: "what are the most talked-about things in Hardware tickets vs Access tickets vs HR Support tickets?"
"""
import spacy
import pandas as pd
from pathlib import Path
from collections import Counter

nlp = None
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def get_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return nlp


def extract_chunks(df: pd.DataFrame) -> list:
    all_noun_chunks = []

    docs = get_nlp().pipe(df['Document'])

    for doc, row in zip(docs, df.itertuples(index=False)):
        chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks]
        all_noun_chunks.append({
            'category': row.Topic_group,  # attribute access, not dict key
            'chunks': chunks
        })

    return all_noun_chunks

def aggregate_chunks_by_category(df: pd.DataFrame, top_n: int = 10) -> dict:
    # Get chunk data 
    chunk_data = extract_chunks(df)

    # Step 1: group raw chunks by category
    category_chunks = {}
    for entry in chunk_data:
        cat = entry['category']
        if cat not in category_chunks:
            category_chunks[cat] = []
        category_chunks[cat].extend(entry['chunks'])  # flatten into one list per category
    
    # Step 2: count and return top N per category
    return {
        cat: Counter(chunks).most_common(top_n)
        for cat, chunks in category_chunks.items()
    }
def save_processed(df: pd.DataFrame, filename: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / filename, index=False)

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "service_tickets_cleaned.csv")
    
    aggregated = aggregate_chunks_by_category(df)
    
    rows = [
        {"category": cat, "chunk": chunk, "count": count}
        for cat, entries in aggregated.items()
        for chunk, count in entries
    ]
    
    result_df = pd.DataFrame(rows)
    save_processed(result_df, "aggregated_chunks.csv")