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
    all_noun_chunks = []  # fix: list, not dict
    
    docs = get_nlp().pipe(df['Document'])  # lazy generator over all texts
    
    for doc, (_, row) in zip(docs, df.iterrows()):  # zip to retain category
        chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks]
        all_noun_chunks.append({
            'category': row['Topic_group'],  # now correctly paired
            'chunks': chunks
        })
    
    return all_noun_chunks

def aggregate_chunks_by_category(chunk_data: list, top_n: int = 10) -> dict:
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
        


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "service_tickets.csv")
    
    chunk_data = extract_chunks(df.iloc[0:100])
    print(chunk_data)
    aggregated = aggregate_chunks_by_category(chunk_data, top_n=5)
    print(aggregated)
    for category, top_chunks in aggregated.items():
        print(f"\n=== {category} ===")
        for phrase, count in top_chunks:
            print(f"  {phrase:30s}  {count}")
