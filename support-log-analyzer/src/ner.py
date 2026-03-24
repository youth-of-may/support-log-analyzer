"""
ner.py — Extract named entities from support log text using spaCy.
Useful for spotting product names, error codes, feature mentions, etc.
"""
import spacy
import pandas as pd
from collections import Counter

nlp = None


def get_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return nlp


def extract_entities(text: str) -> list:
    doc = get_nlp()(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


def top_entities(df: pd.DataFrame, text_col: str = "description", label_filter: str = None, top_n: int = 20):
    all_ents = []
    for text in df[text_col].dropna():
        ents = extract_entities(str(text))
        if label_filter:
            ents = [e for e in ents if e["label"] == label_filter]
        all_ents.extend([e["text"] for e in ents])
    return Counter(all_ents).most_common(top_n)


if __name__ == "__main__":
    samples = [
        "User cannot login to the billing portal",
        "Error 500 on checkout page",
        "Payment declined for Visa card",
    ]
    for s in samples:
        print(s, "→", extract_entities(s))
