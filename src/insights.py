"""
insights.py — Aggregate cleaned data into insight-ready summaries.
These feed directly into the Streamlit dashboard.
"""
import pandas as pd

# metrics

def total_tickets(df: pd.DataFrame):
    return df['Document'].count()

def unique_topics(df: pd.DataFrame):
    return df['Topic_group'].value_counts().count()

def most_common_group(df: pd.DataFrame):
    return df['Topic_group'].value_counts().reset_index()['Topic_group'].iloc[0]

def average_document_length(df: pd.DataFrame):
    return df['Document'].transform(lambda x: len(x)).mean()

# for charts
def topic_distribution(df: pd.DataFrame):
    return df['Topic_group'].value_counts().reset_index().rename(columns={"Topic_group": "Topic", "count": "Count"})

def char_length_distribution(df: pd.DataFrame):
    return df.groupby('Topic_group')['char_length'].sum().reset_index().rename(columns={"Topic_group": "Topic", "char_length": "Document Length"})

def word_length_distribution(df: pd.DataFrame):
    return df.groupby('Topic_group')['word_length'].sum().reset_index()

def longest_ticket(df: pd.DataFrame):
    return df.sort_values(by=["char_length"], ascending=False).iloc[0:4]