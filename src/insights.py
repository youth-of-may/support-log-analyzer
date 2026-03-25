"""
insights.py — Aggregate cleaned data into insight-ready summaries.
These feed directly into the Streamlit dashboard.
"""
import pandas as pd

# metrics

def total_tickets(df: pd.DataFrame):
    return df['Document'].count()

def unique_topics(df: pd.DataFrame):
    return df['Topic_groups'].value_counts().count()

def most_common_group(df: pd.DataFrame):
    return df['Topic_groups'].value_counts().iloc[0]

def average_document_length(df: pd.DataFrame):
    return df['Document'].transform(lambda x: len(x)).mean()

def topic_distribution(df: pd.DataFrame):
    return df['Topic_groups'].value_counts().reset_index()

def char_length_distribution(df: pd.DataFrame):
    return df.groupby('Topic_group')['char_length'].sum().reset_index()

def word_length_distribution(df: pd.DataFrame):
    return df.groupby('Topic_group')['word_length'].sum().reset_index()

