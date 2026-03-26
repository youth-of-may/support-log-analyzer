# Support Log Analysis Pipeline

ML-powered pipeline for analyzing customer support tickets — classifies by category, extracts noun chunks with spaCy, and surfaces insights via a Streamlit dashboard.

## Stack
- **scikit-learn** — TF-IDF + Logistic Regression classifier
- **spaCy** — Noun chunk extraction on ticket text
- **Streamlit + Plotly** — Interactive analytics dashboard
- **pandas** — Data cleaning and aggregation

## Dataset
Uses the [IT Service Ticket Dataset](https://www.kaggle.com/) (~50K rows, 7 categories). Place the raw file at `data/raw/service_tickets.csv`. The dataset has two relevant columns:

```
Document, Topic_group
```

## Project structure
```
support-log-analyzer/
├── data/
│   ├── raw/                    # original CSVs
│   └── processed/              # cleaned output
├── notebooks/                  # EDA
├── src/
│   ├── ingest.py               # cleaning pipeline
│   ├── classify.py             # TF-IDF + LogReg classifier
│   ├── ner.py                  # spaCy noun chunk extraction
│   ├── insights.py             # aggregation functions
│   └── explainer.py            # classification explanation (planned)
├── app/
│   ├── dashboard.py            # overview dashboard
│   └── pages/
│       ├── 1_NLP_Explorer.py         # category drill-down
│       └── 2_Classifier.py       # live prediction + explanation
├── models/                     # saved .pkl files
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`python -m spacy download en_core_web_sm` may fail depending on your spaCy version. Install the model directly instead:
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

## Run
```bash
# 1. Clean raw data → data/processed/service_tickets_cleaned.csv
python src/ingest.py

# 2. Train classifier → models/classifier.pkl
python src/classify.py

# 3. Launch dashboard
streamlit run app/dashboard.py
```

## Pending
- **LLM explanation layer** — the classifier page has a placeholder for `explain_classification()` but the LLM-powered explanation is not yet implemented. `explainer.py` exists but returns stub output.