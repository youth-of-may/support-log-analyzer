# Support Log Analysis Pipeline

ML-powered pipeline for analyzing customer support tickets — classifies by category, extracts entities with spaCy, and surfaces insights via a Streamlit dashboard.

## Stack
- **scikit-learn** — TF-IDF + Logistic Regression classifier
- **spaCy** — Named entity recognition on ticket text
- **Streamlit + Plotly** — Interactive analytics dashboard
- **pandas** — Data cleaning and aggregation

## Project structure
```
support-log-analyzer/
├── data/
│   ├── raw/              # original CSVs (logs_sample.csv included)
│   └── processed/        # cleaned output
├── notebooks/            # EDA
├── src/
│   ├── ingest.py         # cleaning pipeline
│   ├── classify.py       # TF-IDF + LogReg classifier
│   ├── ner.py            # spaCy entity extraction
│   └── insights.py       # aggregation functions
├── app/
│   └── dashboard.py      # Streamlit dashboard
├── models/               # saved .pkl files
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run
```bash
# 1. Clean raw data → data/processed/logs_clean.csv
python src/ingest.py

# 2. Train classifier → models/classifier.pkl
python src/classify.py

# 3. Launch dashboard
streamlit run app/dashboard.py
```

## Data format
Place your CSV at `data/raw/logs.csv` with these columns:
```
ticket_id, created_at, category, priority, description, resolution_time_hrs
```
A 30-row sample is included at `data/raw/logs_sample.csv`.
