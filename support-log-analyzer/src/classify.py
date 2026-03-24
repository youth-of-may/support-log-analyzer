"""
classify.py — Train and run a text classifier on support log descriptions.
Uses TF-IDF + Logistic Regression. Saves model to models/.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODELS_DIR = Path(__file__).parent.parent / "models"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
    ])


def train(df: pd.DataFrame, text_col: str = "description", label_col: str = "category"):
    X = df[text_col]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "classifier.pkl")
    print("Model saved to models/classifier.pkl")
    return pipe


def load_model() -> Pipeline:
    return joblib.load(MODELS_DIR / "classifier.pkl")


def predict(texts: list) -> list:
    model = load_model()
    return model.predict(texts).tolist()


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "logs_clean.csv")
    train(df)
