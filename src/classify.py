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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODELS_DIR = Path(__file__).parent.parent / "models"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

#build pipeline, split dataset into training, testing, and verification splits, train model 
# load model, predict
def build_pipeline() -> Pipeline:
    pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')),
                     ('clf', LinearSVC(dual=False, random_state=0))])
    return pipe


def train(df: pd.DataFrame, description: str = "Ticket Description", category: str = "Ticket Type"):
    X = df[description]
    Y = df[category]
    X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    # verification set
    #Y_train, y_test, Z_train, z_test = train_test_split(Y_train, y_test, test_size=0.15, random_state=42)
    pipe = build_pipeline()
    pipe.fit(X_train, Y_train)
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
    df = pd.read_csv(PROCESSED_DIR / "tickets_cleaned.csv")
    train(df)
