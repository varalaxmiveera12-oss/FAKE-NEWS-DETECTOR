from pathlib import Path
import csv
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "news.csv"
MODEL_FILE = BASE_DIR / "model.pkl"

def train():
    texts, labels = [], []

    with open(DATA_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if text and label in {"real", "fake"}:
                texts.append(text)
                labels.append(1 if label == "real" else 0)

    if len(set(labels)) < 2:
        raise ValueError("Dataset must contain both real and fake examples.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(texts, labels)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "classifier": pipeline
        }, f)

    print(f"Model trained successfully using {len(texts)} examples.")
    print(f"Saved to: {MODEL_FILE}")

if __name__ == "__main__":
    train()
