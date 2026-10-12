from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pickle
import re

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.pkl"

app = Flask(__name__)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_model():
    if not MODEL_FILE.exists():
        from train_model import train
        train()
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

model = load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if len(text) < 20:
        return jsonify({
            "success": False,
            "message": "Please enter at least 20 characters of news text."
        }), 400

    cleaned = clean_text(text)
    prediction = model["classifier"].predict([cleaned])[0]
    probabilities = model["classifier"].predict_proba([cleaned])[0]
    classes = list(model["classifier"].classes_)
    confidence = float(max(probabilities)) * 100

    label = "FAKE NEWS" if prediction == 0 else "REAL NEWS"

    return jsonify({
        "success": True,
        "label": label,
        "confidence": round(confidence, 2),
        "note": "This is an ML prediction based on the training dataset. It is not a definitive fact-check."
    })

if __name__ == "__main__":
    app.run(debug=True)
