import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """Preprocesses raw ticket text by lowercasing, stripping template placeholders,
    removing punctuation and normalizing whitespace."""
    text = str(text).lower()
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_save_all_models(data_path: str = None):
    """Loads historical dataset, trains multiclass Category model and separate Urgency model,
    evaluates both with F1, precision, recall, and confusion matrix, and saves artifacts."""
    if data_path is None:
        data_path = os.path.join(BASE_DIR, "ticket.zip")
        if not os.path.exists(data_path):
            data_path = os.path.join(BASE_DIR, "customer_support_tickets.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading dataset from: {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Prepare raw text and targets
    df["ticket_text"] = df["Ticket Subject"].fillna("") + " " + df["Ticket Description"].fillna("")
    df["clean_text"] = df["ticket_text"].apply(clean_text)
    df["category"] = df["Ticket Type"]
    df["urgency"] = df["Ticket Priority"]

    X = df["clean_text"]
    y_cat = df["category"]
    y_urg = df["urgency"]

    # 2. Train / Test Split (Stratified on category)
    X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
        X, y_cat, y_urg, test_size=0.20, random_state=42, stratify=y_cat
    )

    # 3. TF-IDF Feature Extraction
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 4. Train Multiclass Category Model
    print("Training Category Classifier (Multiclass)...")
    cat_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    cat_model.fit(X_train_tfidf, y_cat_train)
    cat_pred = cat_model.predict(X_test_tfidf)

    # 5. Train Separate Urgency Model
    print("Training Urgency Classifier (Multiclass)...")
    urg_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    urg_model.fit(X_train_tfidf, y_urg_train)
    urg_pred = urg_model.predict(X_test_tfidf)

    # 6. Evaluation - Category Model
    cat_report_dict = classification_report(y_cat_test, cat_pred, output_dict=True, zero_division=0)
    cat_cm = confusion_matrix(y_cat_test, cat_pred, labels=cat_model.classes_)

    cat_metrics = {
        "accuracy": float(accuracy_score(y_cat_test, cat_pred)),
        "macro_f1": float(f1_score(y_cat_test, cat_pred, average="macro")),
        "weighted_f1": float(f1_score(y_cat_test, cat_pred, average="weighted")),
        "classes": list(cat_model.classes_),
        "classification_report": cat_report_dict,
        "confusion_matrix": cat_cm.tolist()
    }

    # 7. Evaluation - Urgency Model
    urg_report_dict = classification_report(y_urg_test, urg_pred, output_dict=True, zero_division=0)
    urg_cm = confusion_matrix(y_urg_test, urg_pred, labels=urg_model.classes_)

    urg_metrics = {
        "accuracy": float(accuracy_score(y_urg_test, urg_pred)),
        "macro_f1": float(f1_score(y_urg_test, urg_pred, average="macro")),
        "weighted_f1": float(f1_score(y_urg_test, urg_pred, average="weighted")),
        "classes": list(urg_model.classes_),
        "classification_report": urg_report_dict,
        "confusion_matrix": urg_cm.tolist()
    }

    all_metrics = {
        "dataset_total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "category_model": cat_metrics,
        "urgency_model": urg_metrics
    }

    # 8. Save Models and Metrics
    print("Saving models and metrics to disk...")
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(cat_model, os.path.join(MODELS_DIR, "category_model.joblib"))
    joblib.dump(urg_model, os.path.join(MODELS_DIR, "urgency_model.joblib"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("Model training and evaluation successfully completed!")
    return all_metrics


class TicketClassifier:
    """Inference engine providing category and urgency predictions with confidence scoring
    and automated flagging for human review."""

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.tfidf = None
        self.cat_model = None
        self.urg_model = None
        self.metrics = None
        self.load_models()

    def load_models(self):
        tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
        cat_path = os.path.join(MODELS_DIR, "category_model.joblib")
        urg_path = os.path.join(MODELS_DIR, "urgency_model.joblib")
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")

        if not (os.path.exists(tfidf_path) and os.path.exists(cat_path) and os.path.exists(urg_path)):
            print("Models not found on disk. Training now...")
            train_and_save_all_models()

        self.tfidf = joblib.load(tfidf_path)
        self.cat_model = joblib.load(cat_path)
        self.urg_model = joblib.load(urg_path)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    def predict(self, text: str, subject: str = "", threshold: float = None):
        if threshold is None:
            threshold = self.confidence_threshold

        full_raw = f"{subject} {text}".strip()
        cleaned = clean_text(full_raw)

        if not cleaned:
            return {
                "error": "Empty ticket text provided",
                "clean_text": "",
                "category": {"prediction": "Unknown", "confidence": 0.0, "all_probabilities": {}},
                "urgency": {"prediction": "Medium", "confidence": 0.0, "all_probabilities": {}},
                "overall_confidence": 0.0,
                "needs_human_review": True,
                "review_reason": "Empty ticket text"
            }

        vec = self.tfidf.transform([cleaned])

        # Category prediction and probabilities
        cat_probs = self.cat_model.predict_proba(vec)[0]
        cat_idx = np.argmax(cat_probs)
        cat_pred = self.cat_model.classes_[cat_idx]
        cat_conf = float(cat_probs[cat_idx])
        cat_all = {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.cat_model.classes_, cat_probs)}

        # Urgency prediction and probabilities
        urg_probs = self.urg_model.predict_proba(vec)[0]
        urg_idx = np.argmax(urg_probs)
        urg_pred = self.urg_model.classes_[urg_idx]
        urg_conf = float(urg_probs[urg_idx])
        urg_all = {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.urg_model.classes_, urg_probs)}

        # Overall confidence is the minimum confidence between category and urgency
        overall_conf = min(cat_conf, urg_conf)
        needs_review = bool(overall_conf < threshold)

        reasons = []
        if cat_conf < threshold:
            reasons.append(f"Low category confidence ({cat_conf:.1%})")
        if urg_conf < threshold:
            reasons.append(f"Low urgency confidence ({urg_conf:.1%})")

        return {
            "raw_text": full_raw,
            "clean_text": cleaned,
            "category": {
                "prediction": cat_pred,
                "confidence": round(cat_conf, 4),
                "all_probabilities": cat_all
            },
            "urgency": {
                "prediction": urg_pred,
                "confidence": round(urg_conf, 4),
                "all_probabilities": urg_all
            },
            "overall_confidence": round(overall_conf, 4),
            "confidence_threshold": threshold,
            "needs_human_review": needs_review,
            "review_reason": "; ".join(reasons) if reasons else None
        }


if __name__ == "__main__":
    train_and_save_all_models()
