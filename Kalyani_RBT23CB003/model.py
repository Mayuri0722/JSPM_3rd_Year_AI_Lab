import json
import os
import re
from typing import List

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from preprocessing import preprocess_series, ensure_nltk_resources


DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_LOCAL_PATH = "sms.tsv"
MODEL_PATH = "spam_model.pkl"
METRICS_PATH = "metrics.json"


_lemmatizer = None
_stopwords = None


def download_dataset(url: str = DATA_URL, dest_path: str = DATA_LOCAL_PATH) -> str:
    if os.path.exists(dest_path):
        return dest_path
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    return dest_path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    # Normalize labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("clean", FunctionTransformer(preprocess_series, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs"),
            ),
        ]
    )


def train_and_evaluate(random_state: int = 42):
    ensure_nltk_resources()
    download_dataset()
    df = load_data(DATA_LOCAL_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=random_state, stratify=df["label"]
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "samples": {"train": int(len(X_train)), "test": int(len(X_test))},
        "model_path": MODEL_PATH,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    m = train_and_evaluate()
    print(json.dumps({"accuracy": m["accuracy"], "samples": m["samples"]}, indent=2))


