import os
import re
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd


_lemmatizer: Optional[WordNetLemmatizer] = None
_stopwords: Optional[set[str]] = None


def ensure_nltk_resources() -> None:
    global _lemmatizer, _stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    if _stopwords is None:
        _stopwords = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    ensure_nltk_resources()
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 1]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_series(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(preprocess_text)


