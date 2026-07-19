"""
Unified message analyzer — combines both models behind one function:

  1. Spam/Ham   — TF-IDF + Naive Bayes (English, our own trained model)
  2. Phishing    — pretrained BERT (ealvaradob/bert-finetuned-phishing)

Each sub-check is independent and fails gracefully: if a model file or
dependency is missing, that section of the result just reports itself as
unavailable instead of crashing the whole call.

Usage:
    from unified_detector import analyze_message
    result = analyze_message("Win a free iPhone now, click here!")
"""

import json
import pickle
import re
import string
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"


# ---------------------------------------------------------------------------
# Shared text cleaning (used by the spam/ham model only — the phishing
# model wants raw, unstemmed text)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _nltk_tools():
    import nltk
    for pkg in ["stopwords", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    return set(stopwords.words("english")), PorterStemmer()


def _clean_text_for_spam_model(text: str) -> str:
    from nltk.tokenize import word_tokenize
    stop_words, ps = _nltk_tools()
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [ps.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 1. Spam / Ham
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_spam_model():
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def _check_spam(text: str) -> dict:
    try:
        model, vectorizer = _load_spam_model()
        cleaned = _clean_text_for_spam_model(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]
        return {
            "available": True,
            "is_spam": bool(pred == 1),
            "spam_probability": round(float(proba) * 100, 1),
        }
    except FileNotFoundError:
        return {"available": False, "error": "model.pkl / vectorizer.pkl not found — run train.py first"}
    except Exception as e:
        return {"available": False, "error": str(e)}


# ---------------------------------------------------------------------------
# 2. Phishing
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_phishing_pipeline():
    from transformers import pipeline
    return pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")


@lru_cache(maxsize=1)
def _phishing_label_index():
    clf = _load_phishing_pipeline()
    id2label = clf.model.config.id2label
    for idx, name in id2label.items():
        if "phish" in str(name).lower():
            return idx, id2label
    return 1, id2label


def _check_phishing(text: str) -> dict:
    try:
        clf = _load_phishing_pipeline()
        phishing_idx, id2label = _phishing_label_index()
        phishing_label_name = id2label[phishing_idx]

        result = clf(text, top_k=None)
        if isinstance(result, dict):
            result = [result]
        if isinstance(result[0], list):
            result = result[0]

        scores = {r["label"]: r["score"] for r in result}
        phishing_prob = scores.get(phishing_label_name, max(scores.values()))
        top_label = max(scores, key=scores.get)

        return {
            "available": True,
            "is_phishing": top_label == phishing_label_name,
            "phishing_probability": round(float(phishing_prob) * 100, 1),
        }
    except ImportError:
        return {"available": False, "error": "transformers/torch not installed — pip install transformers torch"}
    except Exception as e:
        return {"available": False, "error": f"{e} (needs internet access on first run to download the model)"}


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------
def analyze_message(text: str) -> dict:
    """
    Runs spam/ham and phishing checks on one piece of text and returns a
    single combined result.

    Returns:
        {
            "text": str,
            "spam": {...},          # from the English TF-IDF/Naive Bayes model
            "phishing": {...},      # from the pretrained BERT phishing model
            "verdict": "Phishing" | "Spam" | "Ham" | "Unknown",
        }
    """
    spam_result = _check_spam(text)
    phishing_result = _check_phishing(text)

    # Simple priority for the overall verdict: phishing > spam > ham.
    # Phishing is treated as the most serious/actionable signal, so it wins
    # even if the spam model itself said "ham" (e.g. a very well-written
    # phishing email that doesn't look spammy in TF-IDF terms).
    if phishing_result.get("available") and phishing_result.get("is_phishing"):
        verdict = "Phishing"
    elif spam_result.get("available") and spam_result.get("is_spam"):
        verdict = "Spam"
    elif spam_result.get("available"):
        verdict = "Ham"
    else:
        verdict = "Unknown"

    return {
        "text": text,
        "spam": spam_result,
        "phishing": phishing_result,
        "verdict": verdict,
    }


if __name__ == "__main__":
    tests = [
        "Win a free iPhone now, click here!!!",
        "Team meeting at 4 PM, please confirm attendance",
        "Your PayPal account has been limited. Click here to verify: http://paypa1-secure.com/verify",
    ]
    for t in tests:
        result = analyze_message(t)
        print(f"\n{t}")
        print(f"  Verdict: {result['verdict']}")
        print(f"  Spam model:     {result['spam']}")
        print(f"  Phishing model: {result['phishing']}")