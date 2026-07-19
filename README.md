# 📧 Spam & Phishing Email Detector

An end-to-end NLP + ML project that classifies email/message text as **Spam**, **Phishing**, or **Ham** (legit) — combining a self-trained classifier with a pretrained phishing-detection model, behind a single Streamlit app.

## What it does

- **Spam/Ham** — our own TF-IDF + Naive Bayes model, trained from scratch on a public SMS dataset
- **Phishing** — a pretrained BERT model, used as-is, trained specifically on phishing URLs/emails/SMS/websites
- Both run together through one function, `analyze_message()`, and the app shows a single combined verdict plus a per-model breakdown

## Results

Trained on the public SMS Spam Collection dataset (5,572 labeled messages, ~13% spam) with TF-IDF features and 3 candidate models compared:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 98.1% | 100% | 85.9% | 92.4% |
| **Naive Bayes (deployed)** | 97.1% | 100% | 78.5% | 88.0% |
| Logistic Regression | 97.0% | 100% | 77.9% | 87.5% |

**Random Forest scores highest on paper, but Naive Bayes is what's actually deployed.** The dataset is ~20 years old (2005-era UK SMS spam), and Random Forest overfits to that specific vocabulary — it fails on realistic modern spam like *"Win a free iPhone, click here"* because "iPhone" never appeared during training. Naive Bayes generalizes noticeably better to unseen phrasing in practice. See `train.py` for the sanity-check that catches this. Live numbers are in `models/metadata.json` after each training run.

The phishing model (`ealvaradob/bert-finetuned-phishing`) reports ~97% accuracy on its own eval set; we use it as-is with no fine-tuning.

## Folder structure

```
spam-detector/
├── data -> spam.csv               # dataset (label, text)
├── models -> model.pkl and tfidf_vectorizer.pkl                    # model.pkl, vectorizer.pkl, metadata.json, confusion_matrix.png
├── notebooks/spam_detector.ipynb     # full training walkthrough, already executed
├── app.py                       # Streamlit app — one input box, one Analyze button
├── train.py                     # standalone training script for the spam/ham model   
├── unified_detector.py            # combines both models behind analyze_message()
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Train the spam/ham model

Either run the notebook (`notebooks/training.ipynb`) step by step, or run:

```bash
python train.py
```

This cleans the text (lowercase → strip punctuation/numbers → tokenize → remove stopwords → stem), vectorizes it with TF-IDF, trains and compares 3 models, and saves `model.pkl` + `vectorizer.pkl` + `metadata.json` into `models/`.

## Set up the phishing model

No training needed — it's a pretrained model, used directly:

```bash
pip install transformers torch
python phishing_detector.py   # runs a few built-in test examples
```

> ⚠️ **This needs real internet access.** It downloads ~1.3GB of weights from `huggingface.co` on first run. It will **not** work in a network-restricted sandbox — run it on your own machine, Google Colab, or a Hugging Face Space. After the first download it's cached locally (`~/.cache/huggingface`) and works offline.

## Run the app

```bash
streamlit run app.py
```

Paste any email/message into the one input box and hit **Analyze**. You'll get:
- An overall verdict — **Phishing > Spam > Ham** priority (phishing is treated as the more serious signal, so it wins even if the text doesn't look "spammy")
- A two-column breakdown showing each model's individual prediction and probability
- If a model isn't set up yet, its column just shows "⚠️ Unavailable" instead of breaking the app

## How `unified_detector.py` works

```python
from unified_detector import analyze_message
result = analyze_message("Win a free iPhone now, click here!")
```

Each sub-check (`_check_spam`, `_check_phishing`) is wrapped independently — if a model file is missing or a dependency isn't installed, that part of the result reports itself as unavailable rather than crashing the whole call. This means you can use the spam model alone today and add the phishing model later without touching any calling code.

## Roadmap

- [x] Phishing detection (pretrained BERT model)
- [x] Combine both models behind one function + one UI
- [ ] Expand to a real email dataset (headers, sender, subject as extra features)
- [ ] Deploy on Hugging Face Spaces or Streamlit Community Cloud

**Dropped:** multi-language support. We prototyped it with pretrained multilingual sentence embeddings + a classifier trained on our English data (zero-shot cross-lingual transfer), but it's a weaker approach than training on labeled spam in each target language, so we cut it for now. Worth revisiting with a proper multilingual dataset later.
