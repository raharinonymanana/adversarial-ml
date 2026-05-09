"""
train.py
--------
Trains a Logistic Regression spam classifier on TF-IDF features
extracted from the Enron email dataset.

Pipeline summary:
  raw text  →  TF-IDF vectorizer  →  Logistic Regression  →  prediction

Why TF-IDF + Logistic Regression?
  - Highly interpretable: every word gets a coefficient we can inspect
  - Fast on CPU with no GPU needed
  - Achieves ~95–98% accuracy on the Enron dataset
  - Logistic regression coefficients directly tell us which words are
    most "spam-like" — which we exploit in attack.py

Artifacts saved to disk:
  model.pkl         — trained LogisticRegression object
  vectorizer.pkl    — fitted TfidfVectorizer object
  top_spam_words.json — top spam/ham words + their weights (for attack.py)
  test_data.pkl     — raw test texts + true labels (for evaluate.py)

Usage:
  python train.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_CSV   = BASE_DIR / "data" / "emails.csv"
MODEL_PKL  = BASE_DIR / "model.pkl"
TFIDF_PKL  = BASE_DIR / "vectorizer.pkl"
WORDS_JSON = BASE_DIR / "top_spam_words.json"
TEST_PKL   = BASE_DIR / "test_data.pkl"

# ── TF-IDF parameters ─────────────────────────────────────────────────────────
# max_features: keep only the 15 000 most common tokens (limits RAM usage)
# min_df=2: ignore words appearing in fewer than 2 documents (noise)
# ngram_range=(1,2): use single words AND two-word phrases as features
#   e.g. "click here" is a stronger spam signal than "click" alone
# stop_words="english": remove very common words (the, is, a …) that carry
#   no discriminative power
# sublinear_tf=True: use log(1 + tf) instead of raw tf, so one word
#   appearing 100 times doesn't dominate over one appearing 10 times
TFIDF_PARAMS = dict(
    max_features=15_000,
    min_df=2,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)

# ── Logistic Regression parameters ───────────────────────────────────────────
# C=1.0: default regularisation strength (smaller C → stronger regularisation)
# max_iter=1000: lbfgs solver might need more iterations on large vocab
# solver="lbfgs": works well for dense multi-class problems; cpu-friendly
LR_PARAMS = dict(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)

# Number of top spam / ham words to export for attack.py
TOP_N = 60


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Read emails.csv and return (texts, labels) as Python lists."""
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found.\n"
            "  Run:  python data/download_enron.py   first."
        )

    print(f"[INFO] Loading {DATA_CSV} …")
    df = pd.read_csv(DATA_CSV)
    df["label"] = df["label"].map({1: "spam", 0: "ham"})

    # Belt-and-suspenders check for expected column names
    assert {"label", "text"}.issubset(df.columns), (
        "emails.csv must have 'label' and 'text' columns.  Re-run download_enron.py."
    )

    df = df.dropna(subset=["text"])          # remove rows with missing text
    df["text"] = df["text"].astype(str)      # ensure all texts are strings

    ham_n  = (df["label"] == "ham").sum()
    spam_n = (df["label"] == "spam").sum()
    print(f"  {len(df):,} emails total  |  {ham_n:,} ham  |  {spam_n:,} spam")

    return df["text"].tolist(), df["label"].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Feature importance extraction
# ══════════════════════════════════════════════════════════════════════════════

def get_top_words(vectorizer, model, n=TOP_N):
    """
    Return the words with the highest and lowest logistic regression coefficients.

    In a binary LR model trained with class 1 = spam:
      positive coefficient  → word pushes prediction toward SPAM
      negative coefficient  → word pushes prediction toward HAM

    These lists are used in attack.py to decide WHICH words to obfuscate
    or substitute.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]   # shape: (n_features,)

    # argsort returns the *indices* that would sort the array ascending
    # [-n:] gives the last n indices (= highest n values); [::-1] reverses to descending
    top_spam_idx = np.argsort(coefs)[-n:][::-1]
    top_ham_idx  = np.argsort(coefs)[:n]

    top_spam_words  = feature_names[top_spam_idx].tolist()
    top_ham_words   = feature_names[top_ham_idx].tolist()
    top_spam_scores = coefs[top_spam_idx].tolist()

    return top_spam_words, top_ham_words, top_spam_scores


# ══════════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    texts, labels = load_data()

    # ── 2. Train / test split ─────────────────────────────────────────────────
    # stratify=labels ensures the spam/ham ratio is the same in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"\n[INFO] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. TF-IDF vectorisation ───────────────────────────────────────────────
    # fit_transform() on training data: LEARN the vocabulary, THEN encode
    # transform() on test data: encode using the ALREADY LEARNED vocabulary
    # (We never let the model see test data during training — data leakage prevention)
    print("[INFO] Fitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,} tokens")

    # ── 4. Train Logistic Regression ──────────────────────────────────────────
    # LR learns a weight (coefficient) for every token in the vocabulary.
    # spam_probability = sigmoid( w₀ + w₁·tfidf₁ + w₂·tfidf₂ + … )
    print("[INFO] Training Logistic Regression …")
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train_vec, y_train)
    print("  Training complete.")

    # ── 5. Evaluate on the held-out test set ─────────────────────────────────
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {accuracy*100:.2f}%")
    print(f"{'='*50}")

    if accuracy < 0.90:
        print("[WARN] Accuracy below 90% — check dataset quality or tune hyperparameters.")

    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"]))

    # ── 6. Extract top spam / ham words ───────────────────────────────────────
    top_spam_words, top_ham_words, top_spam_scores = get_top_words(vectorizer, model)

    print("[INFO] Top 15 spam-indicative words (by LR coefficient):")
    for word, score in zip(top_spam_words[:15], top_spam_scores[:15]):
        bar = "█" * min(int(score * 2), 20)   # visual bar for the coefficient
        print(f"  {word:<25} {score:+.3f}  {bar}")

    # ── 7. Save all artifacts ─────────────────────────────────────────────────
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[INFO] Model saved      → {MODEL_PKL}")

    with open(TFIDF_PKL, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Vectorizer saved → {TFIDF_PKL}")

    # Save the word lists as plain JSON so attack.py can read them without
    # unpickling the full model (faster, and avoids pickle security concerns)
    words_data = {
        "top_spam_words":  top_spam_words,
        "top_ham_words":   top_ham_words,
        "top_spam_scores": top_spam_scores,
    }
    with open(WORDS_JSON, "w", encoding="utf-8") as f:
        json.dump(words_data, f, indent=2)
    print(f"[INFO] Top words saved  → {WORDS_JSON}")

    # Save raw test texts + true labels so evaluate.py uses the EXACT same
    # test set without re-loading or re-splitting the CSV
    test_data = {"X_test": X_test, "y_test": y_test}
    with open(TEST_PKL, "wb") as f:
        pickle.dump(test_data, f)
    print(f"[INFO] Test data saved  → {TEST_PKL}")

    print("\n[DONE] Next step: python evaluate.py")


if __name__ == "__main__":
    main()
