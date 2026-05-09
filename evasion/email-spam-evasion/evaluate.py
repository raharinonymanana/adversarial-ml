"""
evaluate.py
-----------
Evaluates all three evasion attacks and prints a comparison table.

Methodology:
  1. Load the trained model, TF-IDF vectorizer, and saved test set.
  2. Run the model on the test set WITHOUT any attack → baseline.
  3. Isolate the spam emails that the model CORRECTLY detects.
     (We only test attacks on correctly-detected spam — otherwise we would
     be counting emails the model already misses, which inflates results.)
  4. Apply each attack to those emails and re-classify them.
  5. Count how many slip through (predicted ham after attack).

Metrics printed per attack:
  Original Detection Rate  — how many spam emails were detected before attack
  Post-Attack Detection %  — how many are still detected after attack
  Evasion Rate             — % of spam that bypassed the filter (higher = stronger attack)

Usage:
  python evaluate.py
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

# Import attack infrastructure from attack.py (same directory)
from attack import ATTACKS, ATTACK_NAMES, load_top_spam_words

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
MODEL_PKL = BASE_DIR / "model.pkl"
TFIDF_PKL = BASE_DIR / "vectorizer.pkl"
TEST_PKL  = BASE_DIR / "test_data.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_artifacts():
    """Load and return (model, vectorizer, test_data_dict)."""
    for path in (MODEL_PKL, TFIDF_PKL, TEST_PKL):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.\n"
                "  Run:  python train.py   first."
            )

    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    with open(TFIDF_PKL, "rb") as f:
        vectorizer = pickle.load(f)
    with open(TEST_PKL, "rb") as f:
        test_data = pickle.load(f)

    return model, vectorizer, test_data


def _predict(model, vectorizer, texts: list) -> np.ndarray:
    """
    Vectorise raw text strings with the TF-IDF vectorizer and return
    the model's predictions ("ham" or "spam").
    """
    X = vectorizer.transform(texts)
    return model.predict(X)


# ══════════════════════════════════════════════════════════════════════════════
# Per-attack evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _run_attack(attack_fn, spam_texts: list, model, vectorizer, top_spam_words):
    """
    Apply one attack function to the list of spam emails and measure how
    many slip past the classifier.

    Args:
        attack_fn       — one of the three functions from attack.py
        spam_texts      — spam emails that are CURRENTLY correctly detected
        model           — trained LogisticRegression
        vectorizer      — fitted TfidfVectorizer
        top_spam_words  — top spam words list (some attacks use it)

    Returns a dict:
        n_tested                 — number of spam emails tested
        n_evaded                 — number now classified as ham (evasion success)
        evasion_rate             — n_evaded / n_tested
        post_attack_detection    — 1 - evasion_rate (still caught as spam)
    """
    # Step 1: apply the attack → get modified texts
    attacked_texts = attack_fn(spam_texts, top_spam_words)

    # Step 2: classify the modified texts
    preds = _predict(model, vectorizer, attacked_texts)

    # Step 3: count successes (prediction flipped from spam → ham)
    n_tested = len(spam_texts)
    n_evaded = int((preds == "ham").sum())   # predicted "ham" → attack succeeded

    evasion_rate          = n_evaded / n_tested if n_tested > 0 else 0.0
    post_attack_detection = 1.0 - evasion_rate

    return {
        "n_tested":              n_tested,
        "n_evaded":              n_evaded,
        "evasion_rate":          evasion_rate,
        "post_attack_detection": post_attack_detection,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════════

def _print_table(results: dict, original_detection_rate: float, n_detected: int):
    """Render a clean ASCII comparison table."""

    # Column widths
    W_NAME    = 30
    W_TESTED  = 10
    W_EVADED  = 10
    W_POST    = 16
    W_EVASION = 12

    sep_len  = W_NAME + W_TESTED + W_EVADED + W_POST + W_EVASION + 5   # 5 spaces between
    sep_thin = "-" * sep_len
    sep_thick = "=" * sep_len

    print()
    print(sep_thick)
    print("  ADVERSARIAL EVASION RESULTS — Spam Email Classifier")
    print(sep_thick)
    print(
        f"\n  Baseline spam detection rate : "
        f"{original_detection_rate*100:.1f}%  "
        f"({n_detected} emails correctly flagged before any attack)\n"
    )

    # Header row
    hdr = (
        f"{'Attack':<{W_NAME}}"
        f"{'Tested':>{W_TESTED}}"
        f"{'Evaded':>{W_EVADED}}"
        f"{'Post-Atk Det%':>{W_POST}}"
        f"{'Evasion%':>{W_EVASION}}"
    )
    print(hdr)
    print(sep_thin)

    for key, m in results.items():
        name = ATTACK_NAMES[key]

        # Add a ▲ or ▼ hint to make the numbers more readable at a glance
        # Evasion% high → attack is effective → show ▲ (threat is up)
        evasion_pct = m["evasion_rate"] * 100
        flag = "▲" if evasion_pct >= 50 else ("►" if evasion_pct >= 20 else "▼")

        row = (
            f"{name:<{W_NAME}}"
            f"{m['n_tested']:>{W_TESTED}}"
            f"{m['n_evaded']:>{W_EVADED}}"
            f"{m['post_attack_detection']*100:>{W_POST-1}.1f}%"
            f"  {flag} {evasion_pct:>{W_EVASION-4}.1f}%"
        )
        print(row)

    print(sep_thin)

    # Legend
    print()
    print("  Legend")
    print("  ──────")
    print("  Tested        — spam emails correctly detected BEFORE the attack")
    print("  Evaded        — emails that slipped past the filter AFTER the attack")
    print("  Post-Atk Det% — spam detection rate after the attack  (↓ = attack works)")
    print("  Evasion%      — share of spam that bypassed the filter (↑ = attack works)")
    print("  ▲ ≥50%  ►20–49%  ▼<20%  (rough effectiveness indicator)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load saved artifacts ───────────────────────────────────────────────
    print("[INFO] Loading model, vectorizer, and test data …")
    model, vectorizer, test_data = _load_artifacts()
    top_spam_words = load_top_spam_words()

    X_test = test_data["X_test"]              # raw email strings
    y_test = np.array(test_data["y_test"])    # true labels ("ham"/"spam")

    # ── 2. Baseline: classify without any attack ──────────────────────────────
    print("[INFO] Running baseline classification (no attack) …")
    baseline_preds = _predict(model, vectorizer, X_test)

    overall_accuracy = accuracy_score(y_test, baseline_preds)
    print(f"  Overall accuracy : {overall_accuracy*100:.2f}%")

    # Select spam emails that are CORRECTLY detected (true positive)
    # mask = True where the email is spam AND the model correctly predicts spam
    correctly_detected_mask = (y_test == "spam") & (baseline_preds == "spam")
    spam_texts = [X_test[i] for i in range(len(X_test)) if correctly_detected_mask[i]]

    total_spam       = int((y_test == "spam").sum())
    n_detected       = len(spam_texts)
    detection_rate   = n_detected / total_spam if total_spam > 0 else 0.0

    print(f"  Total spam in test set : {total_spam}")
    print(
        f"  Correctly detected     : {n_detected}  "
        f"({detection_rate*100:.1f}%)"
    )
    print(f"\n[INFO] Attacking these {n_detected} detected spam emails …\n")

    # ── 3. Run each attack ────────────────────────────────────────────────────
    results = {}
    for key, attack_fn in ATTACKS.items():
        print(f"[INFO] {ATTACK_NAMES[key]} …")
        metrics = _run_attack(attack_fn, spam_texts, model, vectorizer, top_spam_words)
        results[key] = metrics
        print(
            f"       Evaded {metrics['n_evaded']} / {metrics['n_tested']}  "
            f"({metrics['evasion_rate']*100:.1f}% evasion rate)\n"
        )

    # ── 4. Print results table ────────────────────────────────────────────────
    _print_table(results, detection_rate, n_detected)

    # ── 5. Print interpretation hint ─────────────────────────────────────────
    best_attack = max(results, key=lambda k: results[k]["evasion_rate"])
    best_rate   = results[best_attack]["evasion_rate"] * 100
    print(
        f"  Most effective attack: {ATTACK_NAMES[best_attack]}  "
        f"({best_rate:.1f}% evasion)\n"
    )
    print(
        "  Takeaway: even a simple TF-IDF classifier can be partially fooled\n"
        "  by surface-level text mutations that don't change human meaning.\n"
        "  Defence strategies include: character normalisation, semantic\n"
        "  embeddings (word2vec / BERT), and adversarial training.\n"
    )


if __name__ == "__main__":
    main()
