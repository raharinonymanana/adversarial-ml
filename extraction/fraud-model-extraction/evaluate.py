"""
evaluate.py
-----------
Scores the extraction attack and prints a clean comparison table:
the victim vs. all three stolen "shadow" models on the SAME held-out test set.

It loads the artifacts produced by the earlier steps (it trains nothing new),
so the numbers here exactly match what extract.py reported.

Metrics printed:
  Victim accuracy      — how good the real model is, vs the TRUE labels.
  Shadow accuracy      — how good each clone is, vs the TRUE labels.
  Agreement rate       — % of test transactions where a shadow predicts the
                         SAME label as the victim. This is the headline
                         extraction metric: it measures how well we copied the
                         victim, regardless of whether the victim was right.
  Winner               — the shadow with the highest agreement.

Why agreement, not accuracy, decides the winner:
  The attacker's goal is to CLONE the victim — to reproduce its decisions,
  including its mistakes. A shadow that agrees 99% of the time has effectively
  stolen the model even if both are imperfect fraud detectors.

Usage:
  python evaluate.py    (run train_victim.py and extract.py first)
"""

import pickle
from pathlib import Path

import numpy as np

# Reuse the attack's own metric helper so the math is defined in exactly one place.
from extract import agreement_rate

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
VICTIM_PKL  = MODELS_DIR / "victim_svm.pkl"
TEST_PKL    = MODELS_DIR / "test_data.pkl"
SHADOWS_PKL = MODELS_DIR / "shadow_models.pkl"
RESULTS_PKL = MODELS_DIR / "extraction_results.pkl"


def _load_all():
    """Load victim, shadow models, results, and the held-out test set."""
    for path in (VICTIM_PKL, TEST_PKL, SHADOWS_PKL):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.\n"
                "  Run, in order:  python train_victim.py  then  python extract.py"
            )

    with open(VICTIM_PKL, "rb") as f:
        victim = pickle.load(f)
    with open(SHADOWS_PKL, "rb") as f:
        shadows = pickle.load(f)
    with open(TEST_PKL, "rb") as f:
        test = pickle.load(f)

    # n_queries is nice-to-have context; default if results file is missing.
    n_queries = None
    if RESULTS_PKL.exists():
        with open(RESULTS_PKL, "rb") as f:
            n_queries = pickle.load(f).get("n_queries")

    return victim, shadows, test["X_test"], test["y_test"], n_queries


def _accuracy(y_true, y_pred):
    """Plain accuracy = fraction of predictions that match the true labels."""
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def _print_table(victim_acc, rows, best_name):
    """
    Render an ASCII comparison table.
    `rows` is a list of (name, accuracy, agreement) tuples for the shadows.
    """
    W_NAME, W_ACC, W_AGREE = 24, 12, 14
    width = W_NAME + W_ACC + W_AGREE + 4
    thick = "=" * width
    thin  = "-" * width

    print()
    print(thick)
    print("  MODEL EXTRACTION RESULTS - Credit-Card Fraud Classifier")
    print(thick)
    print(f"\n  Victim (SVM) accuracy on test set : {victim_acc*100:.2f}%")
    print("  (The victim is the reference; shadows are judged on AGREEMENT.)\n")

    # Header
    print(f"{'Shadow model':<{W_NAME}}{'Accuracy':>{W_ACC}}{'Agreement':>{W_AGREE}}")
    print(thin)
    for name, acc, agree in rows:
        # Mark the winner with a star so it stands out at a glance.
        star = "  <-- best clone" if name == best_name else ""
        print(f"{name:<{W_NAME}}{acc*100:>{W_ACC-1}.2f}%{agree*100:>{W_AGREE-1}.2f}%{star}")
    print(thin)

    print("\n  Legend")
    print("  ------")
    print("  Accuracy   - clone's correctness vs the TRUE fraud labels")
    print("  Agreement  - % of test rows where the clone matches the VICTIM")
    print("               (this is what 'stealing the model' really measures)")
    print()


def main():
    # ── 1. Load everything ────────────────────────────────────────────────────
    print("[INFO] Loading victim, shadow models, and test set ...")
    victim, shadows, X_test, y_test, n_queries = _load_all()
    if n_queries:
        print(f"  (Attack used {n_queries:,} victim queries.)")

    # ── 2. Victim performance, and its predictions (the agreement reference) ──
    victim_preds = victim.predict(X_test)
    victim_acc   = _accuracy(y_test, victim_preds)

    # ── 3. Score every shadow: accuracy vs truth, agreement vs victim ─────────
    rows = []
    for name, model in shadows.items():
        shadow_preds = model.predict(X_test)
        acc   = _accuracy(y_test, shadow_preds)
        agree = agreement_rate(victim_preds, shadow_preds)
        rows.append((name, acc, agree))

    # ── 4. Decide the winner = highest agreement with the victim ──────────────
    best_name, _, best_agree = max(rows, key=lambda r: r[2])

    # ── 5. Print the comparison table ─────────────────────────────────────────
    _print_table(victim_acc, rows, best_name)

    print(f"  Best clone: {best_name}  ({best_agree*100:.2f}% agreement with the victim)\n")
    print(
        "  Takeaway: with only black-box, hard-label queries an attacker can\n"
        "  build a model that mimics the victim on the vast majority of inputs\n"
        "  -- no access to the victim's data, weights, or code required.\n"
    )


if __name__ == "__main__":
    main()
