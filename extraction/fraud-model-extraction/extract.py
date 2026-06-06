"""
extract.py
----------
The ATTACK. We play an attacker who can only do one thing to the victim:
send it transactions and read back the hard label it returns. From nothing but
those query answers we train our own "shadow" models that copy the victim's
behaviour — this is a *model extraction* (a.k.a. model stealing) attack.

Attacker's situation (what we are and aren't allowed to use):
  ✓ A pool of UNLABELED transactions (attacker_pool.pkl) — features only.
  ✓ Black-box query access to the victim: victim.predict(X) -> labels.
  ✗ NO access to the victim's training labels, code, weights, or scores.

The attack, step by step:
  1. Draw N_QUERIES (10,000) transactions from the unlabeled pool.
  2. Ask the victim to label them  ->  these become our STOLEN labels.
  3. Treat (features, victim_label) pairs as a brand-new training set.
  4. Train three different shadow models on that stolen dataset:
       - Logistic Regression
       - Decision Tree
       - Random Forest
  5. Pick the winner by AGREEMENT RATE: which shadow most often predicts the
     same label as the victim on a held-out test set. High agreement = we have
     successfully cloned the victim's decision boundary.

Class imbalance note:
  Fraud is ~0.17% of the data, so every shadow model also uses
  class_weight="balanced". Otherwise a shadow would just learn "always legit".

The functions here (build_stolen_dataset, get_shadow_models, train_shadow,
agreement_rate) are also imported by the notebook to draw the
"agreement vs query budget" curve — so all attack logic lives in ONE place.

Artifacts saved to models/:
  shadow_models.pkl      — dict of the three fitted shadow models
  extraction_results.pkl — metrics + which model won (used by evaluate.py)

Usage:
  python extract.py
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
VICTIM_PKL = MODELS_DIR / "victim_svm.pkl"
TEST_PKL   = MODELS_DIR / "test_data.pkl"
POOL_PKL   = MODELS_DIR / "attacker_pool.pkl"
SHADOWS_PKL = MODELS_DIR / "shadow_models.pkl"
RESULTS_PKL = MODELS_DIR / "extraction_results.pkl"

# ── Attack budget ─────────────────────────────────────────────────────────────
# How many times the attacker is allowed to query the victim. Every query is
# one (feature row -> stolen label) pair. More queries usually = better clone.
N_QUERIES = 10_000


# ══════════════════════════════════════════════════════════════════════════════
# Loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_victim():
    """Load the trained victim Pipeline (the black box we are stealing)."""
    if not VICTIM_PKL.exists():
        raise FileNotFoundError(f"{VICTIM_PKL} not found. Run: python train_victim.py")
    with open(VICTIM_PKL, "rb") as f:
        return pickle.load(f)


def load_pool():
    """Load the attacker's pool of UNLABELED transaction features."""
    if not POOL_PKL.exists():
        raise FileNotFoundError(f"{POOL_PKL} not found. Run: python train_victim.py")
    with open(POOL_PKL, "rb") as f:
        return pickle.load(f)["X_pool"]


def load_test():
    """Load the held-out test set (X_test, y_test) for honest evaluation."""
    if not TEST_PKL.exists():
        raise FileNotFoundError(f"{TEST_PKL} not found. Run: python train_victim.py")
    with open(TEST_PKL, "rb") as f:
        data = pickle.load(f)
    return data["X_test"], data["y_test"]


# ══════════════════════════════════════════════════════════════════════════════
# Core attack building blocks  (also imported by the notebook)
# ══════════════════════════════════════════════════════════════════════════════

def build_stolen_dataset(victim, pool, n_queries, random_state=RANDOM_STATE):
    """
    Step 1-3 of the attack.

    Draw `n_queries` random rows from the unlabeled `pool`, ask the `victim`
    to label them, and return (X_stolen, y_stolen).

    Args:
        victim       — the fitted victim Pipeline (black box)
        pool         — DataFrame of unlabeled transaction features
        n_queries    — how many transactions to query (the query budget)
        random_state — seed so the SAME rows are drawn every run

    Returns:
        X_stolen — DataFrame of the queried feature rows
        y_stolen — numpy array of victim-predicted labels (our stolen labels)
    """
    # Never request more queries than there are rows in the pool.
    n_queries = min(n_queries, len(pool))

    # .sample() picks n_queries random rows; the seed makes it repeatable.
    X_stolen = pool.sample(n=n_queries, random_state=random_state)

    # THE QUERY: this is the only information the attacker gets from the victim.
    y_stolen = victim.predict(X_stolen)

    return X_stolen, y_stolen


def get_shadow_models():
    """
    Return a fresh dict of the three (untrained) shadow models.

    All three use class_weight="balanced" because the stolen labels inherit the
    victim's heavy fraud imbalance. Logistic Regression is wrapped with a
    StandardScaler (it, like the victim's SVM, is distance/scale sensitive);
    the tree-based models are scale-invariant so they need no scaler.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,                 # ensure convergence on 30 features
                random_state=RANDOM_STATE,
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,                  # 100 trees voting together
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,                         # use all CPU cores (still CPU-only)
        ),
    }


def train_shadow(model, X_stolen, y_stolen):
    """Fit one shadow model on the stolen (features, victim_label) dataset."""
    model.fit(X_stolen, y_stolen)
    return model


def agreement_rate(victim_labels, shadow_labels):
    """
    The key metric of an extraction attack.

    Fraction of samples where the shadow predicts the SAME label as the victim.
    1.0 = perfect clone (always agrees); 0.5 = no better than a coin flip on a
    balanced two-class problem. We compare predicted-vs-predicted (not vs the
    TRUE labels) because the attacker's goal is to copy the victim, bugs and all.
    """
    victim_labels = np.asarray(victim_labels)
    shadow_labels = np.asarray(shadow_labels)
    return float((victim_labels == shadow_labels).mean())


# ══════════════════════════════════════════════════════════════════════════════
# Main attack run
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load the victim, the unlabeled pool, and the test set ──────────────
    print("[INFO] Loading victim, attacker pool, and test set ...")
    victim = load_victim()
    pool   = load_pool()
    X_test, y_test = load_test()

    # ── 2. Build the stolen dataset by querying the victim ────────────────────
    print(f"[INFO] Querying the victim {N_QUERIES:,} times to steal labels ...")
    X_stolen, y_stolen = build_stolen_dataset(victim, pool, N_QUERIES)

    n_fraud_stolen = int((y_stolen == 1).sum())
    print(
        f"  Stolen dataset: {len(X_stolen):,} rows  |  "
        f"victim called {n_fraud_stolen} of them fraud"
    )

    # The victim's own predictions on the test set are our "ground truth" for
    # agreement: a perfect clone would reproduce these exactly.
    victim_test_preds = victim.predict(X_test)

    # ── 3. Train all three shadow models on the stolen dataset ────────────────
    print("[INFO] Training 3 shadow models on the stolen labels ...\n")
    shadows = get_shadow_models()
    results = {}      # name -> dict of metrics
    fitted  = {}      # name -> fitted model (saved for evaluate.py)

    for name, model in shadows.items():
        print(f"  - Training {name} ...")
        train_shadow(model, X_stolen, y_stolen)
        fitted[name] = model

        # How often does this shadow agree with the victim on the test set?
        shadow_test_preds = model.predict(X_test)
        agree = agreement_rate(victim_test_preds, shadow_test_preds)

        # Accuracy vs the TRUE labels — how good a fraud detector the clone is
        # in its own right (a side metric; agreement is what we rank on).
        acc = float((np.asarray(y_test) == shadow_test_preds).mean())

        results[name] = {"agreement": agree, "accuracy": acc}
        print(f"      agreement with victim: {agree*100:.2f}%   "
              f"(own accuracy: {acc*100:.2f}%)\n")

    # ── 4. Pick the winner: highest agreement with the victim ─────────────────
    best_name = max(results, key=lambda n: results[n]["agreement"])
    print(f"{'='*55}")
    print(f"  WINNER: {best_name}  "
          f"({results[best_name]['agreement']*100:.2f}% agreement)")
    print(f"{'='*55}")

    # ── 5. Save the fitted shadows and the results ────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    with open(SHADOWS_PKL, "wb") as f:
        pickle.dump(fitted, f)
    print(f"\n[INFO] Shadow models saved -> {SHADOWS_PKL}")

    with open(RESULTS_PKL, "wb") as f:
        pickle.dump({
            "results": results,
            "best_name": best_name,
            "n_queries": int(len(X_stolen)),
        }, f)
    print(f"[INFO] Results saved       -> {RESULTS_PKL}")

    print("\n[DONE] Next step: python evaluate.py")


if __name__ == "__main__":
    main()
