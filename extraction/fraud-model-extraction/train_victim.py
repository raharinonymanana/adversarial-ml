"""
train_victim.py
---------------
Trains the VICTIM model — the model our attacker will later try to steal.

Scenario:
  A bank deploys a fraud detector behind an API. You send it a transaction,
  it answers "fraud" (1) or "legit" (0). You never see its code or weights.
  In this lab THAT model is the victim, and `extract.py` plays the attacker.

Model choice: an RBF-kernel Support Vector Machine (sklearn.svm.SVC).
  - probability=False  → the API only returns a hard label, not a score.
    (Hard labels are the *hardest* case for an attacker — they leak the
     least information — which makes the extraction result more impressive.)
  - class_weight="balanced" → the fraud class is only ~0.17% of the data,
    so without re-weighting the SVM would just predict "legit" for everything
    and score 99.8% "accuracy" while catching zero fraud. "balanced" tells
    sklearn to penalise mistakes on the rare class much more heavily.

Why we scale the features:
  SVMs measure distances between points, so a feature measured in the
  thousands (Amount) would completely drown out a feature measured in
  fractions (V1..V28). We wrap StandardScaler + SVC in a single Pipeline so
  scaling happens automatically every time the victim is queried — the
  attacker just sends raw transactions and never sees this preprocessing.

Why we subsample the training data:
  An RBF SVM costs roughly O(n^2)-O(n^3) to train. The full 80% split is
  ~227,000 rows, which would take hours and a lot of RAM on a CPU. We train
  the victim on a stratified random sample (VICTIM_TRAIN_SIZE rows) instead.
  The held-out 20% TEST set is still the full size, so evaluation is honest.

Artifacts saved to models/:
  victim_svm.pkl     — the fitted Pipeline (StandardScaler + SVC) = the victim
  test_data.pkl      — the held-out test set (X_test, y_test) for evaluation
  attacker_pool.pkl  — the training-portion FEATURES only (no labels). This is
                       the pool of unlabeled transactions the attacker is
                       assumed to have collected and will query the victim with.

Usage:
  python train_victim.py
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Reproducibility ───────────────────────────────────────────────────────────
# Seed EVERYTHING at 42 so every run produces identical splits and models.
RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_CSV   = BASE_DIR / "data"   / "creditcard.csv"
MODELS_DIR = BASE_DIR / "models"
VICTIM_PKL = MODELS_DIR / "victim_svm.pkl"
TEST_PKL   = MODELS_DIR / "test_data.pkl"
POOL_PKL   = MODELS_DIR / "attacker_pool.pkl"

# ── Hyper-parameters / sizes ──────────────────────────────────────────────────
TEST_SIZE = 0.20            # 80% train / 20% test split, as specified
# How many rows to actually train the RBF SVM on (see module docstring).
# Raise this if you have time/RAM to spare; the pipeline is unchanged.
VICTIM_TRAIN_SIZE = 20_000


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Read creditcard.csv and return (X, y):
      X — DataFrame of the 30 numeric features (Time, V1..V28, Amount)
      y — Series of the fraud label (Class: 0 = legit, 1 = fraud)
    """
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found.\n"
            "  Download the Kaggle 'Credit Card Fraud Detection' dataset:\n"
            "    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "  and place creditcard.csv in the data/ folder."
        )

    print(f"[INFO] Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)

    # The dataset must contain a 'Class' column (the fraud label).
    assert "Class" in df.columns, "creditcard.csv must contain a 'Class' column."

    # Split columns into features (everything except Class) and the label.
    X = df.drop(columns=["Class"])
    y = df["Class"]

    n_fraud  = int((y == 1).sum())
    n_legit  = int((y == 0).sum())
    fraud_pct = 100 * n_fraud / len(y)
    print(f"  {len(df):,} transactions  |  {n_legit:,} legit  |  {n_fraud:,} fraud")
    print(f"  Fraud is only {fraud_pct:.3f}% of the data - heavily imbalanced.")

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y = load_data()

    # ── 2. Train / test split (80 / 20) ───────────────────────────────────────
    # stratify=y keeps the same tiny fraud ratio in BOTH splits — otherwise a
    # random split might put almost no fraud cases in the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] Full split  ->  train: {len(X_train):,}   test: {len(X_test):,}")

    # ── 3. Subsample the training data for the SVM ────────────────────────────
    # If the training set is larger than VICTIM_TRAIN_SIZE, take a stratified
    # random subsample so RBF-SVM training stays fast on CPU. We sample with
    # train_test_split again, using `train_size` and stratify to preserve the
    # fraud ratio. The DISCARDED portion (`_`) is simply not used for fitting.
    if len(X_train) > VICTIM_TRAIN_SIZE:
        X_fit, _, y_fit, _ = train_test_split(
            X_train, y_train,
            train_size=VICTIM_TRAIN_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
        print(
            f"[INFO] Subsampled victim training set to {len(X_fit):,} rows "
            f"(RBF-SVM is too slow on all {len(X_train):,})."
        )
    else:
        X_fit, y_fit = X_train, y_train

    print(f"  Fraud cases in victim training subsample: {int((y_fit == 1).sum())}")

    # ── 4. Build the victim model (StandardScaler + SVC in one Pipeline) ──────
    # Wrapping the scaler and the SVC together means the attacker can send RAW
    # transactions to .predict() and the scaling happens internally — exactly
    # like a real black-box API.
    victim = Pipeline([
        ("scaler", StandardScaler()),                  # standardise every feature
        ("svc", SVC(
            kernel="rbf",                              # non-linear decision boundary
            probability=False,                         # hard labels only (no scores)
            class_weight="balanced",                   # counter the 0.17% fraud rate
            random_state=RANDOM_STATE,
        )),
    ])

    print("[INFO] Training the victim SVM (this is the model we will steal) ...")
    victim.fit(X_fit, y_fit)
    print("  Training complete.")

    # ── 5. Evaluate the victim on the FULL held-out test set ──────────────────
    y_pred = victim.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  Victim test accuracy : {accuracy*100:.3f}%")
    print(f"{'='*55}")

    # Accuracy alone is misleading on imbalanced data — always read precision
    # and recall for the fraud class (label 1).
    print("\n[INFO] Classification report (focus on the 'Fraud' row):")
    print(classification_report(
        y_test, y_pred, target_names=["Legit (0)", "Fraud (1)"], digits=3
    ))

    # Confusion matrix: rows = actual, cols = predicted.
    cm = confusion_matrix(y_test, y_pred)
    print("[INFO] Confusion matrix  (rows = actual, cols = predicted):")
    print(f"             pred legit   pred fraud")
    print(f"  actual legit  {cm[0,0]:>8}   {cm[0,1]:>9}")
    print(f"  actual fraud  {cm[1,0]:>8}   {cm[1,1]:>9}")

    # ── 6. Save artifacts ─────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)

    with open(VICTIM_PKL, "wb") as f:
        pickle.dump(victim, f)
    print(f"\n[INFO] Victim model saved   -> {VICTIM_PKL}")

    # Save the EXACT test set so extract.py and evaluate.py judge everyone on
    # the same data (no re-splitting, no leakage).
    with open(TEST_PKL, "wb") as f:
        pickle.dump({"X_test": X_test, "y_test": y_test}, f)
    print(f"[INFO] Test set saved       -> {TEST_PKL}")

    # The attacker's "pool" = the training-portion FEATURES with the labels
    # thrown away. The attacker is assumed to have collected these unlabeled
    # transactions; the only labels it can ever obtain come from querying the
    # victim. We save the full training features (not the 20k subsample) so the
    # attacker has plenty to sample 10,000 queries from.
    with open(POOL_PKL, "wb") as f:
        pickle.dump({"X_pool": X_train}, f)
    print(f"[INFO] Attacker pool saved  -> {POOL_PKL}  ({len(X_train):,} unlabeled rows)")

    print("\n[DONE] Next step: python extract.py")


if __name__ == "__main__":
    main()
