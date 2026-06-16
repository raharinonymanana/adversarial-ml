"""
train_victim.py
---------------
Trains the VICTIM model — the model our attacker will later clone and then run
COMPLETELY OFFLINE (that offline part is the whole point of Lab 3).

Scenario:
  An ML service exposes a handwritten-digit classifier behind an API. You send
  it an 8x8 image, it answers with a digit 0-9. You never see its code, its
  weights, or any confidence score — only the predicted label. In this lab THAT
  model is the victim, and `extract.py` plays the attacker that steals it.

Why digits (sklearn.datasets.load_digits):
  - Ships INSIDE scikit-learn, so there is no download and no data/ folder — the
    whole lab runs from a fresh checkout on a CPU in seconds.
  - 1,797 tiny 8x8 images (64 features), 10 balanced classes — big enough to
    train a real neural net, small enough to stay fast.

The THREE disjoint slices (this split is what keeps the numbers honest):
  1. victim-train   — the only data the victim ever learns from.
  2. attacker-pool  — FEATURES ONLY, labels thrown away. This is the attacker's
                      stock of unlabeled images; the only labels it can obtain
                      come from querying the victim.
  3. held-out test  — touched by NOBODY during training. Both the victim and the
                      stolen clone are judged on this same untouched set.
  Because the three slices never overlap, the clone's score cannot be inflated by
  having secretly seen the test images during the attack.

Model choice — a small neural net (MLPClassifier) wrapped in a Pipeline:
  - hidden_layer_sizes=(128,): one small hidden layer is plenty for 8x8 digits
    and keeps CPU training to a few seconds.
  - We wrap StandardScaler + MLPClassifier in ONE Pipeline. Neural nets train by
    gradient descent, which converges far better when every input feature is on
    the same scale. Bundling the scaler in means the attacker can send RAW images
    to the victim and the scaling happens internally — exactly like a real API.
  - The clone in extract.py is a Random Forest instead: DIFFERENT internals on
    purpose. We are stealing the victim's BEHAVIOUR, not copying its weights.

The black box:
  We expose ONE function, victim_predict(X) -> labels. The attacker imports only
  this. It lazily loads the saved pipeline once and returns hard labels — the
  caller never touches the model object, just like querying a remote API.

Artifacts saved to models/:
  victim_mlp.pkl     — the fitted Pipeline (StandardScaler + MLP) = the victim
  test_data.pkl      — the held-out test set (X_test, y_test) for evaluation
  attacker_pool.pkl  — the attacker's UNLABELED image features (no labels)

Usage:
  python train_victim.py
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Reproducibility ───────────────────────────────────────────────────────────
# Seed EVERYTHING at 42 so every run produces identical splits and models.
RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
VICTIM_PKL = MODELS_DIR / "victim_mlp.pkl"
TEST_PKL   = MODELS_DIR / "test_data.pkl"
POOL_PKL   = MODELS_DIR / "attacker_pool.pkl"

# ── Split sizes (counts, not fractions, so the slices are explicit) ───────────
# 1,797 digits total  ->  300 test  +  1,000 attacker-pool  +  497 victim-train.
# Integer sizes make the three-way carve easy to read and verify.
TEST_SIZE = 300      # held-out rows nobody trains on
POOL_SIZE = 1000     # unlabeled rows handed to the attacker (>= N_QUERIES)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading + the three-way disjoint split
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load the built-in digits dataset and return (X, y):
      X — (1797, 64) array of pixel values (each row is a flattened 8x8 image)
      y — (1797,)    array of the true digit label (0-9)
    No download, no files — the data ships inside scikit-learn.
    """
    print("[INFO] Loading sklearn digits dataset (no download needed) ...")
    X, y = load_digits(return_X_y=True)
    print(f"  {len(X):,} images  |  {X.shape[1]} features each (8x8 pixels)  |  "
          f"{len(np.unique(y))} balanced classes (0-9)")
    return X, y


def make_splits(X, y):
    """
    Carve X, y into the THREE disjoint slices described in the module docstring.

    Done in two stratified steps (stratify keeps all 10 digits evenly present):
      step 1: peel off the held-out TEST set (TEST_SIZE rows).
      step 2: from what remains, peel off the attacker POOL (POOL_SIZE rows);
              everything still left becomes the victim TRAIN set.

    Returns:
        X_vtrain, y_vtrain — victim's private training data
        X_pool             — attacker's unlabeled images (labels deliberately dropped)
        X_test,  y_test    — shared held-out test set
    """
    # Step 1 — hold out the test set first so it can never leak into anything else.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Step 2 — split the remainder into attacker-pool and victim-train.
    X_vtrain, X_pool, y_vtrain, y_pool = train_test_split(
        X_dev, y_dev, test_size=POOL_SIZE, random_state=RANDOM_STATE, stratify=y_dev
    )
    # y_pool is intentionally discarded: the attacker's pool is UNLABELED.
    del y_pool

    print(f"\n[INFO] Disjoint split  ->  victim-train: {len(X_vtrain):,}   "
          f"attacker-pool: {len(X_pool):,}   test: {len(X_test):,}")
    return X_vtrain, y_vtrain, X_pool, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# The black-box query channel  (the ONLY thing the attacker imports)
# ══════════════════════════════════════════════════════════════════════════════

# Module-level cache so we load the pickled victim from disk only once, no matter
# how many times the attacker queries it.
_victim_model = None


def victim_predict(X):
    """
    The victim's prediction API — the single channel the attacker is allowed.

    Send it feature rows, get back HARD labels (digits 0-9), nothing else. It
    lazily loads the trained pipeline from victim_mlp.pkl on the first call and
    reuses it afterwards. The caller (the attacker) never sees the model object,
    its weights, or any probability — exactly like hitting a remote black-box API.
    """
    global _victim_model
    if _victim_model is None:
        if not VICTIM_PKL.exists():
            raise FileNotFoundError(
                f"{VICTIM_PKL} not found. Run: python train_victim.py"
            )
        with open(VICTIM_PKL, "rb") as f:
            _victim_model = pickle.load(f)
    return _victim_model.predict(X)


# ══════════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load data and make the three disjoint slices ───────────────────────
    X, y = load_data()
    X_vtrain, y_vtrain, X_pool, X_test, y_test = make_splits(X, y)

    # ── 2. Build the victim model (StandardScaler + small MLP in one Pipeline) ─
    # Bundling the scaler with the net means the attacker sends RAW images and
    # the standardisation happens internally — invisible, just like a real API.
    victim = Pipeline([
        ("scaler", StandardScaler()),                  # put every pixel on one scale
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128,),                 # one small hidden layer
            max_iter=2000,                             # plenty of room to converge
            random_state=RANDOM_STATE,
        )),
    ])

    print("[INFO] Training the victim MLP (this is the model we will steal) ...")
    victim.fit(X_vtrain, y_vtrain)
    print("  Training complete.")

    # ── 3. Evaluate the victim on the held-out test set ───────────────────────
    y_pred = victim.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  Victim test accuracy : {accuracy*100:.2f}%")
    print(f"{'='*55}")

    # Per-digit precision/recall — digits are balanced, so accuracy is honest
    # here, but the report shows which digits the victim confuses.
    print("\n[INFO] Classification report (per digit):")
    print(classification_report(y_test, y_pred, digits=3))

    # Confusion matrix: rows = actual digit, cols = predicted digit. A clean
    # diagonal means few mistakes. We print it as a compact 10x10 grid.
    cm = confusion_matrix(y_test, y_pred)
    print("[INFO] Confusion matrix  (rows = actual digit, cols = predicted):")
    header = "        " + "".join(f"{d:>4}" for d in range(10))
    print(header)
    for actual_digit, row in enumerate(cm):
        cells = "".join(f"{count:>4}" for count in row)
        print(f"  act {actual_digit} |{cells}")

    # ── 4. Save artifacts ─────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)

    with open(VICTIM_PKL, "wb") as f:
        pickle.dump(victim, f)
    print(f"\n[INFO] Victim model saved   -> {VICTIM_PKL}")

    # Save the EXACT test set so extract.py and evaluate.py judge everyone on the
    # same untouched data (no re-splitting, no leakage).
    with open(TEST_PKL, "wb") as f:
        pickle.dump({"X_test": X_test, "y_test": y_test}, f)
    print(f"[INFO] Test set saved       -> {TEST_PKL}")

    # The attacker's pool = unlabeled image FEATURES only. The labels were dropped
    # in make_splits(); the attacker must query the victim to get any label.
    with open(POOL_PKL, "wb") as f:
        pickle.dump({"X_pool": X_pool}, f)
    print(f"[INFO] Attacker pool saved  -> {POOL_PKL}  ({len(X_pool):,} unlabeled images)")

    print("\n[DONE] Next step: python extract.py")


if __name__ == "__main__":
    main()
