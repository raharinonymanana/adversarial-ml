"""
extract.py
----------
The ATTACK. We play an attacker who can do exactly one thing to the victim: call
its black-box API, victim_predict(X), and read back the hard label. From nothing
but those answers we train our OWN model that copies the victim's behaviour — a
*model extraction* (a.k.a. model stealing) attack.

Attacker's situation (what we are and aren't allowed to use):
  ✓ A pool of UNLABELED images (attacker_pool.pkl) — features only.
  ✓ Black-box query access: victim_predict(X) -> labels (imported from train_victim).
  ✗ NO access to the victim's training data, code, weights, or probabilities.

The attack, step by step:
  1. Draw N_QUERIES (1,000) images from the unlabeled pool.
  2. Ask the victim to label them  ->  these become our STOLEN labels.
  3. Treat (image, victim_label) pairs as a brand-new training set.
  4. Train a Random Forest clone on that stolen dataset. A Random Forest has
     COMPLETELY DIFFERENT internals from the victim's neural net — we copy
     behaviour, not weights, which is exactly what makes extraction dangerous.
  5. Measure ACCURACY (clone vs TRUE labels) and AGREEMENT (clone vs victim) on
     the held-out test set. High agreement = we have cloned the victim.

Why we run it FIVE times (honesty over a single lucky number):
  A single extraction reports one number, and that number wobbles depending on
  which 1,000 images we happened to query and how the forest happened to grow. So
  we repeat the WHOLE attack across SEEDS = [42..46] — varying only the attack —
  on the SAME fixed victim and SAME fixed test set, and report mean ± std. We do
  NOT retrain the victim per seed (a real attacker faces one deployed model), and
  no hyperparameter is tuned to make a reported number look better.

Why this clone matters for Lab 3:
  We save the seed-42 clone as clone_rf.pkl — a self-contained file. Lab 3's
  punch line (offline_inference.py) loads exactly that file and predicts with the
  network physically disabled — proving the stolen model is now a standalone asset.

build_stolen_dataset(), agreement_rate(), and run_one_extraction() are also
imported by the notebook, so the attack math lives in ONE place.

Artifacts saved to models/:
  clone_rf.pkl           — the seed-42 Random Forest clone (the deployed asset)
  extraction_results.pkl — per-seed accuracy/agreement lists + mean/std (read by
                           evaluate.py, which trains nothing new)

Usage:
  python extract.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# The attacker's ONLY channel to the victim: the black-box prediction function.
# We import the function, never the model object — that is the whole point.
from train_victim import victim_predict

# We print "mean ± std" with the ± symbol. The default Windows console (cp1252)
# would crash on some Unicode glyphs, so switch stdout to UTF-8 up front.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
TEST_PKL    = MODELS_DIR / "test_data.pkl"
POOL_PKL    = MODELS_DIR / "attacker_pool.pkl"
CLONE_PKL   = MODELS_DIR / "clone_rf.pkl"
RESULTS_PKL = MODELS_DIR / "extraction_results.pkl"

# ── Attack budget ─────────────────────────────────────────────────────────────
# How many times the attacker is allowed to query the victim. Every query is one
# (image -> stolen label) pair. More queries usually = a better clone.
N_QUERIES = 1000

# ── Robustness seeds ──────────────────────────────────────────────────────────
# We repeat the WHOLE attack once per seed (varying only the attack) and report
# mean ± std, so a single lucky/unlucky run can't masquerade as "the" result.
# Five seeds is a small sample for an std — illustrative, not definitive.
SEEDS = [42, 43, 44, 45, 46]


# ══════════════════════════════════════════════════════════════════════════════
# Loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_pool():
    """Load the attacker's pool of UNLABELED image features."""
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
# Core attack building blocks  (also imported by evaluate.py and the notebook)
# ══════════════════════════════════════════════════════════════════════════════

def build_stolen_dataset(query_fn, pool, n_queries, random_state=RANDOM_STATE):
    """
    Steps 1-3 of the attack.

    Draw `n_queries` random images from the unlabeled `pool`, ask the victim to
    label them via `query_fn`, and return (X_stolen, y_stolen).

    Args:
        query_fn     — the victim's black-box API (victim_predict). The attacker
                       holds only this function, never the model behind it.
        pool         — array of unlabeled image features
        n_queries    — how many images to query (the query budget)
        random_state — seed so the SAME rows are drawn every run

    Returns:
        X_stolen — the queried image rows
        y_stolen — victim-predicted labels for them (our stolen labels)
    """
    # Never request more queries than there are images in the pool.
    n_queries = min(n_queries, len(pool))

    # Pick n_queries distinct rows at random; the seed makes the choice repeatable.
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(pool), size=n_queries, replace=False)
    X_stolen = pool[idx]

    # THE QUERY: the only information the attacker ever gets from the victim.
    y_stolen = query_fn(X_stolen)

    return X_stolen, y_stolen


def get_clone(seed=RANDOM_STATE):
    """
    Return a fresh, untrained Random Forest clone for a given random seed.

    Deliberately a DIFFERENT model family from the victim's neural net: a forest
    of decision trees voting together. Trees are scale-invariant, so (unlike the
    victim) the clone needs no StandardScaler. We are after the victim's
    behaviour, not its architecture.

    Every hyperparameter EXCEPT the seed is left at scikit-learn's default on
    purpose. Nothing here is tuned to push a reported number over a threshold —
    that would make the results dishonest. `seed` is the only knob, and we vary it
    only to measure how much the attack's outcome wobbles run-to-run.
    """
    return RandomForestClassifier(
        random_state=seed,           # the only knob we set: varies the forest per seed
        n_jobs=-1,                   # use all CPU cores (speed only; still CPU-only)
    )


def train_clone(model, X_stolen, y_stolen):
    """Fit the clone on the stolen (image, victim_label) dataset."""
    model.fit(X_stolen, y_stolen)
    return model


def run_one_extraction(victim, pool, X_test, y_test, seed):
    """
    Run ONE complete extraction with a given seed and return its metrics.

    A *pure* helper: it touches no global state, no disk, and never the victim
    object — given the same inputs and seed it always returns the same numbers.
    Repeating it across seeds is how main() measures the attack's variance.

    Args:
        victim — the black-box query function (victim_predict). We may CALL it,
                 but never inspect the model behind it.
        pool   — the attacker's unlabeled image pool
        X_test, y_test — the FIXED held-out test set (shared across all seeds)
        seed   — controls BOTH which pool rows we query AND the forest's growth

    Returns:
        dict with keys:
          accuracy  — clone correctness vs the TRUE digit labels
          agreement — fraction of test images where clone matches the victim
          clone     — the fitted Random Forest (so the caller can save seed 42)
    """
    # Steal a fresh dataset; this seed controls WHICH 1,000 pool rows we query.
    X_stolen, y_stolen = build_stolen_dataset(victim, pool, N_QUERIES, random_state=seed)

    # Train the clone; the same seed controls the forest's internal randomness.
    clone = get_clone(seed)
    train_clone(clone, X_stolen, y_stolen)

    # Score on the FIXED test set. The victim's labels are the agreement reference
    # (identical every seed, since the victim never changes).
    victim_preds = victim(X_test)
    clone_preds  = clone.predict(X_test)
    return {
        "accuracy":  float((np.asarray(y_test) == clone_preds).mean()),
        "agreement": agreement_rate(victim_preds, clone_preds),
        "clone": clone,
    }


def agreement_rate(victim_labels, clone_labels):
    """
    The key metric of an extraction attack.

    Fraction of samples where the clone predicts the SAME label as the victim.
    1.0 = perfect clone (always agrees); for 10 balanced digit classes, ~0.1 is
    what blind guessing would score. We compare predicted-vs-predicted (not vs
    the TRUE labels) because the attacker's goal is to copy the victim, mistakes
    and all. (Same definition as Lab 2, kept here so this lab is self-contained.)
    """
    victim_labels = np.asarray(victim_labels)
    clone_labels = np.asarray(clone_labels)
    return float((victim_labels == clone_labels).mean())


# ══════════════════════════════════════════════════════════════════════════════
# Main attack run
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load the unlabeled pool and the FIXED test set (once, shared) ──────
    print("[INFO] Loading attacker pool and test set ...")
    pool = load_pool()
    X_test, y_test = load_test()

    # Victim accuracy is identical across seeds (the victim never changes); compute
    # it once, through the black-box API, purely for context in the report.
    victim_acc = float((np.asarray(y_test) == victim_predict(X_test)).mean())

    # ── 2. Repeat the WHOLE attack once per seed, varying only the attack ──────
    print(f"[INFO] Running {len(SEEDS)} independent extractions "
          f"({N_QUERIES:,} queries each); victim + test set held fixed ...\n")

    accuracies, agreements = [], []
    canonical_clone = None        # the seed-42 clone = the asset we will deploy

    print(f"  {'seed':>5}   {'accuracy':>9}   {'agreement':>10}")
    print(f"  {'-'*5}   {'-'*9}   {'-'*10}")
    for seed in SEEDS:
        res = run_one_extraction(victim_predict, pool, X_test, y_test, seed)
        accuracies.append(res["accuracy"])
        agreements.append(res["agreement"])
        if seed == RANDOM_STATE:
            canonical_clone = res["clone"]
        print(f"  {seed:>5}   {res['accuracy']*100:>8.2f}%   {res['agreement']*100:>9.2f}%")

    # ── 3. Summarise as mean ± std (sample std, ddof=1, over the seeds) ────────
    acc_mean, acc_std = float(np.mean(accuracies)), float(np.std(accuracies, ddof=1))
    agr_mean, agr_std = float(np.mean(agreements)), float(np.std(agreements, ddof=1))

    print(f"\n{'='*55}")
    print(f"  Clone accuracy  : {acc_mean*100:5.2f}% ± {acc_std*100:.2f}%   "
          f"(mean ± std, {len(SEEDS)} seeds)")
    print(f"  Clone agreement : {agr_mean*100:5.2f}% ± {agr_std*100:.2f}%   "
          f"(mean ± std, {len(SEEDS)} seeds)")
    print(f"{'='*55}")

    # ── 4. Save the canonical (seed-42) clone — the asset offline_inference uses ─
    MODELS_DIR.mkdir(exist_ok=True)
    with open(CLONE_PKL, "wb") as f:
        pickle.dump(canonical_clone, f)
    print(f"\n[INFO] Canonical clone (seed {RANDOM_STATE}) saved -> {CLONE_PKL}")

    # ── 5. Save per-seed metrics + summary for evaluate.py (it trains nothing) ─
    with open(RESULTS_PKL, "wb") as f:
        pickle.dump({
            "seeds": list(SEEDS),
            "accuracy": accuracies,
            "agreement": agreements,
            "accuracy_mean": acc_mean,   "accuracy_std": acc_std,
            "agreement_mean": agr_mean,  "agreement_std": agr_std,
            "victim_accuracy": victim_acc,
            "n_queries": int(N_QUERIES),
            "n_test": int(len(y_test)),
        }, f)
    print(f"[INFO] Per-seed results saved          -> {RESULTS_PKL}")

    print("\n[DONE] Next steps: python evaluate.py   then   python offline_inference.py")


if __name__ == "__main__":
    main()
