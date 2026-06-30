"""
art_assessment.py — Black-box robustness assessment using ART's HopSkipJump.

Run AFTER train_target.py (which must have created models/*.pkl first).

What this script does, step by step:
  1. Load the pre-trained MLP and the saved test split.
  2. Wrap the model in ART's SklearnClassifier adaptor.
  3. Attack it with HopSkipJump — a decision-based attack that needs only
     the model's predicted LABEL (no gradients, no probabilities).
     This mirrors real-world black-box access where an attacker can query
     a remote model but cannot inspect its weights or architecture.
  4. Repeat over 5 random seeds (multi-seed honesty) — a single run might
     be lucky or unlucky; averaging across seeds gives a fairer picture.
  5. Print mean ± std for ASR, L2, and query count.
  6. Save robustness_curve.png and adversarial_examples.png.
  7. Save numeric results to models/results.pkl.
"""

import os
import pickle
import numpy as np

# Use a non-interactive matplotlib backend BEFORE importing pyplot.
# 'Agg' writes to file without opening a display window — this prevents
# the script from hanging on headless machines or Windows environments
# where a GUI window would block further execution.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump

# Import our shared metrics — defined once in metrics.py, used everywhere.
from metrics import asr, l2_perturbation, robustness_curve

# ── 1. Load model and test split ──────────────────────────────────────────────
print("Loading model and test split ...")

with open("models/target_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/test_split.pkl", "rb") as f:
    split  = pickle.load(f)
    X_test = split["X_test"]   # shape (n_test, 64), values in [0, 1]
    y_test = split["y_test"]   # shape (n_test,)

print(f"Test set size : {len(X_test)} samples")

# ── 2. Wrap the sklearn model in ART ─────────────────────────────────────────
# SklearnClassifier is ART's adaptor layer between a plain sklearn estimator
# and the ART attack / defence ecosystem.
# clip_values=(0.0, 1.0) tells ART the valid input range — it clips any
# adversarial perturbation that would push a pixel below 0 or above 1.
art_classifier = SklearnClassifier(model=model, clip_values=(0.0, 1.0))

# Save a reference to the REAL predict method before we ever monkey-patch it.
# We'll need this both inside the counter and when restoring after each seed.
_real_predict = art_classifier.predict

# ── 3. Multi-seed assessment ──────────────────────────────────────────────────
# Why multiple seeds?
# A single random subsample might happen to contain only easy (or only hard)
# samples.  Running 5 seeds gives 5 independent estimates; reporting mean ± std
# shows both the typical result and how much it varies — this is "honest" ML.

SEEDS       = [0, 1, 2, 3, 4]
N_SUBSAMPLE = 50   # test points to sample per seed (kept small for CPU speed)

seed_results  = []   # accumulates per-seed dicts
last_seed_data = {}  # overwritten each seed; used for plots after the loop

for seed in SEEDS:
    print(f"\n-- Seed {seed} " + "-" * 52)
    np.random.seed(seed)

    # Draw a fresh random subset of test indices for this seed.
    idx   = np.random.choice(len(X_test), size=min(N_SUBSAMPLE, len(X_test)), replace=False)
    X_sub = X_test[idx]
    y_sub = y_test[idx]

    # Get original (clean) predictions.
    # ART's predict() returns a probability matrix (n, n_classes); argmax → label.
    orig_preds = np.argmax(art_classifier.predict(X_sub), axis=1)

    # Filter to samples the model originally got RIGHT.
    # There is no point attacking a wrong prediction — it is already wrong.
    correct_mask       = (orig_preds == y_sub)
    X_correct          = X_sub[correct_mask]
    y_correct          = y_sub[correct_mask]
    orig_preds_correct = orig_preds[correct_mask]
    n_correct          = len(X_correct)

    print(f"  Originally correct : {n_correct} / {len(X_sub)}")

    if n_correct == 0:
        print("  No correctly-classified samples in this seed — skipping.")
        continue

    # ── Query counting via monkey-patching ────────────────────────────────────
    # HopSkipJump is a QUERY-based attack: it only needs predict() labels.
    # We measure how many model calls it makes by temporarily replacing
    # art_classifier.predict with a wrapper that counts batch sizes.
    #
    # How the patch works:
    #   Python resolves attributes in order: instance dict → class dict.
    #   Setting art_classifier.predict = wrapper writes to the instance dict,
    #   so all calls to art_classifier.predict(x) now hit our wrapper first.
    #   After the attack we delete the instance attribute to restore the
    #   class method (no side-effects on future seeds).
    query_count = [0]   # list (not int) so the nested function can mutate it

    def _counting_predict(x, **kwargs):
        # x is always a batch; len(x) is the number of samples in this call.
        query_count[0] += len(x)
        return _real_predict(x, **kwargs)   # forward to the real method

    art_classifier.predict = _counting_predict   # activate counter

    # ── Run HopSkipJump ───────────────────────────────────────────────────────
    # HopSkipJump (Chen et al., 2020) is a decision-based attack.
    # It starts from a random point in the wrong class, then uses binary
    # search + Monte-Carlo gradient estimation to walk toward the decision
    # boundary, minimising the L2 distance from the original input.
    #
    # CPU caps chosen to keep each seed under ~3 minutes on a laptop:
    #   max_iter=15   — number of boundary-walking iterations
    #   max_eval=1000 — max gradient-direction evaluations per iteration
    #   init_eval=10  — evaluations for the random initialisation step
    attack = HopSkipJump(
        classifier = art_classifier,
        targeted   = False,   # untargeted: change the label to ANYTHING else
        norm       = 2,       # minimise L2 (Euclidean) perturbation size
        max_iter   = 15,
        max_eval   = 1000,
        init_eval  = 10,
    )

    print(f"  Running HopSkipJump on {n_correct} samples "
          f"(max_iter=15, max_eval=1000) - may take a few minutes ...")
    X_adv = attack.generate(x=X_correct)   # returns adversarial examples, same shape

    # ── Restore the real predict ──────────────────────────────────────────────
    # Deleting the instance attribute lets Python fall back to the class method.
    try:
        delattr(art_classifier, "predict")
    except AttributeError:
        art_classifier.predict = _real_predict   # fallback (should not be needed)

    # ── Compute per-seed metrics ──────────────────────────────────────────────
    adv_preds    = np.argmax(art_classifier.predict(X_adv), axis=1)
    success_mask = (adv_preds != orig_preds_correct)   # True = prediction flipped

    seed_asr = asr(orig_preds_correct, adv_preds, y_correct)
    l2_vals  = l2_perturbation(X_correct, X_adv)   # per-sample L2 distance

    # Mean L2 is only meaningful over SUCCESSFUL adversarial examples.
    mean_l2_success = (
        float(l2_vals[success_mask].mean()) if success_mask.any() else 0.0
    )

    # Average queries per sample.
    # Note: HopSkipJump processes all samples in batch, so per-sample
    # query counts are not individually accessible.  We report total / n
    # as an average (labelled accordingly).
    avg_queries = query_count[0] / n_correct

    print(f"  ASR                     : {seed_asr:.3f}")
    print(f"  Mean L2 (successful)    : {mean_l2_success:.4f}")
    print(f"  Avg queries / sample    : {avg_queries:.0f}")

    seed_results.append({
        "seed"       : seed,
        "asr"        : seed_asr,
        "mean_l2"    : mean_l2_success,
        "avg_queries": avg_queries,
    })

    # Keep this seed's raw arrays for plotting.
    # We overwrite each iteration; after the loop this holds the last seed.
    last_seed_data = {
        "X_orig"      : X_correct,
        "X_adv"       : X_adv,
        "orig_preds"  : orig_preds_correct,
        "adv_preds"   : adv_preds,
        "y_true"      : y_correct,
        "l2_vals"     : l2_vals,
        "success_mask": success_mask,
    }

# ── 4. Aggregate across seeds ─────────────────────────────────────────────────
if not seed_results:
    print("\nNo seeds completed — cannot aggregate or plot.  Exiting.")
    raise SystemExit(1)

asrs    = [r["asr"]         for r in seed_results]
l2s     = [r["mean_l2"]     for r in seed_results]
queries = [r["avg_queries"] for r in seed_results]

print("\n" + "=" * 60)
print("Multi-seed summary  (mean +/- std across completed seeds)")
print("-" * 60)
print(f"  ASR                     : {np.mean(asrs):.3f} +/- {np.std(asrs):.3f}")
print(f"  Mean L2 (successful)    : {np.mean(l2s):.4f} +/- {np.std(l2s):.4f}")
print(f"  Avg queries / sample    : {np.mean(queries):.0f} +/- {np.std(queries):.0f}")
print("=" * 60)

# ── 5. Robustness curve ───────────────────────────────────────────────────────
# Uses the last seed's data — representative of a single run.
l2_last   = last_seed_data["l2_vals"]
succ_last = last_seed_data["success_mask"]

# Build the x-axis grid from 0 to just above the max L2 observed.
eps_max  = float(l2_last.max()) * 1.1 + 0.01
eps_grid = np.linspace(0, eps_max, 60)

eps_vals, asr_at_eps = robustness_curve(l2_last, succ_last, eps_grid)

plt.figure(figsize=(8, 5))
plt.plot(eps_vals, asr_at_eps, color="crimson", linewidth=2)
plt.fill_between(eps_vals, asr_at_eps, alpha=0.15, color="crimson")
plt.xlabel("L2 budget  ε", fontsize=12)
plt.ylabel("Attack success rate", fontsize=12)
plt.title(
    "Robustness Curve — HopSkipJump on MLP Digit Classifier\n"
    "(fraction of originally-correct samples fooled vs. L2 perturbation budget)",
    fontsize=11,
)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("robustness_curve.png", dpi=150)
plt.close()
print("\nSaved  robustness_curve.png")

# ── 6. Before / after image grid ─────────────────────────────────────────────
# Show a handful of original digits (top row) vs. their adversarial versions
# (bottom row), so we can see visually how much the image changed.
n_show      = min(6, int(succ_last.sum()))
success_idx = np.where(succ_last)[0][:n_show]

if n_show == 0:
    print("No successful adversarial examples to display — skipping image grid.")
else:
    # squeeze=False keeps axes 2-D even when n_show == 1.
    fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 4.5), squeeze=False)

    for col, i in enumerate(success_idx):
        # ── Top row: original image ───────────────────────────────────────────
        # The 64-element vector is reshaped to 8×8 for display.
        axes[0, col].imshow(
            last_seed_data["X_orig"][i].reshape(8, 8),
            cmap="gray_r", vmin=0, vmax=1,
        )
        axes[0, col].set_title(
            f"true: {last_seed_data['y_true'][i]}\n"
            f"pred: {last_seed_data['orig_preds'][i]}",
            fontsize=8,
        )
        axes[0, col].axis("off")

        # ── Bottom row: adversarial image ─────────────────────────────────────
        axes[1, col].imshow(
            last_seed_data["X_adv"][i].reshape(8, 8),
            cmap="gray_r", vmin=0, vmax=1,
        )
        axes[1, col].set_title(
            f"adv pred: {last_seed_data['adv_preds'][i]}\n"
            f"L2: {last_seed_data['l2_vals'][i]:.2f}",
            fontsize=8,
        )
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original",    fontsize=9, labelpad=6)
    axes[1, 0].set_ylabel("Adversarial", fontsize=9, labelpad=6)

    plt.suptitle(
        "Adversarial Examples — HopSkipJump\n"
        "(minimal L2 perturbation that changes the predicted digit)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig("adversarial_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved  adversarial_examples.png")

# ── 7. Save numeric results ───────────────────────────────────────────────────
results = {
    "seed_results": seed_results,
    "summary": {
        "asr_mean"     : float(np.mean(asrs)),
        "asr_std"      : float(np.std(asrs)),
        "l2_mean"      : float(np.mean(l2s)),
        "l2_std"       : float(np.std(l2s)),
        "queries_mean" : float(np.mean(queries)),
        "queries_std"  : float(np.std(queries)),
    },
}

with open("models/results.pkl", "wb") as f:
    pickle.dump(results, f)

print("Saved  models/results.pkl")
print("\nDone.")
