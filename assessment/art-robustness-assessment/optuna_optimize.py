"""
optuna_optimize.py — Tune HopSkipJump to find the GENTLEST effective attack.

Goal: find the combination of HopSkipJump hyperparameters that minimises the
mean L2 perturbation (smallest change to the input) while keeping ASR >= ~0.95
and staying within the Stage 1 query budget of ~580 queries/sample.

This is the "attacks as optimisation" idea from NVIDIA DLI Module 4: instead
of picking attack parameters by hand, we let Optuna's TPE sampler search for
the configuration that makes the attack most efficient -- fewer pixel changes,
same fooling power.

Run AFTER train_target.py.  Does NOT modify any Stage 1 files.

Outputs (to models/, which is git-ignored):
  models/optuna_results.pkl     -- study results and best params
  models/optuna_comparison.png  -- baseline vs. best comparison bar chart
"""

import os
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- saves PNGs without a display window
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump

# Import shared metrics -- defined ONCE in metrics.py, never redefined here.
from metrics import asr, l2_perturbation

# Suppress Optuna's per-trial INFO logs; we print our own one-liner per trial.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────
N_TRIALS     = 30     # number of Optuna trials
N_SUBSAMPLE  = 30     # test points per trial (small so each trial is fast)
N_COMPARE    = 50     # points for the final baseline vs. best comparison
QUERY_BUDGET = 600    # Stage 1 averaged ~580 queries/sample; this is our ceiling

# Stage 1's exact HopSkipJump settings -- used as the comparison baseline.
# init_size=100 matches ART's default (Stage 1 did not set it explicitly).
BASELINE_PARAMS = dict(max_iter=15, max_eval=1000, init_eval=10, init_size=100)

# ── 1. Load model + test split (same files as Stage 1) ───────────────────────
# We NEVER retrain -- the whole point is to assess the same model.
print("Loading model and test split ...")

with open("models/target_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/test_split.pkl", "rb") as f:
    split  = pickle.load(f)
    X_test = split["X_test"]
    y_test = split["y_test"]

print(f"Test set size : {len(X_test)} samples")

# ── 2. ART wrapper -- created ONCE and shared across all trials ───────────────
# Wrapping inside the objective would work but wastes time re-wrapping every trial.
art_classifier = SklearnClassifier(model=model, clip_values=(0.0, 1.0))
_real_predict  = art_classifier.predict   # saved before any monkey-patching


# ── 3. Shared attack runner ───────────────────────────────────────────────────
def run_attack(max_iter, max_eval, init_eval, init_size, seed, n_points):
    """
    Attack a seeded random subsample of the test set with one HopSkipJump config.

    Uses the same correctness gate as Stage 1:
      - subsample n_points test points
      - keep only the ones the model ORIGINALLY classifies correctly
      - attack those; compute ASR and L2 on the results

    Returns
    -------
    (trial_asr, mean_l2_successful, avg_queries_per_sample)
    All floats.  mean_l2 = 0.0 if no samples were successfully flipped.
    """
    np.random.seed(seed)
    idx   = np.random.choice(len(X_test), size=min(n_points, len(X_test)), replace=False)
    X_sub = X_test[idx]
    y_sub = y_test[idx]

    # Only attack samples the model originally gets right.
    orig_preds   = np.argmax(art_classifier.predict(X_sub), axis=1)
    correct_mask = (orig_preds == y_sub)
    X_correct    = X_sub[correct_mask]
    y_correct    = y_sub[correct_mask]
    orig_correct = orig_preds[correct_mask]
    n_correct    = len(X_correct)

    if n_correct == 0:
        # Degenerate seed -- nothing to attack.  Objective will penalise.
        return 0.0, 0.0, 0.0

    # Query counter via monkey-patch -- same technique as art_assessment.py.
    # Python finds instance attributes before class attributes, so setting
    # art_classifier.predict = wrapper intercepts all of ART's predict() calls.
    query_count = [0]

    def _counting_predict(x, **kwargs):
        query_count[0] += len(x)
        return _real_predict(x, **kwargs)

    art_classifier.predict = _counting_predict

    attack = HopSkipJump(
        classifier = art_classifier,
        targeted   = False,
        norm       = 2,
        max_iter   = max_iter,
        max_eval   = max_eval,
        init_eval  = init_eval,
        init_size  = init_size,
    )
    X_adv = attack.generate(x=X_correct)

    # Remove the instance override so the real predict is restored.
    try:
        delattr(art_classifier, "predict")
    except AttributeError:
        art_classifier.predict = _real_predict   # fallback

    adv_preds    = np.argmax(art_classifier.predict(X_adv), axis=1)
    success_mask = (adv_preds != orig_correct)

    trial_asr = asr(orig_correct, adv_preds, y_correct)
    l2_vals   = l2_perturbation(X_correct, X_adv)
    mean_l2   = float(l2_vals[success_mask].mean()) if success_mask.any() else 0.0
    avg_q     = query_count[0] / n_correct

    return trial_asr, mean_l2, avg_q


# ── 4. Optuna objective ───────────────────────────────────────────────────────
def objective(trial):
    """
    Composite score to MINIMISE.

    Primary goal : small mean L2   (gentler perturbation)
    Soft constraint 1: ASR >= 0.95  (the attack still has to work)
    Soft constraint 2: avg queries <= QUERY_BUDGET  (stay within budget)

    The penalties prevent Optuna from "cheating": e.g., returning a tiny L2
    by finding params that simply fail to move the input to the wrong class.

    Full formula:
      score = mean_L2
            + 5.0  * max(0, 0.95 - ASR)              # heavy ASR penalty
            + 0.001 * max(0, avg_queries - BUDGET)    # light query penalty
    """
    # max_eval must be sampled FIRST because init_eval's upper bound depends on it.
    # ART raises an error if init_eval > max_eval.
    max_iter  = trial.suggest_int("max_iter",  5,   30)
    max_eval  = trial.suggest_int("max_eval",  200, 1500)
    init_eval = trial.suggest_int("init_eval", 1,   min(100, max_eval))
    init_size = trial.suggest_int("init_size", 1,   100)

    # Each trial uses a different subsample (seeded by trial number) so the
    # study doesn't overfit to a single set of 30 test points.
    trial_asr, mean_l2, avg_q = run_attack(
        max_iter  = max_iter,
        max_eval  = max_eval,
        init_eval = init_eval,
        init_size = init_size,
        seed      = trial.number,
        n_points  = N_SUBSAMPLE,
    )

    # If the attack produced zero successful examples, penalise heavily.
    # We must return a finite number -- NaN or inf would crash the study.
    if trial_asr == 0.0:
        return 10.0

    asr_penalty   = 5.0   * max(0.0, 0.95 - trial_asr)
    query_penalty = 0.001 * max(0.0, avg_q - QUERY_BUDGET)

    return mean_l2 + asr_penalty + query_penalty


# ── 5. Run the study ──────────────────────────────────────────────────────────
# TPE (Tree-structured Parzen Estimator) builds a probabilistic model of which
# hyperparameter regions yield low scores and samples from them more often.
#
# NOTE: HopSkipJump is internally stochastic -- it samples random perturbation
# directions at each iteration.  TPESampler(seed=42) makes the SEARCH
# reproducible (which configs are tried and in what order), but per-trial L2
# values can vary slightly across different machines or numpy versions because
# each trial draws fresh attack randomness from the global numpy RNG.
print(f"\nStarting Optuna study  ({N_TRIALS} trials, n_jobs=1) ...")
print("Objective = mean_L2 + 5.0*(max(0, 0.95-ASR)) + 0.001*(max(0, avg_q-600))\n")

sampler = TPESampler(seed=42)
study   = optuna.create_study(direction="minimize", sampler=sampler)


def _trial_callback(study, trial):
    """Print one concise line per trial so the search is easy to follow."""
    if trial.value is None:
        return   # pruned or failed trial -- skip
    best_marker = "  <-- best" if trial.value == study.best_value else ""
    print(
        f"  Trial {trial.number:>2d}:  score={trial.value:.4f}  "
        f"max_iter={trial.params.get('max_iter'):>2}  "
        f"max_eval={trial.params.get('max_eval'):>4}  "
        f"init_eval={trial.params.get('init_eval'):>3}  "
        f"init_size={trial.params.get('init_size'):>3}"
        + best_marker
    )


study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, callbacks=[_trial_callback])

print(f"\nBest trial   : #{study.best_trial.number}")
print(f"Best score   : {study.best_value:.4f}")
print("Best params  :")
for k, v in study.best_params.items():
    print(f"  {k:12s} = {v}")

# ── 6. Baseline vs. best comparison ──────────────────────────────────────────
# Both runs use the SAME seed so they attack the exact same subset of test
# points -- the comparison is apples-to-apples, not confounded by sample luck.
COMPARE_SEED = 999

print(f"\nRunning comparison on {N_COMPARE} points (seed={COMPARE_SEED}) ...")
print("  [1/2] Baseline (Stage 1 config) ...")
b_asr, b_l2, b_q = run_attack(**BASELINE_PARAMS, seed=COMPARE_SEED, n_points=N_COMPARE)

print("  [2/2] Optuna best config ...")
o_asr, o_l2, o_q = run_attack(**study.best_params, seed=COMPARE_SEED, n_points=N_COMPARE)

print("\n" + "=" * 58)
print(f"{'config':<14} {'mean L2':>10} {'ASR':>8} {'avg queries':>12}")
print("-" * 58)
print(f"{'baseline':<14} {b_l2:>10.4f} {b_asr:>8.3f} {b_q:>12.0f}")
print(f"{'optuna-best':<14} {o_l2:>10.4f} {o_asr:>8.3f} {o_q:>12.0f}")
print("=" * 58)

# ── 7. Comparison bar chart ───────────────────────────────────────────────────
# Saved to models/ (git-ignored) because the notebook will render all figures
# inline at the final stage -- no loose PNGs in the repo root for this file.
os.makedirs("models", exist_ok=True)

configs = ["Baseline\n(Stage 1)", "Optuna Best"]
colors  = ["steelblue", "crimson"]

panel_data = [
    ([b_l2,  o_l2],  "Mean L2 perturbation",  "L2 Perturbation\n(lower = gentler attack)"),
    ([b_asr, o_asr], "Attack success rate",    "Attack Success Rate\n(higher = more effective)"),
    ([b_q,   o_q],   "Avg queries / sample",  "Query Budget\n(lower = cheaper)"),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (values, ylabel, title) in zip(axes, panel_data):
    bars = ax.bar(configs, values, color=colors, width=0.5,
                  edgecolor="black", linewidth=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    # Label each bar with its numeric value.
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

# Draw the query-budget reference line on the rightmost (query) panel.
axes[2].axhline(
    QUERY_BUDGET, color="darkorange", linestyle="--",
    linewidth=1.2, label=f"Stage 1 budget ({QUERY_BUDGET})",
)
axes[2].legend(fontsize=8)

plt.suptitle("Baseline vs. Optuna-Tuned HopSkipJump", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("models/optuna_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved  models/optuna_comparison.png")

# ── 8. Save numeric results ───────────────────────────────────────────────────
results = {
    "best_params" : study.best_params,
    "best_value"  : study.best_value,
    "n_trials"    : N_TRIALS,
    "comparison"  : {
        "baseline"   : {"asr": b_asr, "mean_l2": b_l2, "avg_queries": b_q},
        "optuna_best": {"asr": o_asr, "mean_l2": o_l2, "avg_queries": o_q},
    },
    "all_trials"  : [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials
    ],
}

with open("models/optuna_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("Saved  models/optuna_results.pkl")
print("\nDone.")
