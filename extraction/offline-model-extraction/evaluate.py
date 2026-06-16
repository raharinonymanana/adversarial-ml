"""
evaluate.py
-----------
Reports the extraction attack's results WITH uncertainty. It trains nothing and
re-runs nothing: extract.py already ran the whole attack across several random
seeds and saved the per-seed metrics, so this script just loads that file and
summarises it.

Why mean ± std instead of a single number:
  One extraction gives one number, and that number wobbles depending on which
  images were queried and how the forest grew. Reporting the mean and standard
  deviation across seeds is the honest way to say "this is roughly what the attack
  achieves, give or take this much."

Why we judge the MEAN, not a bare PASS/FAIL:
  A hard PASS/FAIL on a single run invites tuning the attack until one number
  crosses a line. Instead each metric gets one of three verdicts:
    PASS          - the mean already meets the target.
    ≈ within ±1σ  - the mean is below target, but by less than one std (the target
                    sits inside the normal run-to-run spread).
    BELOW         - the mean is below target by more than one std.

Usage:
  python evaluate.py    (run train_victim.py and extract.py first)
"""

import pickle
import sys
from pathlib import Path

# We print Unicode (± and σ). The default Windows console (cp1252) crashes on some
# of these glyphs, so switch stdout to UTF-8 before printing anything.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Success thresholds (the lab's targets) ────────────────────────────────────
ACC_TARGET   = 0.95     # clone accuracy vs ground truth
AGREE_TARGET = 0.90     # clone agreement with the victim

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
RESULTS_PKL = MODELS_DIR / "extraction_results.pkl"


def _verdict(mean, std, target):
    """Three-way verdict that refuses to be a knife-edge PASS/FAIL (see docstring)."""
    if mean >= target:
        return "PASS"
    if (target - mean) <= std:        # target lies within one std of the mean
        return "≈ within ±1σ"
    return "BELOW"


def _row(name, mean, std, target):
    """Format one table row: metric, mean ± std, target, verdict."""
    value = f"{mean*100:.2f}% ± {std*100:.2f}%"
    tgt   = f"~{target*100:.0f}%"
    return f"{name:<26}{value:>16}{tgt:>10}{_verdict(mean, std, target):>16}"


def main():
    # ── 1. Load the saved per-seed results (train nothing, re-run nothing) ─────
    if not RESULTS_PKL.exists():
        raise FileNotFoundError(
            f"{RESULTS_PKL} not found.\n"
            "  Run, in order:  python train_victim.py  then  python extract.py"
        )
    with open(RESULTS_PKL, "rb") as f:
        r = pickle.load(f)

    seeds      = r["seeds"]
    acc_mean,  acc_std  = r["accuracy_mean"],  r["accuracy_std"]
    agr_mean,  agr_std  = r["agreement_mean"], r["agreement_std"]
    victim_acc = r.get("victim_accuracy")
    n_test     = r.get("n_test", 300)
    n_queries  = r.get("n_queries")

    # ── 2. Print the report ───────────────────────────────────────────────────
    width = 68
    thick = "=" * width
    thin  = "-" * width

    print()
    print(thick)
    print("  OFFLINE MODEL EXTRACTION RESULTS - Handwritten Digits")
    print(thick)
    print(f"\n  Attack repeated over {len(seeds)} seeds : {seeds}")
    if n_queries:
        print(f"  Budget per run            : {n_queries:,} black-box victim queries")
    if victim_acc is not None:
        print(f"  Victim (MLP) test accuracy: {victim_acc*100:.2f}%  (fixed across all seeds)")
    print("  (Only the attack varies between seeds; the victim and test set do not.)\n")

    print(f"{'Metric':<26}{'mean ± std':>16}{'Target':>10}{'Verdict':>16}")
    print(thin)
    print(_row("Clone accuracy (vs truth)",   acc_mean, acc_std, ACC_TARGET))
    print(_row("Clone agreement (vs victim)", agr_mean, agr_std, AGREE_TARGET))
    print(thin)

    # ── 3. The two distinct noise sources, spelled out ────────────────────────
    print(f"\n  Measurement floor: n={n_test} test set => ~±2% binomial CI; per-seed")
    print("  std above captures attack variance, a separate source.")

    print("\n  Legend")
    print("  ------")
    print("  Accuracy   - clone's correctness vs the TRUE digit labels")
    print("  Agreement  - % of test images where the clone matches the VICTIM")
    print("  Verdict    - PASS / within one std of target / clearly BELOW")
    print()


if __name__ == "__main__":
    main()
