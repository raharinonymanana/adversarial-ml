# Lab: ART Robustness Assessment

**Module:** Assessment (NVIDIA DLI Module 4)  
**Tools:** [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) · [Optuna](https://optuna.org)

This lab builds a black-box robustness assessment pipeline for a digit-classifying MLP.
**Module 4 is not a sixth attack family** -- it is the methodology and tooling layer that
teaches standard practice for evaluating any ML system. Two tools are combined:

- **ART HopSkipJump** -- a decision-based attack that probes the model using only its
  predicted labels (no gradients, no probabilities), mirroring real-world black-box access.
- **Optuna TPE search** -- frames the attacker's goal as an optimisation problem: find the
  HopSkipJump configuration that achieves the highest attack effectiveness at the lowest
  query cost.

The assessment is repeated over five random seeds and summarised as mean +/- std
(multi-seed honesty) to give a variance-aware picture of robustness.

---

## Results

| Config | Mean L2 | ASR | Avg queries |
|--------|---------|-----|-------------|
| Baseline (Stage 1, 5-seed mean) | 0.7196 +/- 0.0544 | 1.000 +/- 0.000 | 580 +/- 0 |
| Optuna-best (matched comparison, seed=999) | 0.7302 | 1.000 | 413 |

**Verdict: Query-efficiency gain, no L2 improvement.**

Optuna's 30-trial TPE search did not find a configuration that reduces the mean L2
perturbation below the Stage 1 baseline. The baseline was already near-optimal for this
model geometry. What the search DID reveal is that the same 100% ASR can be achieved
with ~29% fewer model queries (581 -> 413) -- demonstrating the precision-vs-query-cost
trade-off inherent to HopSkipJump. None of the 30 trials hit the 600-query budget ceiling,
so the penalty term never actively constrained the search.

---

## Honest Limitation

The digits dataset uses 8x8 low-resolution images. Several adversarial examples in the
image grid are visibly distorted to a human eye. The "imperceptibility" property that makes
adversarial attacks alarming in real-world systems does not fully hold at this resolution.
This lab validates the **assessment methodology and tooling** -- the same pipeline applies
to any sklearn-compatible model and any dataset.

---

## Files

| File | Purpose |
|------|---------|
| `train_target.py` | Train and save the MLP digit classifier |
| `art_assessment.py` | 5-seed black-box robustness assessment (ART HopSkipJump) |
| `optuna_optimize.py` | 30-trial Optuna hyperparameter search over HopSkipJump settings |
| `metrics.py` | Shared metrics: `asr()`, `l2_perturbation()`, `robustness_curve()` |
| `requirements.txt` | Python dependencies |
| `notebooks/assessment_exploration.ipynb` | Full end-to-end notebook report with inline plots |

**Run order:** `train_target.py` -> `art_assessment.py` -> `optuna_optimize.py`  
(or run the notebook end-to-end for all steps in one place)
