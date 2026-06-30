"""
metrics.py — shared evaluation functions for the ART robustness assessment lab.

Defined ONCE here and imported by every other script in this lab.
That way there is only one place to read (and fix) each metric definition.
"""

import numpy as np


def asr(orig_preds, adv_preds, y_true):
    """
    Attack Success Rate.

    Among the samples the model originally classified CORRECTLY,
    what fraction of predictions CHANGED after the attack?

    Why we filter first: attacking an already-wrong prediction proves
    nothing about robustness — the model was already failing.

    Parameters
    ----------
    orig_preds : array-like (n,)   — clean predictions
    adv_preds  : array-like (n,)   — predictions on adversarial inputs
    y_true     : array-like (n,)   — ground-truth labels

    Returns
    -------
    float in [0, 1].  Higher = attack is more effective.
    """
    orig_preds = np.asarray(orig_preds)
    adv_preds  = np.asarray(adv_preds)
    y_true     = np.asarray(y_true)

    # Only count samples the model originally got right.
    correct_mask = (orig_preds == y_true)
    n_correct    = correct_mask.sum()

    if n_correct == 0:
        return 0.0   # nothing meaningful to measure

    # Among those correct samples, did the prediction flip?
    flipped = (adv_preds[correct_mask] != orig_preds[correct_mask])
    return float(flipped.mean())


def l2_perturbation(X, X_adv):
    """
    Per-sample L2 norm of the adversarial perturbation  ||X_adv - X||_2.

    Smaller L2 = the attacker changed the input less = stealthier attack.
    Standard measure of perturbation size for L2-norm attacks.

    Parameters
    ----------
    X     : array-like (n, d)   — original inputs
    X_adv : array-like (n, d)   — adversarial inputs

    Returns
    -------
    np.ndarray of shape (n,) — one L2 distance per sample.
    """
    X     = np.asarray(X,     dtype=float)
    X_adv = np.asarray(X_adv, dtype=float)

    # Flatten each sample to 1-D before computing the norm.
    # This works whether inputs are already flat (64,) or shaped (8, 8).
    diff = (X_adv - X).reshape(len(X), -1)
    return np.linalg.norm(diff, axis=1)


def robustness_curve(l2_per_sample, success_mask, eps_grid):
    """
    Robustness curve: fraction of originally-correct samples fooled
    as a function of the L2 perturbation budget ε.

    Interpretation: "If the attacker is limited to changes of magnitude
    ≤ ε, what fraction of correctly-classified test samples can they fool?"

    The curve naturally rises from 0 (no budget) to the overall ASR
    (unlimited budget), making it easy to see how much perturbation is
    needed to reach any level of attack effectiveness.

    Parameters
    ----------
    l2_per_sample : ndarray (n,)       — L2 perturbation per originally-correct sample
    success_mask  : bool ndarray (n,)  — True where attack flipped the prediction
    eps_grid      : ndarray (k,)       — budget values, e.g. np.linspace(0, 5, 50)

    Returns
    -------
    eps_grid        : same ndarray as input (returned for convenience in plt.plot)
    asr_at_each_eps : ndarray (k,)
    """
    n = len(l2_per_sample)
    if n == 0:
        return eps_grid, np.zeros(len(eps_grid))

    l2_per_sample = np.asarray(l2_per_sample)
    success_mask  = np.asarray(success_mask, dtype=bool)

    # For each budget value, count samples that satisfy BOTH:
    #   1. the attack actually flipped the prediction   (success_mask)
    #   2. the perturbation used was within the budget  (L2 ≤ ε)
    # Divide by total originally-correct samples (not just successes)
    # so the y-axis stays interpretable as a fraction of all attackable samples.
    asr_at_each_eps = np.array([
        ((success_mask) & (l2_per_sample <= eps)).sum() / n
        for eps in eps_grid
    ])

    return eps_grid, asr_at_each_eps
