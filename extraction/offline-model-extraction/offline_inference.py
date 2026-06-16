"""
offline_inference.py
--------------------
THE PUNCH LINE OF LAB 3.

The previous steps stole the victim's behaviour into clone_rf.pkl. This script
proves the payoff: a stolen clone is a STANDALONE ASSET that needs no network,
no API, and no access to the victim ever again. We run it with the network
physically disabled and it still classifies digits perfectly.

How we "pull the network cable" in pure Python:
  Before loading ANYTHING, we monkeypatch Python's `socket` module so that every
  attempt to open a network connection raises RuntimeError. From that line on,
  any code that tries to phone home — to the victim's API, a license server, a
  telemetry endpoint, anything — crashes instantly. So if this script finishes,
  it is PROOF that classifying those digits required zero network access.

Deliberate design choices (read these):
  - We do NOT import train_victim or anything that could load the victim. The
    whole point is that the clone stands on its own; importing the victim would
    cheat. The only files we touch are clone_rf.pkl and test_data.pkl — plain
    local file reads.
  - The socket patch happens at the very TOP, before importing numpy or
    unpickling, so nothing can sneak a connection in first.

If any network call were attempted, this script would crash by design — and that
crash would itself be the proof the clone is (or is not) self-contained.

Usage:
  python offline_inference.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — CUT THE NETWORK  (must run before any other import or file load)
# ══════════════════════════════════════════════════════════════════════════════
import socket
import sys

_OriginalSocket = socket.socket


class _BlockedSocket(_OriginalSocket):
    """
    A socket that refuses to exist. Trying to create one (i.e. actually opening
    a network connection) raises immediately.

    Why subclass the real socket instead of using a plain function? Because the
    standard library defines `class SSLSocket(socket): ...` at import time, so
    `socket.socket` must remain a subclass-able class or importing ssl (pulled in
    transitively when we unpickle the clone) would crash for the wrong reason.
    Subclassing keeps the type machinery happy while still blocking every
    *instantiation* — which is the only thing that opens a real connection.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Network access is DISABLED in offline_inference.py. "
            "The stolen clone is meant to run with no connection at all."
        )


def _network_disabled(*args, **kwargs):
    """Stand-in for socket.create_connection: refuse all network access."""
    raise RuntimeError(
        "Network access is DISABLED in offline_inference.py. "
        "The stolen clone is meant to run with no connection at all."
    )


# Patch the two entry points that open outbound connections. `socket.socket` is
# the low-level constructor every higher-level networking call ends up using;
# `socket.create_connection` is the common convenience wrapper. After this, any
# outbound network attempt raises RuntimeError instantly.
socket.socket = _BlockedSocket
socket.create_connection = _network_disabled
print("[INFO] Network has been disabled (socket calls now raise RuntimeError).")

# ── Prove the guard is LIVE, not a no-op, before we rely on it ────────────────
# Creating a socket must now raise. We attempt exactly that: if it somehow
# SUCCEEDS, the patch silently failed and the "offline" claim below would be
# meaningless, so we stop immediately. If it raises, the guard is real and every
# line after this runs with the network genuinely cut.
try:
    socket.socket()
    sys.exit("[FAIL] guard is a no-op - socket created")
except RuntimeError:
    print("[OK] network guard live - socket creation raised\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — now (and only now) import what we need and load the clone
# ══════════════════════════════════════════════════════════════════════════════
import pickle
from pathlib import Path

import numpy as np   # numpy does no networking; safe to import after the patch

# ── Paths (local files only — no URLs anywhere in this script) ─────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
CLONE_PKL  = MODELS_DIR / "clone_rf.pkl"
TEST_PKL   = MODELS_DIR / "test_data.pkl"

# How many fresh test digits to classify in the demo.
SAMPLE_SIZE = 10


def main():
    # ── Load the stolen clone and the test digits from local disk ─────────────
    for path in (CLONE_PKL, TEST_PKL):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.\n"
                "  Run, in order:  python train_victim.py  then  python extract.py"
            )

    print("[INFO] Loading the stolen clone and test digits from local disk ...")
    with open(CLONE_PKL, "rb") as f:
        clone = pickle.load(f)
    with open(TEST_PKL, "rb") as f:
        test = pickle.load(f)
    X_test, y_test = test["X_test"], test["y_test"]

    # The clone was trained with n_jobs=-1. Force single-process prediction so we
    # don't even spin up worker processes — pure local CPU, nothing to coordinate.
    clone.n_jobs = 1

    # ── Classify a handful of fresh test digits, fully offline ────────────────
    X_sample = X_test[:SAMPLE_SIZE]
    y_sample = y_test[:SAMPLE_SIZE]
    preds = clone.predict(X_sample)

    print(f"[INFO] Classifying {SAMPLE_SIZE} fresh test digits with the clone:\n")
    print("   #   predicted   actual   match")
    print("  --   ---------   ------   -----")
    for i, (p, t) in enumerate(zip(preds, y_sample)):
        mark = "ok" if p == t else "X"
        print(f"  {i:>2}       {p}          {t}       {mark}")

    sample_acc = float((preds == y_sample).mean())
    # Full-set accuracy too, so we see the clone is fully functional offline.
    full_acc = float((clone.predict(X_test) == y_test).mean())

    print(f"\n[INFO] Accuracy on these {SAMPLE_SIZE} digits : {sample_acc*100:.1f}%")
    print(f"[INFO] Accuracy on all {len(X_test)} test digits : {full_acc*100:.2f}%")

    # ── The headline ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  [PASS] predicted with network disabled")
    print("  The stolen clone is a self-contained asset: no API, no")
    print("  victim, no internet - just a pickle file and a CPU.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
