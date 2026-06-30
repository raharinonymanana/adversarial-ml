"""
train_target.py — Train and save the MLP digit classifier (the model under assessment).

Run this ONCE before art_assessment.py.

Why we save both the model AND the test split:
  art_assessment.py must evaluate on the exact same held-out samples that
  were never seen during training.  Saving the split here guarantees that —
  no matter when art_assessment.py is run, it loads the same test set.

Output (saved to models/, which is git-ignored because it contains .pkl files):
  models/target_model.pkl  — the trained MLPClassifier
  models/test_split.pkl    — {"X_test": ..., "y_test": ...}
"""

import os
import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ── 1. Load and normalise the digits dataset ──────────────────────────────────
# sklearn's digits: 1797 samples, 64 features each (8×8 pixels), 10 classes.
# Raw pixel values are integers 0–16.  Dividing by 16.0 maps them into [0, 1].
# This normalisation matters because ART enforces clip_values=(0.0, 1.0) —
# any adversarial example that drifts outside that range gets clipped back.
digits = load_digits()
X = digits.data / 16.0   # shape (1797, 64), values now in [0, 1]
y = digits.target         # shape (1797,), classes 0–9

# ── 2. Train / test split ─────────────────────────────────────────────────────
# random_state=42 makes this split fully deterministic — every run of this
# script produces the exact same X_train / X_test partition.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")

# ── 3. Train a small MLP ──────────────────────────────────────────────────────
# Two hidden layers (64 → 32 neurons).
# Chosen to be fast on CPU while still reaching ~97 % test accuracy.
# random_state fixes the weight initialisation for reproducibility.
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42,
)
model.fit(X_train, y_train)

# ── 4. Report accuracy ────────────────────────────────────────────────────────
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
print(f"\nTrain accuracy : {train_acc:.4f}")
print(f"Test  accuracy : {test_acc:.4f}")

# ── 5. Save to models/ ────────────────────────────────────────────────────────
# models/ is listed in the root .gitignore — these binary files stay local.
os.makedirs("models", exist_ok=True)

with open("models/target_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/test_split.pkl", "wb") as f:
    # Save as a dict so the loader can use named keys — harder to mix up.
    pickle.dump({"X_test": X_test, "y_test": y_test}, f)

print("\nSaved  models/target_model.pkl")
print("Saved  models/test_split.pkl")
