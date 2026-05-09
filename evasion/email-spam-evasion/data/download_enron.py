"""
download_enron.py
-----------------
Downloads and preprocesses the Enron spam dataset.

The Enron-Spam dataset contains ~33,000 real emails from Enron employees,
labeled as ham (legitimate) or spam. It is a standard academic benchmark
for spam detection, originally published by Metsis et al. (2006).

We try two methods in order:
  1. The `enron-spam` pip package (fastest if installed)
  2. Direct download of the 6 preprocessed tar.gz archives from the
     Ioannis Androutsopoulos / AUEB research server

Output: data/emails.csv
  - Column "label": 0 = ham, 1 = spam
  - Column "text":  email body text (headers stripped)

Usage:
  python data/download_enron.py
"""

import csv
import io
import sys
import tarfile
from pathlib import Path

import requests

# ── Where we will write the final CSV ─────────────────────────────────────────
# Path(__file__).parent  → the data/ folder this script lives in
DATA_DIR   = Path(__file__).parent
OUTPUT_CSV = DATA_DIR / "emails.csv"

# ── Source URLs ────────────────────────────────────────────────────────────────
# The six pre-processed Enron folders from the original research page.
# Each .tar.gz contains a ham/ and spam/ subdirectory with .txt files.
AUEB_BASE = (
    "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/"
)
ENRON_ARCHIVES = [f"enron{i}.tar.gz" for i in range(1, 7)]


# ══════════════════════════════════════════════════════════════════════════════
# Method 1: enron-spam pip package
# ══════════════════════════════════════════════════════════════════════════════

def _try_package():
    """
    Attempt to load the dataset via the `enron-spam` pip package.
    Returns (texts, labels) on success, or None if the package is
    unavailable or returns unexpected data.
    """
    try:
        import enron_spam  # pip install enron-spam
    except ImportError:
        print("[INFO] enron-spam package not installed — trying direct download.")
        return None

    print("[INFO] enron-spam package found. Fetching dataset …")
    try:
        df = enron_spam.fetch_enron_dataset()
    except Exception as exc:
        print(f"[WARN] enron-spam package raised an error: {exc}")
        return None

    # The package returns a DataFrame; column names differ by version.
    # We probe for the text column and the label column defensively.
    text_col = next(
        (c for c in df.columns if c.lower() in ("message", "text", "email", "body")),
        df.columns[0],   # fall back to the first column
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ("spam/ham", "label", "class", "spam")),
        df.columns[1],   # fall back to the second column
    )

    texts = df[text_col].astype(str).tolist()

    # Labels may arrive as strings ("spam"/"ham") or integers (1/0).
    labels = [
        1 if str(v).strip().lower() in ("1", "spam", "true") else 0
        for v in df[label_col]
    ]
    return texts, labels


# ══════════════════════════════════════════════════════════════════════════════
# Method 2: Download directly from the AUEB research server
# ══════════════════════════════════════════════════════════════════════════════

def _download_archive(archive_name):
    """
    Download one enronN.tar.gz archive from the AUEB server and extract
    (text, label) pairs from every .txt file inside it.

    Inside each archive the layout is:
        enronN/
          ham/   *.txt   ← legitimate emails
          spam/  *.txt   ← spam emails
    """
    url = AUEB_BASE + archive_name
    print(f"  Downloading {archive_name} …", end=" ", flush=True)

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"FAILED  ({exc})")
        return [], []

    texts, labels = [], []

    # Open the tar archive entirely in memory — no temp files needed.
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            # Determine the label from the parent directory name.
            parts = Path(member.name).parts
            if "ham" in parts:
                label = 0
            elif "spam" in parts:
                label = 1
            else:
                continue   # ignore files outside ham/ and spam/

            fobj = tar.extractfile(member)
            if fobj is None:
                continue

            try:
                raw = fobj.read().decode("utf-8", errors="replace")
            except Exception:
                continue

            # Strip the email headers: everything up to and including
            # the "Subject:" line.  We only want the body text.
            lines = raw.splitlines()
            body_start = 0
            for idx, line in enumerate(lines):
                if line.lower().startswith("subject:"):
                    body_start = idx + 1
                    break
            body = " ".join(lines[body_start:]).strip()

            if body:   # skip empty bodies
                texts.append(body)
                labels.append(label)

    ham_n  = labels.count(0)
    spam_n = labels.count(1)
    print(f"OK  ({ham_n} ham, {spam_n} spam)")
    return texts, labels


def _download_all():
    """Download and merge all six Enron archives."""
    all_texts, all_labels = [], []
    for archive in ENRON_ARCHIVES:
        t, l = _download_archive(archive)
        all_texts.extend(t)
        all_labels.extend(l)
    return all_texts, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# Save to CSV
# ══════════════════════════════════════════════════════════════════════════════

def _save_csv(texts, labels, path):
    """Write texts and labels to a two-column CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])   # header
        for label, text in zip(labels, texts):
            writer.writerow([label, text])
    print(f"[INFO] Saved {len(texts):,} emails → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if OUTPUT_CSV.exists():
        print(f"[INFO] {OUTPUT_CSV} already exists.  Delete it to re-download.")
        return

    print("=" * 60)
    print("  Enron Spam Dataset Downloader")
    print("=" * 60)

    # Try the pip package first (no HTTP request needed)
    result = _try_package()

    if result is None:
        # Fall back to direct download
        print("[INFO] Downloading six archives from AUEB server …")
        texts, labels = _download_all()
    else:
        texts, labels = result

    if not texts:
        print(
            "\n[ERROR] No data was loaded.\n"
            "  • Check your internet connection.\n"
            "  • Or install the package:  pip install enron-spam\n"
        )
        sys.exit(1)

    ham_count  = labels.count(0)
    spam_count = labels.count(1)
    print(f"\n[INFO] Loaded {len(texts):,} emails total")
    print(f"       {ham_count:,} ham  |  {spam_count:,} spam")

    _save_csv(texts, labels, OUTPUT_CSV)
    print("\n[DONE] Next step: python train.py")


if __name__ == "__main__":
    main()
