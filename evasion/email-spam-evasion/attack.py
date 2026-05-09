"""
attack.py
---------
Three adversarial evasion attacks on the TF-IDF + Logistic Regression
spam classifier.

Background — what is an "evasion attack"?
  The attacker (spammer) knows the general type of filter being used
  (keyword-based or statistical) and modifies spam emails so they
  look different to the classifier while still being understood by a
  human recipient.  This is the adversarial ML equivalent of a burglar
  wearing a disguise.

All three attacks exploit the same fundamental weakness of TF-IDF:
  It only knows EXACT tokens.  Change the token, lose the signal.

Attacks:
  1. Character Obfuscation  — replace letters in spam words with lookalikes
     ("free" → "fr€€").  TF-IDF sees an unknown token with zero weight.

  2. Synonym Substitution   — swap spam words for synonyms that appear
     more in ham corpora ("free" → "complimentary").  The meaning is
     preserved but the learned coefficient drops close to zero.

  3. Ham Word Injection     — append a large block of legitimate-sounding
     corporate text.  The spam tokens are diluted by many ham-weighted
     tokens, pushing the average score below the decision boundary.

Each attack function has the same signature:
    attack_fn(texts: list[str], top_spam_words: list[str]) -> list[str]

Usage (standalone demo):
    python attack.py
"""

import json
import re
from pathlib import Path

BASE_DIR   = Path(__file__).parent
WORDS_JSON = BASE_DIR / "top_spam_words.json"


# ══════════════════════════════════════════════════════════════════════════════
# Attack 1 — Character Obfuscation
# ══════════════════════════════════════════════════════════════════════════════

# Each entry maps one ASCII letter to a visually similar Unicode character.
# A human reads "fr€€" as "free"; the tokeniser sees a brand-new string.
CHAR_MAP = {
    "a": "@",   # a → at-sign
    "e": "€",   # e → Euro sign
    "g": "9",   # g → digit nine
    "i": "1",   # i → digit one
    "l": "1",   # l → digit one  (same lookalike — intentional)
    "o": "0",   # o → digit zero
    "s": "$",   # s → dollar sign
}


def _obfuscate_word(word: str) -> str:
    """
    Replace every character in `word` that has a lookalike in CHAR_MAP.
    Non-mapped characters (uppercase, digits, punctuation) are kept as-is.
    """
    return "".join(CHAR_MAP.get(ch.lower(), ch) for ch in word)


def attack_char_obfuscation(texts: list, top_spam_words: list) -> list:
    """
    Find every top-spam word in each email and replace its characters
    with lookalikes so TF-IDF cannot recognise it.

    Why only single-word spam features?
    TF-IDF with ngram_range=(1,2) may produce bigram features like
    "click here".  We cannot obfuscate a phrase token-by-token the same
    way, so we restrict this attack to single-word entries only.

    Args:
        texts          — list of raw email strings
        top_spam_words — high-coefficient words exported by train.py
    Returns:
        list of modified email strings (same length as `texts`)
    """
    # Keep only single-word spam features (no spaces / no bigrams)
    single_word_spam = {w for w in top_spam_words if " " not in w}

    attacked = []
    for text in texts:
        # re.split(r"(\W+)", text) splits on non-word characters but
        # keeps the delimiters (spaces, punctuation) as separate items
        # so we can join everything back without losing formatting.
        tokens = re.split(r"(\W+)", text)
        new_tokens = []
        for token in tokens:
            if token.lower() in single_word_spam:
                new_tokens.append(_obfuscate_word(token))
            else:
                new_tokens.append(token)
        attacked.append("".join(new_tokens))

    return attacked


# ══════════════════════════════════════════════════════════════════════════════
# Attack 2 — Synonym Substitution
# ══════════════════════════════════════════════════════════════════════════════

# Keys: spam-heavy words / phrases that appear in the TF-IDF vocabulary
# Values: semantically equivalent alternatives that appear mostly in ham
#
# Selection criteria:
#   • The synonym carries the same human-readable meaning
#   • The synonym is less frequent in spam corpora (lower LR coefficient)
#   • Multi-word entries are listed FIRST so longer patterns match before
#     shorter sub-patterns (e.g. "click here" before "click")
SYNONYM_MAP = {
    # ── Multi-word phrases first ───────────────────────────────────────────
    "click here":      "follow this link",
    "act now":         "respond when convenient",
    "limited time":    "time-bound",
    "risk-free":       "no-risk",
    "weight loss":     "health improvement",
    "click below":     "see below",
    "opt in":          "sign up",
    "double your":     "increase your",
    "earn money":      "generate income",

    # ── Financial / money ──────────────────────────────────────────────────
    "free":            "complimentary",
    "money":           "funds",
    "cash":            "currency",
    "earn":            "generate",
    "income":          "revenue",
    "profit":          "gain",
    "rich":            "financially successful",
    "prize":           "award",
    "bonus":           "incentive",
    "discount":        "price reduction",
    "cheap":           "cost-effective",
    "investment":      "allocation",
    "savings":         "reserves",
    "dollar":          "usd",
    "million":         "1 000 000",
    "billion":         "1 000 000 000",
    "loan":            "credit facility",
    "debt":            "outstanding balance",
    "mortgage":        "home financing",
    "credit":          "financial standing",

    # ── Action words ───────────────────────────────────────────────────────
    "buy":             "acquire",
    "click":           "visit",
    "order":           "request",
    "purchase":        "obtain",
    "subscribe":       "register",
    "offer":           "opportunity",
    "deal":            "arrangement",
    "claim":           "redeem",
    "verify":          "confirm",
    "transfer":        "remittance",

    # ── Marketing superlatives ─────────────────────────────────────────────
    "guaranteed":      "assured",
    "urgent":          "time-sensitive",
    "winner":          "recipient",
    "selected":        "chosen",
    "congratulations": "well done",
    "exclusive":       "specialized",
    "special":         "particular",
    "amazing":         "notable",
    "incredible":      "remarkable",
    "limited":         "restricted",
    "unsubscribe":     "opt out",

    # ── Specific spam topics ───────────────────────────────────────────────
    "casino":          "gaming establishment",
    "pills":           "supplements",
    "drugs":           "medication",
    "pharmacy":        "dispensary",
    "prescription":    "medical order",
    "nigeria":         "west africa",
    "prince":          "dignitary",
    "bank account":    "financial profile",
    "password":        "credentials",
    "adult":           "mature content",
    "sex":             "intimate",
}


def attack_synonym_substitution(texts: list, top_spam_words: list = None) -> list:
    """
    Replace spam-indicative words and phrases with lower-suspicion synonyms.

    `top_spam_words` is accepted for API consistency but not used here —
    the synonym dictionary is hardcoded because it was built from corpus
    analysis and domain knowledge, not from this specific model run.

    Implementation note: we sort SYNONYM_MAP keys by length (longest first)
    so that "click here" is replaced before "click".  Without this ordering,
    "click here" could become "visit here" (partial match of "click").

    Args:
        texts          — list of raw email strings
        top_spam_words — ignored (kept for uniform API with other attacks)
    Returns:
        list of modified email strings
    """
    # Pre-sort keys: longest phrase first prevents partial-match collisions
    sorted_keys = sorted(SYNONYM_MAP.keys(), key=len, reverse=True)

    attacked = []
    for text in texts:
        new_text = text
        for spam_phrase in sorted_keys:
            synonym = SYNONYM_MAP[spam_phrase]
            # \b = word boundary — ensures we match whole words, not substrings
            # re.IGNORECASE so "FREE", "Free", and "free" all match
            pattern = r"\b" + re.escape(spam_phrase) + r"\b"
            new_text = re.sub(pattern, synonym, new_text, flags=re.IGNORECASE)
        attacked.append(new_text)

    return attacked


# ══════════════════════════════════════════════════════════════════════════════
# Attack 3 — Ham Word Injection
# ══════════════════════════════════════════════════════════════════════════════

# A block of legitimate corporate / academic language.
# Every word here appears frequently in ham emails and has a low (or negative)
# coefficient in the trained logistic regression model.
#
# Why does this work?
#   TF-IDF represents a document as a weighted average over all its tokens.
#   Adding many tokens with near-zero or negative weights drags the final
#   score toward the ham side of the decision boundary.
#   Think of it as: (few strong spam signals) + (many weak ham signals) → net ham.
_HAM_BLOCK = """
meeting agenda attached please review regards thanks project update team
collaboration conference call quarterly review budget planning performance
metrics stakeholder presentation action items follow up best regards sincerely
office hours schedule calendar appointment reminder training session onboarding
documentation technical support customer service feedback survey policy
compliance annual report board meeting corporate governance risk management
human resources payroll benefits enrollment open enrollment workflow process
improvement lean methodology agile sprint retrospective product roadmap feature
request user story acceptance criteria release notes deployment checklist server
maintenance downtime window data backup recovery plan disaster recovery business
continuity security audit penetration testing vulnerability assessment patch
legal counsel contract negotiation terms conditions privacy policy employee
handbook code conduct workplace safety incident report expense report
reimbursement travel itinerary hotel confirmation conference registration abstract
submission peer review academic research weekly standup daily sync team building
offsite retrospective planning session sprint review backlog grooming refinement
engineering leadership architecture design review code review pull request merge
integration testing unit testing regression testing quality assurance continuous
integration continuous deployment pipeline infrastructure monitoring logging
alerting on-call rotation post-mortem root cause analysis corrective action
"""

# Repeat the block twice to make the ham signal strong enough to overcome
# a high-confidence spam prediction.  More repetitions = stronger dilution.
HAM_INJECTION_TEXT = (_HAM_BLOCK.strip() + " ") * 2


def attack_ham_injection(texts: list, top_spam_words: list = None) -> list:
    """
    Append a large block of ham-like vocabulary to the end of each spam email.

    The appended text looks like noise to a human but floods the TF-IDF
    vector with tokens that pull the classification score toward ham.

    Args:
        texts          — list of raw email strings
        top_spam_words — ignored (kept for uniform API)
    Returns:
        list of modified email strings
    """
    attacked = []
    for text in texts:
        # Separator makes it clear (to a human reader) where the injection starts
        new_text = text + "\n\n" + HAM_INJECTION_TEXT
        attacked.append(new_text)
    return attacked


# ══════════════════════════════════════════════════════════════════════════════
# Attack registry — evaluate.py iterates over this dict
# ══════════════════════════════════════════════════════════════════════════════

# Maps a short key → attack function.  The functions share a common signature
# so evaluate.py can call any of them in a loop without special-casing.
ATTACKS = {
    "char_obfuscation":     attack_char_obfuscation,
    "synonym_substitution": attack_synonym_substitution,
    "ham_injection":        attack_ham_injection,
}

# Human-readable display names for the results table
ATTACK_NAMES = {
    "char_obfuscation":     "1. Character Obfuscation",
    "synonym_substitution": "2. Synonym Substitution",
    "ham_injection":        "3. Ham Word Injection",
}


def load_top_spam_words() -> list:
    """Load the top spam words JSON written by train.py."""
    if not WORDS_JSON.exists():
        raise FileNotFoundError(
            f"{WORDS_JSON} not found.\n"
            "  Run:  python train.py   first."
        )
    with open(WORDS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data["top_spam_words"]


# ══════════════════════════════════════════════════════════════════════════════
# Standalone demo — shows a before/after for each attack on one example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    top_spam_words = load_top_spam_words()

    sample = (
        "CONGRATULATIONS! You have been selected as a winner! "
        "Click here to claim your FREE prize money. "
        "This is a limited time offer — act now! "
        "Guaranteed income of $1000 per week. "
        "Buy our exclusive package and earn cash today! "
        "Unsubscribe at any time."
    )

    print("=" * 70)
    print("  Attack Demo — same spam email, three disguises")
    print("=" * 70)

    print("\n[ORIGINAL]")
    print(sample)

    result = attack_char_obfuscation([sample], top_spam_words)
    print("\n[ATTACK 1 — Character Obfuscation]")
    print(result[0])

    result = attack_synonym_substitution([sample])
    print("\n[ATTACK 2 — Synonym Substitution]")
    print(result[0])

    result = attack_ham_injection([sample])
    print("\n[ATTACK 3 — Ham Injection]  (showing first 300 chars)")
    print(result[0][:300], "…")
