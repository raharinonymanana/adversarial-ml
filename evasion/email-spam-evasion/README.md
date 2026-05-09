# Lab: Adversarial Evasion of a Spam Email Classifier

**Course:** NVIDIA DLI — Exploring Adversarial Machine Learning  
**Category:** Evasion attacks  
**Model:** Logistic Regression on TF-IDF features  
**Dataset:** Spam Email Classification Dataset by purusinghvi on Kaggle (83,446 emails)  
**Hardware:** CPU-only, no GPU required

---

## What This Lab Demonstrates

A spam filter based on TF-IDF (bag-of-words statistics) + Logistic Regression
can achieve ~95–98% accuracy on clean data. But it has a fundamental weakness:
it only knows **exact tokens**. If you change the spelling of a word, the model
loses all the information that word carried.

This lab implements three **evasion attacks** that exploit this weakness:

| Attack | Core idea |
|--------|-----------|
| **Character Obfuscation** | Replace letters in spam words with lookalikes (`free` → `fr€€`) |
| **Synonym Substitution** | Swap spam words for semantically equivalent but less-suspicious synonyms |
| **Ham Word Injection** | Append a large block of legitimate corporate text to dilute the spam signal |

---

## Folder Structure

```
email-spam-evasion/
├── data/
│   └── emails.csv          ← manually placed here (see Quickstart step 1)
├── train.py                ← Step 1: train the classifier
├── attack.py               ← attack logic (also runnable as a demo)
├── evaluate.py             ← Step 2: measure attack effectiveness
├── requirements.txt
└── README.md
```

After running the pipeline, these files are also created:
```
├── model.pkl               ← trained LogisticRegression
├── vectorizer.pkl          ← fitted TfidfVectorizer
├── top_spam_words.json     ← top spam words with LR coefficients
└── test_data.pkl           ← held-out test set (raw texts + labels)
```

---

## Quickstart

### 1. Get the dataset

1. Go to: https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset
2. Download `combined_data.csv`
3. Rename it to `emails.csv`
4. Place it in the `data/` folder

The file should be at `data/emails.csv` (83,446 rows, two columns: `label` and `text`).

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train the classifier

```powershell
python train.py
```

Expected output: **~95–98% accuracy** on the held-out test set.

### 5. Run the evaluation

```powershell
python evaluate.py
```

Prints a table like:

```
======================================================================
  ADVERSARIAL EVASION RESULTS — Spam Email Classifier
======================================================================

  Baseline spam detection rate : 98.2%  (1843 emails correctly flagged)

Attack                          Tested    Evaded   Post-Atk Det%    Evasion%
----------------------------------------------------------------------
1. Character Obfuscation          1843       712           61.3%  ▲   38.7%
2. Synonym Substitution           1843       531           71.2%  ►   28.8%
3. Ham Word Injection             1843      1201           34.8%  ▲   65.2%
----------------------------------------------------------------------
```

*(Exact numbers will vary with your dataset split.)*

### 6. See a before/after demo

```powershell
python attack.py
```

---

## How Each Attack Works

### Attack 1 — Character Obfuscation

**What it does:** identifies the top spam words from the trained model's
coefficients, then replaces letters in those words with Unicode lookalikes:

```
e → €    i → 1    o → 0    a → @    s → $    g → 9
```

**Why it works:** TF-IDF is a bag-of-words model. It knows the token `free`
(coefficient ≈ +3.8). It has never seen `fr€€` — so that token gets weight 0.

**Limitation:** a robust normalisation step (strip non-ASCII → re-classify)
would defeat this attack instantly.

---

### Attack 2 — Synonym Substitution

**What it does:** replaces ~50 high-weight spam words with synonyms that
appear more frequently in ham emails:

| Spam word | Synonym used |
|-----------|-------------|
| `free` | `complimentary` |
| `click here` | `follow this link` |
| `guaranteed` | `assured` |
| `earn` | `generate` |

**Why it works:** `complimentary` has a much lower LR coefficient than `free`
because it appears more in corporate ham emails. The message reads identically
to a human but looks very different to the model.

---

### Attack 3 — Ham Word Injection

**What it does:** appends ~200 words of corporate/academic boilerplate to the
end of every spam email: *meeting, agenda, stakeholder, sprint retrospective…*

**Why it works:** TF-IDF represents a document as a weighted sum of all its
token scores. Adding many tokens with negative (ham) coefficients drags the
total score below the decision boundary. Think of it as:

```
score = (few strong spam tokens) + (many weak ham tokens) → net: ham
```

**Limitation:** a length-aware or density-based model would be harder to fool
this way.

---

## Concepts to Review

| Concept | What to read |
|---------|-------------|
| TF-IDF | `train.py` — `TfidfVectorizer` parameters and comments |
| Logistic Regression coefficients | `train.py` — `get_top_words()` function |
| Token-level weakness | `attack.py` — `attack_char_obfuscation()` |
| Decision boundary dilution | `attack.py` — `attack_ham_injection()` |
| Evasion rate calculation | `evaluate.py` — `_run_attack()` |

---

## Defence Ideas (for further study)

- **Character normalisation** — convert `€` → `e`, `0` → `o` before tokenising
- **Semantic embeddings** — word2vec or BERT see `free` and `complimentary`
  as nearby in meaning; harder to fool with synonyms
- **Adversarial training** — include attacked emails in the training set
- **Ensemble models** — combine TF-IDF with character n-grams and URL features
- **Injection detection** — flag emails with unusually long bodies or
  mismatched tone between sections

---

## References

- Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). *Spam filtering with
  naive Bayes — which naive Bayes?* CEAS 2006.
- Dalvi, N. et al. (2004). *Adversarial classification.* KDD 2004.
- Lowd, D., & Meek, C. (2005). *Adversarial learning.* KDD 2005.
