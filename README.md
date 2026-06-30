# AI Security Portfolio

**Course:** NVIDIA DLI — Exploring Adversarial Machine Learning
**Student:** Hasina Rindra

This repository contains hands-on labs organized by attack/assessment family,
following the course module order.

---

## Modules

| # | Folder | Topic |
|---|--------|-------|
| 1 | [`evasion/`](evasion/) | Evasion attacks — fooling models at inference time |
| 2 | [`extraction/`](extraction/) | Model extraction — stealing a model via black-box queries |
| 3 | [`assessment/`](assessment/) | Security/model assessment — ART, TextAttack, Alibi, Optuna |
| 4 | [`inversion/`](inversion/) | Model inversion — reconstructing training data |
| 5 | [`poisoning/`](poisoning/) | Data poisoning — corrupting the training pipeline |
| 6 | [`llm-attacks/`](llm-attacks/) | LLM-specific attacks — prompt injection, jailbreaks |

---

## Labs

### Evasion
- [`evasion/email-spam-evasion/`](evasion/email-spam-evasion/) — Evading a TF-IDF + Logistic Regression spam classifier (Enron dataset)

### Extraction
- [`extraction/fraud-model-extraction/`](extraction/fraud-model-extraction/) — Black-box extraction of an SVM fraud classifier via 10 k queries
- [`extraction/offline-model-extraction/`](extraction/offline-model-extraction/) — Offline model extraction: MLP victim → Random Forest clone on digits

### Assessment
- *(lab in progress)*

### Inversion
- *(no labs yet)*

### Poisoning
- *(no labs yet)*

### LLM Attacks
- *(no labs yet)*
