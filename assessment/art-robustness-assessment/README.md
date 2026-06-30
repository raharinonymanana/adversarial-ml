# Lab: ART Robustness Assessment

**Module:** Assessment (NVIDIA DLI Module 4)
**Tool:** [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

This lab builds a black-box robustness assessment pipeline for a digit-classifying MLP. The target model is attacked with **HopSkipJump** — a decision-based adversarial attack that probes the model using only its predicted labels (no gradients, no probabilities), mirroring real-world threat scenarios where an attacker can query a model but cannot inspect its internals. The assessment is repeated over five random seeds and summarised as mean ± std to give an honest, variance-aware picture of the model's robustness.

**Status:** Stage 1 complete (ART baseline). Optuna hyperparameter optimisation (Stage 2) coming next.
