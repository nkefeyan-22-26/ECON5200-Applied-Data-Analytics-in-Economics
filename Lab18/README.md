# Fraud Detection Model Evaluation — Metrics that Matter

## Objective
Rigorously evaluate a logistic regression classifier on a severely imbalanced real-world dataset, demonstrating why accuracy is a misleading performance metric for rare-event detection and identifying the business-optimal classification threshold under operational constraints.

---

## Methodology

- **Data:** Kaggle Credit Card Fraud Detection dataset — 284,807 European credit card transactions (492 frauds, 0.172% positive class rate) with PCA-anonymized features V1–V28 and transaction Amount
- **Baseline audit:** Established the accuracy paradox by showing a naïve all-negative classifier achieves 99.83% accuracy while capturing zero fraud — motivating the shift to class-specific metrics
- **Model training:** Trained a logistic regression classifier with stratified train/test split to preserve class proportions; scaled the Amount feature using StandardScaler
- **Evaluation framework:** Computed confusion matrix, Precision, Recall, and F1-Score at the default threshold (τ = 0.50); generated the full classification report decomposed by class
- **Threshold-invariant metrics:** Plotted ROC and Precision-Recall curves across all operating points; compared ROC-AUC and PR-AUC to assess discrimination quality under class imbalance
- **Threshold analysis:** Swept τ from 0.01 to 0.99 to map the precision-recall tradeoff and identify the F1-maximizing operating point
- **Business constraint application:** Applied a capacity constraint of 500 maximum daily fraud investigations to select a deployment-ready threshold aligned with operational reality

---

## Key Findings

- The accuracy paradox is empirically stark: 99.83% accuracy is achievable with a model that detects no fraud whatsoever, confirming that accuracy is not a valid performance measure for imbalanced classification problems
- Logistic regression achieved strong discriminative ability (ROC-AUC = 0.956), though PR-AUC (0.742) provides the more honest assessment — directly measuring minority-class precision across thresholds without being inflated by the abundance of true negatives
- The F1-optimal threshold (τ ≈ 0.15) diverges substantially from the scikit-learn default (τ = 0.50), illustrating that threshold calibration is a non-trivial modeling decision with material consequences for fraud capture rates
- Under the 500-investigation capacity constraint, the selected threshold (τ = 0.01) flagged 246 transactions — well within budget — while achieving 88.8% Recall, meaning fewer than 1 in 9 fraud cases went undetected
- The gap between ROC-AUC and PR-AUC highlights a general principle: high ROC-AUC is necessary but not sufficient evidence of strong minority-class performance in imbalanced settings

---

## Tech Stack
`Python` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn`
