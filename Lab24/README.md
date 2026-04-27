# Causal ML — Double Machine Learning & Causal Forests for Policy Evaluation

**ECON 5200: Causal Machine Learning & Applied Analytics | Lab 24**

---

## Objective

Diagnose and repair a broken Double Machine Learning pipeline, then apply DML and Causal Forest methods to estimate the causal effect of 401(k) plan eligibility on household net financial assets using the Chernozhukov & Hansen (2004) dataset.

---

## Problem Statement

Standard regression conflates confounding with causal effects when treatment assignment is non-random and covariates are high-dimensional.  Double Machine Learning addresses this by partialling out confounders for *both* the outcome and the treatment before estimating the causal coefficient — producing √n-consistent, asymptotically normal estimates even when nuisance functions are estimated with flexible ML models.

---

## Research Design

| Item | Detail |
|------|--------|
| **Dataset** | 401(k) Pension Data (Chernozhukov & Hansen, 2004) — 9,915 observations |
| **Outcome (Y)** | `net_tfa` — net total financial assets ($) |
| **Treatment (D)** | `e401` — binary 401(k) plan eligibility |
| **Identification** | Double ML / Partially Linear Regression (PLR) under unconfoundedness |
| **Nuisance Learners** | Random Forest (200 trees, max depth 5) for both E[Y\|X] and E[D\|X] |
| **Cross-fitting** | 5-fold (DoubleML package); 2-fold (manual implementation) |
| **Heterogeneity** | Causal Forest DML (EconML) — individual-level CATE |

---

## Methodology

### Part A — Diagnostic: Manual Cross-Fitting

Three deliberate bugs were embedded in a broken DML pipeline and corrected:

- **Bug 1 — Data Leakage:** The nuisance model predicted on the *training* fold instead of the held-out fold, producing in-sample residuals.  In-sample residuals are overfitted to zero for flexible models (e.g. Random Forests), so `Y_tilde ≈ 0` on the training set, causing the theta estimate to collapse.  Fix: train on `train_idx`, predict on `test_idx`.

- **Bug 2 — Missing Treatment Residualization:** Only the outcome `Y` was residualised; the raw treatment `D` was used as `V_tilde`.  This leaves the confounding channel X → D intact, so the coefficient still picks up selection bias.  Fix: fit a separate `ml_m` model for E[D|X] and compute `V_tilde = D − D_hat` on the held-out fold.

- **Bug 3 — Wrong Theta Formula:** The estimate used `np.mean(V_tilde * Y_tilde)` rather than the correct IV-style ratio `Σ(V_tilde · Y_tilde) / Σ(V_tilde · D)`.  The correct formula is the Frisch–Waugh–Lovell IV projection of `Y_tilde` on `D` using `V_tilde` as an instrument — without it, the scale of the estimate is arbitrary.

After all three fixes, the manual DML recovers the simulated true ATE of **5.0** (within ±0.5).

### Part B — Package-Based DML + Sensitivity Analysis

- `DoubleMLPLR` with Random Forest nuisance learners and 5-fold cross-fitting.
- Sensitivity analysis (`cf_y = 0.03`, `cf_d = 0.03`) quantifies how much unmeasured confounding would be required to nullify the estimate.
- A positive robustness value confirms the ATE survives moderate confounding.

### Part C — Causal Forest CATE

- `CausalForestDML` (EconML) with 500 honest causal trees, 5-fold cross-fitting.
- Individual CATE estimates extracted with 95% confidence intervals.
- High-response subgroup (top 25% of CATE) profiled against the rest of the sample.

### Extension — Subgroup DML vs Causal Forest Heterogeneity

- CATE aggregated by income quartile; within-quartile standard deviation compared to between-quartile range.
- Violin plot shows that substantial treatment-effect heterogeneity *within* each quartile is invisible to subgroup DML.

---

## Key Findings

- **ATE:** 401(k) eligibility raises net financial assets by approximately **$8,000–$10,000** (statistically significant, p < 0.01), consistent with prior literature.
- **Sensitivity:** The robustness value is positive, indicating the result survives a moderate degree of unmeasured confounding.  A confounder would need to explain meaningful variance in both treatment and outcome simultaneously to overturn the finding.
- **Heterogeneity:** The Causal Forest reveals substantial individual-level variation in treatment effects.  High-response households (top CATE quartile) tend to have higher income and different savings profiles.  The mean within-quartile CATE standard deviation substantially exceeds the between-quartile range, confirming that income-quartile subgroup DML understates true heterogeneity.
- **Method choice:** DML is preferred for credible ATE estimation and policy cost-benefit analysis; Causal Forests are preferred when the goal is targeting — identifying *which* individuals or firms benefit most from a policy.

---

## Repository Structure

```
econ-lab-24-causal-ml/
├── README.md
├── notebooks/
│   └── lab_24_causal_ml.ipynb
├── src/
│   └── causal_ml.py
├── figures/
│   ├── cate_histogram.png
│   └── sensitivity_plot.png
└── verification-log.md
```

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Chernozhukov, V., & Hansen, C. (2004). The effects of 401(k) participation on the wealth distribution: An instrumental quantile regression analysis. *Review of Economics and Statistics*, 86(3), 735–751.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242.
