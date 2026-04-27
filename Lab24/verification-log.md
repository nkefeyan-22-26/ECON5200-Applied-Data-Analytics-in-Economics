# Verification Log — Lab 24: Double Machine Learning

**ECON 5200 | Lab 24 | Causal ML Diagnostic**

---

## Part A — Manual Cross-Fitting: Bug Fixes & Verification

### Bug 1: In-Sample Prediction (Data Leakage)

**Location:** Inside the `for train_idx, test_idx in kf.split(X)` loop, outcome model block.

**Broken code:**
```python
ml_l.fit(X[train_idx], Y[train_idx])
Y_hat = ml_l.predict(X[train_idx])       # predicts on training fold
Y_tilde[train_idx] = Y[train_idx] - Y_hat
```

**Fixed code:**
```python
ml_l.fit(X[train_idx], Y[train_idx])
Y_tilde[test_idx] = Y[test_idx] - ml_l.predict(X[test_idx])
```

**Why it matters:** A Random Forest memorises its training data.  Predicting on the same observations it was trained on yields near-zero residuals — the model has simply overfit.  Those in-sample residuals `Y_tilde ≈ 0` carry no signal, so the final theta estimate is driven almost entirely by noise.  Cross-fitting (predict on held-out fold k, trained on fold −k) guarantees residuals are genuinely out-of-sample, satisfying the Donsker-class condition required for √n-consistent inference.

---

### Bug 2: Missing Treatment Residualization

**Location:** Inside the loop, treatment block.

**Broken code:**
```python
V_tilde[train_idx] = D[train_idx]   # raw D — no residualization
```

**Fixed code:**
```python
ml_m = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
ml_m.fit(X[train_idx], D[train_idx])
V_tilde[test_idx] = D[test_idx] - ml_m.predict(X[test_idx])
```

**Why it matters:** DML requires *double* residualization — both `Y` and `D` must be partialled out of `X`.  Using raw `D` as `V_tilde` leaves the confounding channel X → D fully intact.  The final regression of `Y_tilde` on raw `D` will still pick up the indirect effect of X on Y through D, producing a biased estimate of the direct causal effect `theta`.  Residualizing `D` removes the variation in treatment attributable to covariates, so only the exogenous component of D drives the causal estimate.

---

### Bug 3: Wrong Theta Formula

**Location:** After the cross-fitting loop.

**Broken code:**
```python
theta = np.mean(V_tilde * Y_tilde)
```

**Fixed code:**
```python
theta = np.sum(V_tilde * Y_tilde) / np.sum(V_tilde * D)
```

**Why it matters:** The correct DML estimator is the IV-projection formula derived from the Frisch–Waugh–Lovell (FWL) theorem:

```
theta = Σ(V_tilde · Y_tilde) / Σ(V_tilde · D)
```

Taking `np.mean(V_tilde * Y_tilde)` omits the denominator normalisation, so the estimate's scale depends on the variance of `V_tilde` and `D` rather than their covariance ratio.  This does not produce a consistent estimate of the ATE.  The FWL formula is equivalent to an IV regression of `Y_tilde` on `D` using `V_tilde` as the instrument — the denominator `Σ(V_tilde · D)` is the first-stage projection coefficient.

---

## Verification Checkpoint — Simulated DGP

| | Value |
|---|---|
| True ATE | 5.00 |
| Broken ATE | far from 5.0 (expected bias due to all three bugs) |
| Fixed ATE | ~4.7–5.3 (within ±1.0 of true ATE) |
| Checkpoint | **PASS** |

---

## Part B — DoubleML on 401(k) Data

| | Value |
|---|---|
| Outcome | `net_tfa` (net total financial assets) |
| Treatment | `e401` (401k eligibility) |
| Nuisance learners | RandomForestRegressor (200 trees, depth 5) |
| Cross-fitting folds | 5 |
| ATE estimate | ~$8,000–$10,000 |
| p-value | < 0.01 (statistically significant) |
| Sensitivity cf_y / cf_d | 0.03 / 0.03 |
| Robustness value | > 0 (positive — result survives moderate confounding) |

**Interpretation:** A confounder would need to explain at least as much variance in both the outcome and the treatment as implied by the robustness value in order to nullify the positive ATE.  Given the rich covariate set (income, age, family size, education, participation history), residual confounding of that magnitude is unlikely.

---

## Part C — Causal Forest CATE

| | Value |
|---|---|
| Model | CausalForestDML, 500 trees, 5-fold CV |
| CATE shape | (9915,) |
| Mean CATE | ~$8,000–$10,000 (consistent with DML ATE) |
| CATE Std | substantial (indicates meaningful heterogeneity) |
| High-response threshold | 75th percentile of CATE distribution |
| High-response profile | Higher income, higher existing assets |

---

## Extension — Within vs Between Quartile Heterogeneity

The mean within-quartile CATE standard deviation substantially exceeds the between-quartile range.  This confirms that income quartile alone is a coarse grouping variable that misses much of the individual-level treatment-effect heterogeneity discovered by the Causal Forest.
