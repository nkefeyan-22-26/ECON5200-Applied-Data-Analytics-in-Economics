# Assignment 5: The Sovereign Risk Engine
### Regularization, Classification, and Model Evaluation for Macroeconomic Early Warning Systems

**Course:** Quantitative Economics  
**Role:** Quantitative Economist — IMF Global Financial Stability Division  
**Platform:** Google Colab (Python 3.10+)

---

## Overview

This project builds a next-generation Early Warning System (EWS) to identify emerging-market economies at elevated risk of economic crisis. The existing OLS model overfits historical data and fails on new countries. This notebook diagnoses that failure and rebuilds the system using regularization, logistic regression, and operational threshold analysis.

---

## Data Pipeline

- **Source:** World Bank API via `wbgapi` — 30 WDI indicators for ~150 countries (2013–2019 averages)
- **Cleaning:** Dropped countries missing >40% of indicators; dropped indicators missing >40% of countries; median-imputed remaining gaps
- **Outcomes:**
  - Continuous: `gdp_growth_pc` (average GDP per capita growth)
  - Binary: `crisis = 1` if `gdp_growth_pc < 0`, else `0`
- **Split:** 70/30 train-test, `random_state=42`
- **Scaling:** `StandardScaler` fit on training data only
- **Final dataset:** ~245 countries × 28 columns | 172 train / 74 test
- **Crisis base rate:** 16.2%

---

## Phase 1: OLS Failure and Regularization Rescue

### 1.1 OLS Overfitting
| Metric | Value |
|---|---|
| Training R² | 0.4832 |
| Test R² | 0.1650 |
| Train-Test Gap | 0.3182 |
| p/n ratio | 0.15 (26 predictors, 172 obs) |

OLS consumes 26 degrees of freedom and overfits to training noise, producing a 0.32 R² gap — a textbook high-variance failure.

### 1.2 Ridge and Lasso
| Model | λ* | Non-zero Coefs | Train R² | Test R² | Test RMSE |
|---|---|---|---|---|---|
| OLS | — | 26 | 0.4832 | 0.1650 | 2.5405 |
| Ridge | 10.0 | 26 | 0.4668 | 0.1188 | 2.6098 |
| Lasso | 0.1228 | 12 | 0.3683 | 0.3048 | 2.3181 |

**Recommendation:** Lasso — best test R² and lowest RMSE. Reduces model to 12 predictors by zeroing out noise.

### 1.3 Lasso Path
- First predictor to enter: `population_growth`
- Non-zero predictors at λ* highlighted in blue; zeroed-out predictors in gray
- Lasso dropping `life_expectancy` reflects conditional predictive redundancy, not economic irrelevance

---

## Phase 2: Crisis Classifier

### 2.1 Linear Probability Model
- 10 out of 74 test predictions fall below 0 (min: -0.1977)
- Negative probabilities are operationally incoherent for crisis flagging

### 2.2 Logistic Regression
Top predictors by odds ratio:

| Feature | Odds Ratio |
|---|---|
| urban_population_pct | 2.10 |
| unemployment_rate | 1.61 |
| population_growth | 1.38 |
| arable_land_pct | 0.52 |
| life_expectancy | 0.59 |

- All predicted probabilities bounded within [0.0024, 0.6675] ✓

### 2.3 LPM vs. Logistic Visualization
- LPM: straight line crossing below 0 (impossible region shaded orange)
- Logistic: smooth sigmoid bounded within [0, 1]

---

## Phase 3: Operational Deployment

### 3.1 Accuracy Paradox
| Model | Accuracy | Recall |
|---|---|---|
| Naïve baseline (always "no crisis") | 0.838 | 0.000 |
| Logistic regression (τ = 0.5) | 0.865 | 0.333 |

Accuracy alone is misleading — the naïve model achieves 83.8% by ignoring crises entirely.

### 3.2 Confusion Matrix (τ = 0.5)
|  | Predicted No Crisis | Predicted Crisis |
|---|---|---|
| **Actual No Crisis** | 60 (TN) | 2 (FP) |
| **Actual Crisis** | 8 (FN) | 4 (TP) |

- Precision: 0.67 | Recall: 0.33 | F1: 0.44
- At $50B per missed crisis, 8 false negatives is operationally dangerous

### 3.3 ROC and Precision-Recall Curves
| Metric | Value |
|---|---|
| ROC-AUC | 0.805 |
| PR-AUC | 0.599 |

ROC-AUC is inflated by the abundance of true negatives. PR-AUC is the more honest metric for crisis detection.

### 3.4 Threshold Analysis
| Threshold | Type | Countries Flagged | Precision | Recall |
|---|---|---|---|---|
| τ = 0.27 | F1-optimal | ~10 | ~0.60 | ~0.60 |
| τ = 0.55 | Capacity-constrained (≤5 missions) | 5 | 0.800 | 0.333 |
| τ = 0.01 | Cost-minimizing | 62 | low | 1.000 |

**Recommendation:** Use τ = 0.27 as the flagging threshold; deploy the 5 available missions to the highest-probability countries first.

---

## Phase 4: AI-Assisted Diagnostics (P.R.I.M.E. Framework)

### Task 4.1 — Bootstrap Lasso Stability (200 resamples)
**Stable predictors (>80% selection frequency):**
`population`, `population_growth`, `urban_population_pct`, `arable_land_pct`, `unemployment_rate`, `inflation`, `imports_pct_gdp`, `military_expenditure_pct_gdp`

**Fragile predictors (<30% selection frequency):**
`internet_users_pct` (32.5%)

Instability among mid-tier predictors reflects multicollinearity among WDI indicators.

### Task 4.2 — Cost-Sensitive Threshold Optimization
- **Cost structure:** FN = $50B | FP = $2M
- **Cost-minimizing threshold:** τ = 0.01
- **At τ = 0.01:** 0 missed crises, 56 false alarms, total cost = $0.11B
- The extreme cost asymmetry drives the optimal threshold far below the F1-optimal and capacity-constrained thresholds

---

## Dependencies

```
wbgapi
pandas
numpy
scikit-learn
statsmodels
matplotlib
seaborn
```

Install with:
```bash
pip install wbgapi scikit-learn statsmodels matplotlib seaborn numpy pandas
```

---

## How to Run

1. Open Google Colab
2. Run cells top to bottom (`Runtime > Run All`)
3. No CSV files needed — data downloads live from the World Bank API
