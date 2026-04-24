# Tree-Based Models — Random Forests

## Objective
Benchmark ensemble tree-based methods against linear regression on the California Housing dataset, diagnose common evaluation and interpretability pitfalls, and deliver a reproducible feature importance analysis using both MDI and SHAP.

## Methodology
- Loaded and split the California Housing dataset (20,640 observations, 8 features) into 80/20 train/test partitions with a fixed random seed for reproducibility
- Trained and evaluated three model classes on the held-out test set: Ridge Regression (α=1.0), Decision Tree, and Random Forest (default and tuned)
- Identified and corrected a training-set evaluation bug in which RF performance was measured on in-sample data, inflating reported R² from ~0.81 to >0.97
- Tuned RF hyperparameters (`n_estimators`, `max_depth`, `max_features`) using a direct parameter selection approach after establishing diminishing returns beyond 200 trees
- Compared MDI vs. permutation importance on the test set, diagnosing MDI's systematic bias toward high-cardinality continuous features
- Extended the analysis with SHAP TreeExplainer: generated waterfall plots for individual predictions and a beeswarm plot for global feature attribution
- Built an interactive Plotly + ipywidgets dashboard allowing live exploration of how `n_estimators` and `max_features` affect model performance, feature rankings, and train/test R² curves

## Key Findings
- Ridge Regression (R² = 0.576) substantially underperforms ensemble methods, confirming strong nonlinear structure in the data that a linear model cannot capture
- Default Random Forest (R² = 0.805) recovers most of that signal without any tuning, highlighting the practical value of ensemble methods on tabular data
- Tuned Random Forest (R² = 0.812) improves only marginally over the default, suggesting RF is relatively robust to hyperparameter choice on this dataset
- Gradient Boosting (R² = 0.829, RMSE = 0.474) is the best-performing model, outperforming the tuned RF by ~0.017 R² — a modest but consistent edge attributable to sequential error correction
- `MedInc` (median income) is the dominant predictive feature under both permutation and SHAP rankings; however, this is a predictive association and cannot be interpreted causally without a formal identification strategy (e.g., DML, IV, or RDD)
- MDI and SHAP rankings diverge most at mid-tier features (e.g., `AveRooms`), where MDI inflates importance due to cardinality bias — SHAP provides a more reliable attribution
- Beyond ~200 trees, marginal gains in test R² fall below 0.005 per 100 additional estimators, making large forests difficult to justify on computational grounds
