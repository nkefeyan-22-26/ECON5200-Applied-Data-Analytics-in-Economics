"""
shap_analysis.py
================
Reusable SHAP explanation utilities for tree-based sklearn models.
Compatible with Google Colab and standard Python environments.

Portfolio artifact for ECON 5200 Lab 19.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance


def explain_prediction(
    model: BaseEstimator,
    X: pd.DataFrame,
    idx: int,
) -> None:
    """
    Display a SHAP waterfall plot for a single observation.

    Parameters
    ----------
    model : fitted sklearn tree-based estimator (RF, GBR, etc.)
    X     : feature DataFrame (test set recommended)
    idx   : integer row index within X
    """
    if idx < 0 or idx >= len(X):
        raise IndexError(f"idx={idx} is out of range for X with {len(X)} rows.")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.plots.waterfall(shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X.iloc[idx],
        feature_names=list(X.columns)
    ))


def global_importance(
    model: BaseEstimator,
    X: pd.DataFrame,
) -> pd.Series:
    """
    Display a SHAP beeswarm plot summarising global feature importance.

    Parameters
    ----------
    model : fitted sklearn tree-based estimator
    X     : feature DataFrame (test set recommended)

    Returns
    -------
    pd.Series of mean absolute SHAP values, sorted descending
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.plots.beeswarm(shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X,
        feature_names=list(X.columns)
    ))

    return pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False)


def compare_importance(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Side-by-side bar chart comparing MDI vs SHAP feature rankings.

    Parameters
    ----------
    model        : fitted sklearn tree-based estimator
    X            : feature DataFrame (test set recommended)
    y            : true target values matching X
    n_repeats    : shuffles for permutation importance (default 3 = fast)
    random_state : RNG seed

    Returns
    -------
    pd.DataFrame with columns ['MDI', 'SHAP'], sorted by SHAP descending
    """
    # MDI
    mdi = pd.Series(model.feature_importances_, index=X.columns)

    # SHAP
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_imp    = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)

    # Normalise both to [0, 1] so scales are comparable
    mdi_norm  = mdi     / mdi.sum()
    shap_norm = shap_imp / shap_imp.sum()

    df = pd.DataFrame({'MDI': mdi_norm, 'SHAP': shap_norm})
    df = df.sort_values('SHAP', ascending=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    df['MDI'].sort_values().plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_title('MDI Importance\n(biased toward high-cardinality features)')
    ax1.set_xlabel('Normalised importance')

    df['SHAP'].sort_values().plot(kind='barh', ax=ax2, color='tomato')
    ax2.set_title('Mean |SHAP|\n(measures true marginal contribution)')
    ax2.set_xlabel('Normalised importance')

    plt.suptitle('MDI vs. SHAP Feature Ranking', fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\nMDI ranking:  ", list(df.sort_values('MDI',  ascending=False).index))
    print("SHAP ranking: ", list(df.index))
    print("\nWhere they diverge, SHAP is more reliable — MDI inflates continuous")
    print("high-cardinality features (e.g. AveRooms). SHAP measures true marginal contribution.")

    return df
