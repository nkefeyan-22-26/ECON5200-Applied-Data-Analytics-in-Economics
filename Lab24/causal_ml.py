"""
causal_ml.py
============
Reusable causal inference utilities for ECON 5200 — Lab 24.

Functions
---------
manual_dml        : Manual 2-fold cross-fitted Double ML (Partially Linear Regression)
cate_by_subgroup  : Summarise Causal Forest CATE predictions by a discrete grouping variable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from typing import Optional, Tuple


def manual_dml(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    ml_l: Optional[object] = None,
    ml_m: Optional[object] = None,
    n_splits: int = 2,
    random_state: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Manual cross-fitted Double Machine Learning (Partially Linear Regression).

    Implements the DML1 estimator from Chernozhukov et al. (2018):

        theta = sum(V_tilde * Y_tilde) / sum(V_tilde * D)

    where Y_tilde = Y - E[Y|X] and V_tilde = D - E[D|X] are computed via
    K-fold cross-fitting to guarantee out-of-sample residuals.

    Parameters
    ----------
    Y : np.ndarray, shape (n,)
        Outcome variable.
    D : np.ndarray, shape (n,)
        Binary or continuous treatment variable.
    X : np.ndarray, shape (n, p)
        Covariate matrix (confounders / controls).
    ml_l : sklearn-compatible regressor, optional
        Nuisance learner for E[Y|X].  Defaults to
        RandomForestRegressor(n_estimators=200, max_depth=5).
    ml_m : sklearn-compatible regressor, optional
        Nuisance learner for E[D|X].  Defaults to
        RandomForestRegressor(n_estimators=200, max_depth=5).
    n_splits : int, default 2
        Number of cross-fitting folds.
    random_state : int, default 42
        Random seed for KFold splitting.

    Returns
    -------
    theta : float
        DML point estimate of the average treatment effect.
    Y_tilde : np.ndarray, shape (n,)
        Cross-fitted outcome residuals  Y - E[Y|X].
    V_tilde : np.ndarray, shape (n,)
        Cross-fitted treatment residuals  D - E[D|X].

    Notes
    -----
    Each nuisance model is cloned before fitting so that the caller's
    original estimator objects are never mutated.
    """
    if ml_l is None:
        ml_l = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=random_state)
    if ml_m is None:
        ml_m = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=random_state)

    n = len(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    Y_tilde = np.zeros(n)
    V_tilde = np.zeros(n)

    for train_idx, test_idx in kf.split(X):
        l_fold = clone(ml_l)
        l_fold.fit(X[train_idx], Y[train_idx])
        Y_tilde[test_idx] = Y[test_idx] - l_fold.predict(X[test_idx])

        m_fold = clone(ml_m)
        m_fold.fit(X[train_idx], D[train_idx])
        V_tilde[test_idx] = D[test_idx] - m_fold.predict(X[test_idx])

    theta = np.sum(V_tilde * Y_tilde) / np.sum(V_tilde * D)

    return theta, Y_tilde, V_tilde


def cate_by_subgroup(
    cate_predictions: np.ndarray,
    group_labels: pd.Series,
    group_order: Optional[list] = None,
) -> pd.DataFrame:
    """
    Summarise individual-level CATE predictions by a discrete grouping variable.

    Computes mean, standard deviation, median, and count of CATE estimates
    within each group.  Useful for comparing Causal Forest heterogeneity
    against coarse subgroup DML estimates.

    Parameters
    ----------
    cate_predictions : np.ndarray, shape (n,)
        Individual CATE estimates from CausalForestDML.effect(X).
    group_labels : pd.Series, length n
        Categorical or string series that assigns each observation to a group
        (e.g. income quartile labels produced by pd.qcut).
    group_order : list, optional
        Desired display order for the groups.  If None, uses the natural
        sort order of the labels.

    Returns
    -------
    summary : pd.DataFrame
        Columns: ['Mean CATE', 'Std CATE', 'Median CATE', 'N']
        Indexed by the unique values of group_labels.

    Examples
    --------
    >>> summary = cate_by_subgroup(
    ...     cate_predictions,
    ...     pd.qcut(data['inc'], q=4, labels=['Q1','Q2','Q3','Q4']),
    ...     group_order=['Q1','Q2','Q3','Q4']
    ... )
    >>> print(summary)
    """
    temp = pd.DataFrame({
        'cate':  cate_predictions,
        'group': group_labels.values,
    })

    summary = (
        temp
        .groupby('group')['cate']
        .agg(
            **{
                'Mean CATE':   'mean',
                'Std CATE':    'std',
                'Median CATE': 'median',
                'N':           'count',
            }
        )
    )

    if group_order is not None:
        summary = summary.reindex(group_order)

    return summary.round(2)
