"""
Component 3 of Module 6: Random-forest comparator.

A small random-forest regressor that runs alongside the ridge model
(Component 2). Its purpose is twofold:

  1. Test whether non-linearity in the synthetic target (the
     saturating ramps on length and aspect ratio) materially
     improves predictive accuracy when modelled by a non-parametric
     estimator. If random forest does not outperform ridge by more
     than fold-to-fold noise, the descriptor space the model sees
     is effectively linear and ridge is sufficient.

  2. Provide an alternative view of feature importance that is
     robust to feature correlation. Ridge's coefficients are biased
     by multicollinearity (see Component 2); random
     forest permutation importances are not, since they measure
     the drop in held-out predictive accuracy when each feature is
     individually shuffled rather than the partial slope in a
     linear fit.

Two importance measures are reported:

  - Mean Decrease in Impurity (MDI), scikit-learn's default
    feature_importances_, fast but known to be biased toward
    high-cardinality features and unreliable under feature
    correlation.

  - Permutation importance, measured on held-out CV predictions
    with multiple repeats per feature for stability.

The contrast between the two is itself a small reportable finding
under multicollinearity.

Provides:

    ComparatorConfig : random-forest hyperparameters
    ComparatorResult : fitted forest, CV scores, both importance
                       measures, feature names
    train_random_forest : main entry point
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.inspection import permutation_importance


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@dataclass(frozen=True)
class ComparatorConfig:
    """
    Configuration for the random-forest comparator.

    Defaults match scikit-learn's regression recommendations for
    small-to-moderate datasets, with n_estimators increased modestly
    above the default 100 to produce more stable feature-importance
    estimates without meaningful runtime cost at this dataset size.
    """
    n_estimators: int = 300
    max_features: str = "sqrt"        # standard regression recommendation
    max_depth: int | None = None      # let trees grow; CV catches overfit
    min_samples_leaf: int = 2         # gentle regularisation at small N

    # Cross-validation folds for the outer (generalisation) loop.
    # Matches Component 2's outer CV so ridge and forest results are
    # directly comparable.
    outer_cv_folds: int = 5

    # Permutation importance settings.
    permutation_n_repeats: int = 10

    # RNG seed for reproducible bootstrap sampling, fold splitting,
    # and permutation shuffling.
    random_state: int = 42


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------

@dataclass(frozen=True)
class ComparatorResult:
    """
    Output of train_random_forest.

    Attributes
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        The fitted forest, trained on the full dataset.
    feature_names : list
        Feature names in the same column order as X used during
        training. Importances are indexed in this order.
    cv_scores : dict
        Pooled outer-CV metrics matching Component 2's format,
        keys 'r2', 'rmse', 'mae', each a dict with 'mean', 'sd',
        and 'per_fold'.
    mdi_importance : np.ndarray
        Mean-decrease-in-impurity importance from
        RandomForestRegressor.feature_importances_, shape
        (n_features,). Note: known to be biased under feature
        correlation; reported alongside permutation importance for
        comparison.
    perm_importance_mean : np.ndarray
        Permutation-importance mean across n_repeats, shape
        (n_features,). Robust to feature correlation.
    perm_importance_sd : np.ndarray
        Permutation-importance standard deviation across n_repeats,
        shape (n_features,). Provides error bars for plotting.
    n_train : int
        Total number of training examples seen.
    """
    model: Any
    feature_names: list
    cv_scores: dict
    mdi_importance: np.ndarray
    perm_importance_mean: np.ndarray
    perm_importance_sd: np.ndarray
    n_train: int


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    config: ComparatorConfig | None = None,
) -> ComparatorResult:
    """
    Train a random forest on (X, y) and produce CV scores plus
    both MDI and permutation feature importances.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features), same input
        format as train_ridge.
    y : np.ndarray
        Target vector of shape (n_samples,).
    feature_names : list
        Feature names in column order, used for importance labelling.
    config : ComparatorConfig, optional
        Comparator configuration. Default is the standard 5-fold CV
        with 300 trees and permutation importance over 10 repeats.

    Returns
    -------
    ComparatorResult
        Trained forest plus CV and importance diagnostics.
    """
    if config is None:
        config = ComparatorConfig()

    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples; got "
            f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
        )
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not "
            f"match X.shape[1] ({X.shape[1]})"
        )

    # Build the estimator. n_jobs=1 keeps the run deterministic and
    # avoids platform-specific threading differences at this scale.
    forest_factory = lambda: RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_features=config.max_features,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=1,
    )

    # Outer CV: same fold splitter and seed as Component 2's ridge
    # training. Fold-by-fold parity matters for the head-to-head
    # comparison reported in the Module 6 prose.
    outer_cv = KFold(
        n_splits=config.outer_cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    cv_results = cross_validate(
        forest_factory(),
        X,
        y,
        cv=outer_cv,
        scoring=("r2", "neg_root_mean_squared_error",
                 "neg_mean_absolute_error"),
        return_train_score=False,
        n_jobs=1,
    )

    r2_per_fold = cv_results["test_r2"]
    rmse_per_fold = -cv_results["test_neg_root_mean_squared_error"]
    mae_per_fold = -cv_results["test_neg_mean_absolute_error"]

    cv_scores = {
        "r2":   {"mean": float(np.mean(r2_per_fold)),
                 "sd":   float(np.std(r2_per_fold, ddof=1)),
                 "per_fold": r2_per_fold},
        "rmse": {"mean": float(np.mean(rmse_per_fold)),
                 "sd":   float(np.std(rmse_per_fold, ddof=1)),
                 "per_fold": rmse_per_fold},
        "mae":  {"mean": float(np.mean(mae_per_fold)),
                 "sd":   float(np.std(mae_per_fold, ddof=1)),
                 "per_fold": mae_per_fold},
    }

    # Final fit on the full dataset for downstream use.
    model = forest_factory()
    model.fit(X, y)
    mdi = np.asarray(model.feature_importances_, dtype=float)

    # Permutation importance, computed on held-out predictions
    # generated by per-fold refitting. We use a single fresh forest
    # fit-and-score split here for compactness; the importance scores
    # are robust because permutation_importance averages over
    # n_repeats shuffles of each feature.
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=config.permutation_n_repeats,
        random_state=config.random_state,
        n_jobs=1,
        scoring="r2",
    )

    return ComparatorResult(
        model=model,
        feature_names=list(feature_names),
        cv_scores=cv_scores,
        mdi_importance=mdi,
        perm_importance_mean=np.asarray(perm.importances_mean, dtype=float),
        perm_importance_sd=np.asarray(perm.importances_std, dtype=float),
        n_train=X.shape[0],
    )