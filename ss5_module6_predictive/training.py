"""
Component 2 of Module 6: Ridge regression training with nested
cross-validation.

The training pipeline takes a feature matrix X and a target vector
y produced by Component 1 (features.py) and Module 1's
literature-grounded synthetic target, and returns a fitted
scikit-learn estimator alongside diagnostic information sufficient
for downstream reporting and prediction.

Two layers of cross-validation are used:

  - Inner CV (RidgeCV.cv): selects the regularisation strength
    alpha by minimising the mean cross-validated mean-squared error
    over a logarithmic grid spanning six orders of magnitude.

  - Outer CV (cross_validate): produces a held-out estimate of
    generalisation performance. On each outer fold the inner CV
    re-selects alpha from scratch, then the selected model is
    evaluated on the held-out fold. This is the methodologically
    correct way to report generalisation: alpha is a hyperparameter,
    so its selection must be inside the CV loop.

Both CV layers use 5 folds. Features are z-score scaled inside the
pipeline so that ridge's penalty is applied uniformly across
features regardless of their physical units. Coefficients reported
in the TrainingResult are back-transformed to original-feature
units for direct interpretability.

Provides:

    TrainingConfig  : dataclass holding ridge hyperparameters
    TrainingResult  : dataclass holding the fitted pipeline,
                      selected alpha, back-transformed
                      coefficients, and CV scores
    train_ridge     : main entry point
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for ridge-regression training.

    Defaults are appropriate for the Module 6 reduced 6-feature
    set on a ~200-batch training set; adjust if the descriptor
    space or training-set size changes substantially.
    """
    # Logarithmic grid over which RidgeCV searches for the best
    # alpha. Spans six orders of magnitude so the optimum is
    # essentially never on the boundary at this dataset size.
    alpha_grid_min_log: float = -3.0
    alpha_grid_max_log: float = 3.0
    alpha_grid_n: int = 50

    # Cross-validation folds for the inner (alpha-selection) and
    # outer (generalisation-estimate) loops respectively.
    inner_cv_folds: int = 5
    outer_cv_folds: int = 5

    # RNG seed for reproducible fold splitting.
    random_state: int = 42

    @property
    def alphas(self) -> np.ndarray:
        return np.logspace(
            self.alpha_grid_min_log,
            self.alpha_grid_max_log,
            self.alpha_grid_n,
        )


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingResult:
    """
    Output of train_ridge.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted StandardScaler + RidgeCV pipeline. Use directly
        for prediction on new feature vectors.
    selected_alpha : float
        The alpha chosen by inner CV on the full training data.
    coefficients : dict
        Mapping from feature name to back-transformed coefficient
        (in original-feature units). Includes 'intercept' as a
        separate key.
    feature_names : list
        Feature names in the same order as the columns of X used
        during training. Useful for interpretation.
    cv_scores : dict
        Pooled outer-CV metrics: keys 'r2', 'rmse', 'mae'. Each
        value is a dict with 'mean', 'sd', and 'per_fold' (1D array).
    n_train : int
        Total number of training examples seen.
    """
    pipeline: Any
    selected_alpha: float
    coefficients: dict
    feature_names: list
    cv_scores: dict
    n_train: int


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def _back_transform_coefficients(pipeline, feature_names):
    """
    Recover ridge coefficients in original (un-scaled) feature
    units from a StandardScaler + RidgeCV pipeline.

    The pipeline's ridge model fits coefficients in the scaled
    feature space:
        y_hat = intercept_scaled + sum_i  w_scaled_i * z_i
    where z_i = (x_i - mu_i) / sigma_i.

    Substituting and grouping by x_i:
        y_hat = intercept + sum_i  (w_scaled_i / sigma_i) * x_i
    where intercept = intercept_scaled - sum_i  w_scaled_i * mu_i / sigma_i.
    """
    scaler = pipeline.named_steps["standardscaler"]
    ridge = pipeline.named_steps["ridgecv"]

    mu = scaler.mean_
    sigma = scaler.scale_
    w_scaled = ridge.coef_
    b_scaled = ridge.intercept_

    w_orig = w_scaled / sigma
    b_orig = b_scaled - float(np.sum(w_scaled * mu / sigma))

    out = {"intercept": float(b_orig)}
    for name, w in zip(feature_names, w_orig):
        out[name] = float(w)
    return out


def train_ridge(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """
    Train ridge regression with nested cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features). Produced
        by features.extract_feature_matrix.
    y : np.ndarray
        Target vector of shape (n_samples,). The synthetic MOR-like
        target produced by Module 1.
    feature_names : list
        Feature names in column order, used for coefficient labelling.
    config : TrainingConfig, optional
        Training configuration. Default is the standard 5-fold/5-fold
        nested CV with a wide alpha grid.

    Returns
    -------
    TrainingResult
        Trained pipeline plus diagnostic information.
    """
    if config is None:
        config = TrainingConfig()

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

    # Pipeline: standardise features, then ridge with internal CV
    # over the alpha grid. Inner CV folds are deterministic (same
    # KFold splitter seeded with config.random_state).
    inner_cv = KFold(
        n_splits=config.inner_cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    pipeline = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=config.alphas, cv=inner_cv),
    )

    # Outer CV: evaluate held-out generalisation. On each outer
    # fold the pipeline (including alpha selection) is refit from
    # scratch, so this is genuine nested CV.
    outer_cv = KFold(
        n_splits=config.outer_cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=outer_cv,
        scoring=("r2", "neg_root_mean_squared_error",
                 "neg_mean_absolute_error"),
        return_train_score=False,
        n_jobs=1,
    )

    # Pooled CV scores (mean and SD across outer folds)
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

    # Final fit on the full training set: this is the model used
    # for prediction. Inner CV inside RidgeCV re-selects alpha on
    # the full data; that selected alpha is what we report.
    pipeline.fit(X, y)
    selected_alpha = float(pipeline.named_steps["ridgecv"].alpha_)

    coefficients = _back_transform_coefficients(pipeline, feature_names)

    return TrainingResult(
        pipeline=pipeline,
        selected_alpha=selected_alpha,
        coefficients=coefficients,
        feature_names=list(feature_names),
        cv_scores=cv_scores,
        n_train=X.shape[0],
    )