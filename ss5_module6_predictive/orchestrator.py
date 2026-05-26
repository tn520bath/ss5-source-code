"""
Module 6 orchestrator: the single entry point.

The orchestrator couples the four substantive components built in
training.py, comparator.py, validation.py, and prediction.py into
a single training-time call producing a Module6Result that holds
everything needed for the report's figures, tables, and the
runtime-prediction path.

Two distinct usage modes:

  1. fit_module6(X, y, feature_names)
       Training mode: takes a prepared feature matrix and target
       vector, runs ridge training (with nested CV), the random-
       forest comparator (optional, on by default), and the
       held-out CV prediction extraction. Returns a Module6Result.

  2. module6_predict(result, x_point, x_sigma, fallback_sigma_eps)
       Runtime mode: takes a previously-fitted Module6Result plus
       a new batch's point-estimate features and Module 4
       uncertainties, returns a BatchPrediction with 95% interval.

The orchestrator does not perform feature extraction or descriptor
computation; those are the responsibilities of Modules 2, 3, and
the features module. The orchestrator takes the resulting (X, y)
directly so the upstream-data dependency surface is minimal.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .training import (
    train_ridge,
    TrainingConfig,
    TrainingResult,
)
from .comparator import (
    train_random_forest,
    ComparatorConfig,
    ComparatorResult,
)
from .validation import (
    cv_holdout_predictions,
    CVPredictions,
)
from .prediction import (
    predict_with_uncertainty,
    PredictionConfig,
    BatchPrediction,
)


# -------------------------------------------------------------------
# Top-level result dataclass
# -------------------------------------------------------------------

@dataclass(frozen=True)
class Module6Result:
    """
    Complete Module 6 output from a training run.

    Attributes
    ----------
    ridge : TrainingResult
        Ridge training output: fitted pipeline, selected alpha,
        back-transformed coefficients, outer-CV scores.
    cv_predictions : CVPredictions
        Per-batch held-out predictions, actuals, fold IDs, and
        residuals from the outer CV. Used directly by the
        predicted-vs-actual and residual-plot figures.
    comparator : ComparatorResult or None
        Random-forest comparator output if comparator_enabled was
        True; None otherwise.
    feature_names : list
        Feature names in column order, copied from the input.
    n_train : int
        Number of training examples used.
    """
    ridge: TrainingResult
    cv_predictions: CVPredictions
    comparator: ComparatorResult | None
    feature_names: list
    n_train: int

    @property
    def cv_r2_mean(self) -> float:
        return self.ridge.cv_scores["r2"]["mean"]

    @property
    def cv_rmse_mean(self) -> float:
        return self.ridge.cv_scores["rmse"]["mean"]

    @property
    def selected_alpha(self) -> float:
        return self.ridge.selected_alpha


# -------------------------------------------------------------------
# Training-mode entry point
# -------------------------------------------------------------------

def fit_module6(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    training_config: TrainingConfig | None = None,
    comparator_config: ComparatorConfig | None = None,
    comparator_enabled: bool = True,
) -> Module6Result:
    """
    Fit the Module 6 ridge model and (optionally) the random-forest
    comparator, plus the held-out CV predictions.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features), produced
        by ss5_module6_predictive.features.extract_feature_matrix.
    y : np.ndarray
        Target vector of shape (n_samples,).
    feature_names : list
        Feature names in column order, used for coefficient
        labelling and figure axes.
    training_config : TrainingConfig, optional
        Ridge training configuration; defaults are appropriate for
        the standard 6-feature, ~200-batch setup.
    comparator_config : ComparatorConfig, optional
        Random-forest comparator configuration.
    comparator_enabled : bool
        If True (default), run the random-forest comparator
        alongside ridge. If False, skip the comparator (faster, but
        feature-importance figures are not available downstream).

    Returns
    -------
    Module6Result
        All component outputs collected into a single dataclass.
    """
    if training_config is None:
        training_config = TrainingConfig()

    ridge_result = train_ridge(X, y, feature_names, training_config)
    cv_preds = cv_holdout_predictions(X, y, training_config)

    if comparator_enabled:
        if comparator_config is None:
            comparator_config = ComparatorConfig()
        comparator_result = train_random_forest(
            X, y, feature_names, comparator_config
        )
    else:
        comparator_result = None

    return Module6Result(
        ridge=ridge_result,
        cv_predictions=cv_preds,
        comparator=comparator_result,
        feature_names=list(feature_names),
        n_train=X.shape[0],
    )


# -------------------------------------------------------------------
# Runtime-mode entry point
# -------------------------------------------------------------------

def module6_predict(
    result: Module6Result,
    x_point: np.ndarray,
    x_sigma: np.ndarray,
    prediction_config: PredictionConfig | None = None,
    fallback_sigma_eps: float | None = None,
) -> BatchPrediction:
    """
    Predict the synthetic MOR for a new batch with a 95% Monte
    Carlo prediction interval.

    The trained ridge pipeline inside `result` is used for
    prediction. The noise floor sigma_eps is taken from
    prediction_config if given, else from fallback_sigma_eps if
    given, else inferred from the training-CV residual RMSE in
    `result`.

    Parameters
    ----------
    result : Module6Result
        Output of fit_module6.
    x_point : np.ndarray
        Point-estimate feature vector, shape (n_features,).
    x_sigma : np.ndarray
        Per-feature standard uncertainty, shape (n_features,).
        Typically from Module 4.
    prediction_config : PredictionConfig, optional
        Defaults to 1000 samples, 95% CI, percentile method.
    fallback_sigma_eps : float, optional
        Override for the sigma_eps fallback; if not given, uses
        the training-CV residual RMSE from `result`.

    Returns
    -------
    BatchPrediction
        Point estimate, 95% CI, MC SD, and sigma_eps actually used.
    """
    if prediction_config is None:
        prediction_config = PredictionConfig()

    # If neither config.sigma_eps nor an explicit fallback is given,
    # use the training-CV RMSE as the noise-floor estimate.
    effective_fallback = fallback_sigma_eps
    if effective_fallback is None:
        effective_fallback = result.ridge.cv_scores["rmse"]["mean"]

    return predict_with_uncertainty(
        model=result.ridge.pipeline,
        x_point=x_point,
        x_sigma=x_sigma,
        config=prediction_config,
        fallback_sigma_eps=effective_fallback,
    )