"""
Component 4 of Module 6: Monte Carlo prediction intervals.

Couples Module 4's per-descriptor uncertainty to Module 6's trained
ridge model, producing a 95% prediction interval on the synthetic
MOR target for each batch. The methodology mirrors Module 4's own
percentile-method GUM-S1 approach:

  1. For a batch with point-estimate features x and per-feature
     standard uncertainties sigma_x, draw K samples
         x^(k) ~ N(x, diag(sigma_x^2))
  2. For each sample compute the model prediction
         y_hat^(k) = f_theta(x^(k))
  3. Add irreducible target noise drawn from N(0, sigma_eps)
         y_tilde^(k) = y_hat^(k) + epsilon^(k)
  4. Take the 2.5th and 97.5th percentiles of {y_tilde^(k)} as the
     95% prediction-interval bounds.

The irreducible noise sigma_eps represents the floor of prediction
uncertainty even with perfect descriptor knowledge. It can be:

  - inferred from the training cross-validation residual RMSE (the
    empirical noise floor the model has actually learned), or
  - supplied directly (useful in synthetic-data validation where
    the true noise level is known from the target spec).

Two sources of uncertainty therefore combine multiplicatively in
the output interval: input uncertainty propagated through the
model, and irreducible noise on the target.

Provides:

    PredictionConfig : Monte Carlo iterations, noise mode, seed
    BatchPrediction  : point estimate, 95% CI bounds, MC SD
    predict_with_uncertainty       : single-batch prediction
    predict_batch_with_uncertainty : vectorised over many batches
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionConfig:
    """
    Configuration for Monte Carlo prediction-interval generation.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo draws per batch. 1000 matches Module 4's
        K = 1000 convention and is sufficient for stable percentile
        estimates at 95% confidence; raise to 5000 if tighter
        percentile bounds are needed.
    sigma_eps : float or None
        Standard deviation of the irreducible target noise in MPa.
        If None, the value is inferred from the training CV residual
        RMSE supplied at call time.
    confidence : float
        Nominal confidence level, default 0.95. The lower and upper
        percentiles are (1 - confidence) / 2 and 1 - (1 - confidence) / 2
        respectively.
    random_state : int
        RNG seed for reproducible Monte Carlo draws.
    """
    n_samples: int = 1000
    sigma_eps: float | None = None
    confidence: float = 0.95
    random_state: int = 42


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------

@dataclass(frozen=True)
class BatchPrediction:
    """
    Output of predict_with_uncertainty for a single batch.

    Attributes
    ----------
    point_estimate : float
        Model prediction at the point-estimate features (no Monte
        Carlo noise applied).
    ci_lower, ci_upper : float
        Lower and upper bounds of the confidence-level prediction
        interval, computed by the percentile method on the Monte
        Carlo replicate distribution.
    mc_sd : float
        Standard deviation of the Monte Carlo replicate distribution,
        useful as a Gaussian-approximate standard uncertainty when
        the user wants a single number rather than an interval.
    n_samples : int
        Number of Monte Carlo draws used.
    sigma_eps_used : float
        The irreducible-noise SD value actually applied (resolved
        from config.sigma_eps or from the supplied fallback).
    """
    point_estimate: float
    ci_lower: float
    ci_upper: float
    mc_sd: float
    n_samples: int
    sigma_eps_used: float


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def _resolve_sigma_eps(config: PredictionConfig,
                       fallback_sigma_eps: float | None) -> float:
    """
    Resolve sigma_eps from config or fallback.

    If config.sigma_eps is given, use it. Otherwise use the
    fallback (typically the training-CV residual RMSE). Raises
    if neither is available.
    """
    if config.sigma_eps is not None:
        return float(config.sigma_eps)
    if fallback_sigma_eps is None:
        raise ValueError(
            "config.sigma_eps is None and no fallback sigma_eps "
            "was supplied. Provide one of: an explicit sigma_eps "
            "in config, or a fallback (typically the training-CV "
            "residual RMSE) to predict_with_uncertainty."
        )
    return float(fallback_sigma_eps)


def predict_with_uncertainty(
    model: Any,
    x_point: np.ndarray,
    x_sigma: np.ndarray,
    config: PredictionConfig | None = None,
    fallback_sigma_eps: float | None = None,
) -> BatchPrediction:
    """
    Compute a Monte Carlo prediction interval for one batch.

    Parameters
    ----------
    model : sklearn estimator
        A fitted model with a .predict method, typically the
        pipeline produced by train_ridge.
    x_point : np.ndarray
        Point-estimate feature vector, shape (n_features,). The
        Module 3 batch-level descriptors after Module 4 has produced
        their point estimates.
    x_sigma : np.ndarray
        Per-feature standard uncertainty, shape (n_features,). The
        Module 4 standard uncertainties on each feature; zero is
        acceptable for features with negligible measurement noise.
    config : PredictionConfig, optional
        Configuration; defaults to 1000 samples, percentile-method
        95% CI, sigma_eps inferred from fallback.
    fallback_sigma_eps : float, optional
        Used only when config.sigma_eps is None. Typically the
        training-CV residual RMSE.

    Returns
    -------
    BatchPrediction
        Point estimate plus interval and Monte Carlo SD.
    """
    if config is None:
        config = PredictionConfig()

    x_point = np.asarray(x_point, dtype=float).ravel()
    x_sigma = np.asarray(x_sigma, dtype=float).ravel()
    if x_point.shape != x_sigma.shape:
        raise ValueError(
            f"x_point shape {x_point.shape} != x_sigma shape "
            f"{x_sigma.shape}"
        )
    if np.any(x_sigma < 0):
        raise ValueError("x_sigma must be non-negative")

    sigma_eps_used = _resolve_sigma_eps(config, fallback_sigma_eps)

    rng = np.random.default_rng(config.random_state)
    K = config.n_samples
    n_features = x_point.size

    # Step 1-2: sample features around the point estimate and
    # predict for each sample. Vectorised across K samples for speed.
    if np.all(x_sigma == 0):
        # Degenerate case: no input uncertainty. The MC SD then
        # reflects only the irreducible noise.
        X_samples = np.tile(x_point, (K, 1))
    else:
        X_samples = rng.normal(
            loc=x_point[np.newaxis, :],
            scale=x_sigma[np.newaxis, :],
            size=(K, n_features),
        )
    y_hat_samples = model.predict(X_samples)

    # Step 3: add irreducible noise.
    y_tilde_samples = y_hat_samples + rng.normal(
        loc=0.0, scale=sigma_eps_used, size=K
    )

    # Step 4: percentile-method interval.
    alpha = 1.0 - config.confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower, ci_upper = np.percentile(
        y_tilde_samples, [lower_pct, upper_pct]
    )

    point_estimate = float(model.predict(x_point[np.newaxis, :])[0])
    mc_sd = float(np.std(y_tilde_samples, ddof=1))

    return BatchPrediction(
        point_estimate=point_estimate,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        mc_sd=mc_sd,
        n_samples=K,
        sigma_eps_used=sigma_eps_used,
    )


def predict_batch_with_uncertainty(
    model: Any,
    X_point: np.ndarray,
    X_sigma: np.ndarray,
    config: PredictionConfig | None = None,
    fallback_sigma_eps: float | None = None,
) -> list:
    """
    Vectorised over many batches: produce one BatchPrediction per
    input row.

    Parameters
    ----------
    model : sklearn estimator
        A fitted model with .predict method.
    X_point : np.ndarray
        Point-estimate feature matrix, shape (n_batches, n_features).
    X_sigma : np.ndarray
        Per-feature standard uncertainty matrix, shape
        (n_batches, n_features). Each row gives the uncertainties
        for its corresponding row in X_point.
    config : PredictionConfig, optional
    fallback_sigma_eps : float, optional

    Returns
    -------
    list of BatchPrediction
        One entry per input row.
    """
    X_point = np.asarray(X_point, dtype=float)
    X_sigma = np.asarray(X_sigma, dtype=float)
    if X_point.shape != X_sigma.shape:
        raise ValueError(
            f"X_point shape {X_point.shape} != X_sigma shape "
            f"{X_sigma.shape}"
        )
    return [
        predict_with_uncertainty(
            model, X_point[i], X_sigma[i], config, fallback_sigma_eps
        )
        for i in range(X_point.shape[0])
    ]