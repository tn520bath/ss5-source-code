"""
Fits a two-parameter lognormal or two-parameter Weibull distribution
to a 1D sample of fibre measurements via maximum likelihood
estimation. Returns a DistributionFit dataclass containing the
frozen scipy distribution, fitted parameters, log-likelihood at the
MLE, and sample size, ready for downstream consumption by:

  - goodness-of-fit, which uses the log-likelihood
    for AIC and the frozen distribution's CDF for the KS test.
  - Module 4 (uncertainty quantification), which calls .rvs() on
    the frozen distribution during Monte Carlo propagation.
  - Module 6 (predictive modelling), which uses the fitted
    parameters as features for the ML pipeline.

The lognormal and Weibull are fitted with their location parameter
fixed at zero (floc=0). This matches the two-parameter forms used
in the wood-fibre literature and makes fitted parameters directly
comparable across batches and to published values.

Provides:

    fit_distribution(values, distribution, config)
        Fits the requested distribution and returns a DistributionFit.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from .config import DescriptorConfig


@dataclass(frozen=True)
class DistributionFit:
    """
    Container for the result of a single distribution fit.

    Attributes
    ----------
    name : str
        Distribution family ('lognormal' or 'weibull').
    distribution : scipy.stats._distn_infrastructure.rv_frozen
        Frozen scipy distribution at the fitted parameters. Supports
        .pdf(), .cdf(), .ppf(), .rvs(), .mean(), .std(), etc.
    params : dict
        Fitted parameters in their canonical form:
          - lognormal: {'mu': ..., 'sigma': ...}
          - weibull:   {'k': ..., 'lambda': ...}
    log_likelihood : float
        Log-likelihood evaluated at the fitted parameters,
        sum_i log f(x_i; theta_hat).
    n : int
        Sample size used for fitting.
    """
    name: str
    distribution: Any
    params: dict
    log_likelihood: float
    n: int


def fit_distribution(
    values: np.ndarray,
    distribution: str = "lognormal",
    config: DescriptorConfig | None = None,
) -> DistributionFit:
    """
    Fit a two-parameter distribution to a 1D sample by MLE.

    Parameters
    ----------
    values : np.ndarray
        1D array of positive measurements.
    distribution : {'lognormal', 'weibull'}, default 'lognormal'
        Distribution family to fit.
    config : DescriptorConfig, optional
        Configuration object. If None, defaults are used. The
        small-N warning threshold is read from this object.

    Returns
    -------
    DistributionFit
        Fit result; see DistributionFit docstring.

    Raises
    ------
    ValueError
        If `values` is empty, has fewer than two distinct points,
        or contains non-positive entries (lognormal/Weibull have
        positive support).
    ValueError
        If `distribution` is not a recognised family.

    Warns
    -----
    UserWarning
        If `n` is below `config.small_n_warning_threshold`. The fit
        is still produced; suppression is the responsibility of
        Module 8.
    """
    if config is None:
        config = DescriptorConfig()

    values = np.asarray(values, dtype=float)
    n = values.size

    if n < 2:
        raise ValueError(
            f"At least two values are required to fit a "
            f"two-parameter distribution; got {n}."
        )
    if np.any(values <= 0):
        raise ValueError(
            "All values must be positive for lognormal/Weibull "
            "fitting (positive-support distributions)."
        )
    if np.std(values) == 0:
        raise ValueError(
            "All input values are identical; distribution fit is "
            "degenerate."
        )

    if n < config.small_n_warning_threshold:
        warnings.warn(
            f"Fitting distribution to N={n} points (below threshold "
            f"of {config.small_n_warning_threshold}). Result may be "
            "unreliable.",
            stacklevel=2,
        )

    if distribution == "lognormal":
        # scipy lognorm with floc=0 gives the two-parameter form:
        #   X = scale * exp(s * Z), Z ~ N(0, 1)
        # so s = sigma and scale = exp(mu).
        s, loc, scale = stats.lognorm.fit(values, floc=0)
        mu = float(np.log(scale))
        sigma = float(s)
        frozen = stats.lognorm(s=sigma, scale=np.exp(mu))
        params = {"mu": mu, "sigma": sigma}

    elif distribution == "weibull":
        # scipy weibull_min with floc=0 gives the two-parameter form:
        #   f(x; k, lambda) = (k/lambda)(x/lambda)^(k-1) exp(-(x/lambda)^k)
        # where c = k (shape) and scale = lambda.
        c, loc, scale = stats.weibull_min.fit(values, floc=0)
        k = float(c)
        lam = float(scale)
        frozen = stats.weibull_min(c=k, scale=lam)
        params = {"k": k, "lambda": lam}

    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Expected 'lognormal' or 'weibull'."
        )

    log_likelihood = float(np.sum(frozen.logpdf(values)))

    return DistributionFit(
        name=distribution,
        distribution=frozen,
        params=params,
        log_likelihood=log_likelihood,
        n=int(n),
    )