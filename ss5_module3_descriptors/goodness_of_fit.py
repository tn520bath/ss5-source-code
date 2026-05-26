"""
Computes goodness-of-fit metrics for a fitted distribution against
the data it was fitted to:

  - AIC (Akaike Information Criterion): used to compare competing
    distributions for the same quantity. Lower AIC is better.
    Computed from the log-likelihood already stored in the
    DistributionFit, with k=2 fitted parameters.

  - Kolmogorov-Smirnov (KS) test: gives an absolute measure of fit
    quality. The KS statistic D is the maximum vertical distance
    between the empirical and fitted CDFs. The associated p-value
    tests the null hypothesis that the data came from the fitted
    distribution.

Note on KS p-value interpretation: the standard KS test assumes the
distribution parameters were specified independently of the data.
When parameters are estimated from the same sample (as here), the
test becomes conservative — p-values are biased upward. The strict
correction is the Lilliefors test, but for the descriptive use of
KS in this module (alongside AIC and visual QQ inspection) the
standard form is sufficient and is the convention in the wood-fibre
literature.

Provides:

    compute_gof(fit, values)
        Returns a dict of GoF diagnostics for one fit.

    compare_aic(gof_a, gof_b)
        Returns delta-AIC and the preferred distribution name for
        two competing fits of the same quantity.
"""

import numpy as np
from scipy import stats

from .distribution_fitting import DistributionFit

# Number of fitted parameters for the two-parameter lognormal and
# two-parameter Weibull (with floc fixed at 0). If a future
# distribution has more parameters, this constant will need to
# vary by family.
_N_PARAMS = 2


def compute_gof(
    fit: DistributionFit,
    values: np.ndarray,
) -> dict:
    """
    Compute goodness-of-fit diagnostics for a single distribution fit.

    Parameters
    ----------
    fit : DistributionFit
        Result from fit_distribution.
    values : np.ndarray
        The 1D data array the distribution was fitted to.

    Returns
    -------
    dict
        GoF diagnostics with keys:
            'distribution_name' : 'lognormal' or 'weibull'
            'k_params'          : number of fitted parameters (= 2)
            'log_likelihood'    : echoed from the fit object
            'aic'               : 2*k - 2*log_likelihood
            'ks_D'              : KS test statistic
            'ks_p'              : KS test p-value (conservative
                                  when parameters are estimated
                                  from the same data)

    Raises
    ------
    ValueError
        If the data array is empty or its size differs from the
        sample size used at fit time.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot compute GoF on an empty data array.")
    if values.size != fit.n:
        raise ValueError(
            f"Data array size ({values.size}) does not match the "
            f"size used at fit time ({fit.n})."
        )

    aic = 2 * _N_PARAMS - 2 * fit.log_likelihood

    ks_result = stats.kstest(values, fit.distribution.cdf)
    ks_D = float(ks_result.statistic)
    ks_p = float(ks_result.pvalue)

    return {
        "distribution_name": fit.name,
        "k_params": _N_PARAMS,
        "log_likelihood": fit.log_likelihood,
        "aic": float(aic),
        "ks_D": ks_D,
        "ks_p": ks_p,
    }


def compare_aic(gof_a: dict, gof_b: dict) -> dict:
    """
    Compare two GoF results by AIC.

    Convention:
        delta_aic = aic_a - aic_b
        positive => B preferred (lower AIC is better)
        negative => A preferred

    Parameters
    ----------
    gof_a, gof_b : dict
        GoF dicts from compute_gof.

    Returns
    -------
    dict
        Comparison summary with keys:
            'delta_aic'        : aic_a - aic_b
            'preferred'        : name of the distribution with lower AIC
            'evidence_strength': qualitative label:
                                   |delta| < 2  : 'indistinguishable'
                                   2 <= |d| < 4 : 'weak'
                                   4 <= |d| < 7 : 'substantial'
                                   |d| >= 7     : 'strong'
    """
    delta = gof_a["aic"] - gof_b["aic"]
    preferred = gof_b["distribution_name"] if delta > 0 else gof_a["distribution_name"]
    abs_delta = abs(delta)

    if abs_delta < 2:
        strength = "indistinguishable"
    elif abs_delta < 4:
        strength = "weak"
    elif abs_delta < 7:
        strength = "substantial"
    else:
        strength = "strong"

    return {
        "delta_aic": float(delta),
        "preferred": preferred,
        "evidence_strength": strength,
    }