"""
Probability distributions for synthetic fibre measurements.

Implements the parametric distributions used to generate 'true' (noise-free)
fibre length and width values for the synthetic data generator.

Primary distribution: lognormal.
Secondary distribution (length only): Weibull (for sensitivity analysis and comparison).
"""

import numpy as np
from scipy import stats

from .config import LengthDistributionParams, WidthDistributionParams


def sample_lengths(
    n: int,
    params: LengthDistributionParams,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw n 'true' fibre lengths from the configured length distribution.

    The values returned represent the underlying physical fibre lengths
    before any measurement noise is applied.

    Truncation is applied to keep all returned values within
    [params.min_length_mm, params.max_length_mm]. Truncation is implemented
    by rejection sampling rather than clipping, to avoid distorting the
    distribution shape near the boundaries.

    Parameters
    ----------
    n : int
        Number of fibre lengths to draw.
    params : LengthDistributionParams
        Distribution choice and parameter values.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility. If None, a new one
        is created.

    Returns
    -------
    np.ndarray
        Array of length n containing fibre lengths in millimetres.

    Raises
    ------
    ValueError
        If params.distribution is not 'lognormal' or 'weibull'.
    """
    if rng is None:
        rng = np.random.default_rng()

    if params.distribution == "lognormal":
        return _sample_lognormal_truncated(n, params, rng)
    elif params.distribution == "weibull":
        return _sample_weibull_truncated(n, params, rng)
    else:
        raise ValueError(
            f"Unknown distribution '{params.distribution}'. "
            f"Expected 'lognormal' or 'weibull'."
        )


def _sample_lognormal_truncated(
    n: int,
    params: LengthDistributionParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw n samples from a lognormal distribution, truncated to
    [min_length_mm, max_length_mm] by rejection sampling.   
    """
    # We over-sample to reduce the number of rejection-sampling rounds.
    # The fraction of samples within the truncation range is given by
    # the lognormal CDF; we estimate this and oversample accordingly.
    lower_cdf = stats.lognorm.cdf(
        params.min_length_mm, s=params.sigma, scale=np.exp(params.mu)
    )
    upper_cdf = stats.lognorm.cdf(
        params.max_length_mm, s=params.sigma, scale=np.exp(params.mu)
    )
    accept_fraction = upper_cdf - lower_cdf
    oversample = max(1.2, 1.0 / max(accept_fraction, 1e-3))

    samples_collected = []
    while len(samples_collected) < n:
        # rng.lognormal expects mean and sigma of the underlying normal distribution
        candidates = rng.lognormal(
            mean=params.mu,
            sigma=params.sigma,
            size=int(n * oversample),
        )
        in_range = (candidates >= params.min_length_mm) & (
            candidates <= params.max_length_mm
        )
        samples_collected.extend(candidates[in_range].tolist())

    return np.array(samples_collected[:n])


def _sample_weibull_truncated(
    n: int,
    params: LengthDistributionParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw n samples from a Weibull distribution, truncated to
    [min_length_mm, max_length_mm] by rejection sampling.
    """
    lower_cdf = stats.weibull_min.cdf(
        params.min_length_mm, c=params.weibull_shape, scale=params.weibull_scale
    )
    upper_cdf = stats.weibull_min.cdf(
        params.max_length_mm, c=params.weibull_shape, scale=params.weibull_scale
    )
    accept_fraction = upper_cdf - lower_cdf
    oversample = max(1.2, 1.0 / max(accept_fraction, 1e-3))

    samples_collected = []
    while len(samples_collected) < n:
        candidates = params.weibull_scale * rng.weibull(
            a=params.weibull_shape,
            size=int(n * oversample),
        )
        in_range = (candidates >= params.min_length_mm) & (
            candidates <= params.max_length_mm
        )
        samples_collected.extend(candidates[in_range].tolist())

    return np.array(samples_collected[:n])


def sample_widths(
    n: int,
    params: WidthDistributionParams,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw n 'true' fibre widths from a truncated lognormal distribution.

    Width is sampled independently of length. As with length, the
    values returned represent underlying physical fibre widths before
    measurement noise is applied.

    Truncation is applied to keep all returned values within
    [params.min_width_mm, params.max_width_mm], implemented by
    rejection sampling rather than clipping.

    Parameters
    ----------
    n : int
        Number of fibre widths to draw.
    params : WidthDistributionParams
        Distribution parameter values.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility. If None, a new
        one is created.

    Returns
    -------
    np.ndarray
        Array of length n containing fibre widths in millimetres.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Estimate the fraction of mass within the truncation range so we
    # can over-sample efficiently in a single rejection-sampling round.
    lower_cdf = stats.lognorm.cdf(
        params.min_width_mm, s=params.sigma, scale=np.exp(params.mu)
    )
    upper_cdf = stats.lognorm.cdf(
        params.max_width_mm, s=params.sigma, scale=np.exp(params.mu)
    )
    accept_fraction = upper_cdf - lower_cdf
    oversample = max(1.2, 1.0 / max(accept_fraction, 1e-3))

    samples_collected = []
    while len(samples_collected) < n:
        candidates = rng.lognormal(
            mean=params.mu,
            sigma=params.sigma,
            size=int(n * oversample),
        )
        in_range = (candidates >= params.min_width_mm) & (
            candidates <= params.max_width_mm
        )
        samples_collected.extend(candidates[in_range].tolist())

    return np.array(samples_collected[:n])