"""
Provides two functions:

    compute_empirical_descriptors(values)
        Computes mean, median, SD, CV, skewness, and D10/D50/D90 for a
        1D array of measurements (length, width, or aspect ratio).
        Pure function. No model assumptions.

    compute_reconstruction_summary(reconstructed_fraction, is_high)
        Computes the per-batch reconstruction-quality summary: mean
        reconstructed fraction and fraction of fibres flagged as
        high-reconstruction by Module 2.
"""


import numpy as np
from scipy import stats

from .config import DescriptorConfig


def compute_empirical_descriptors(
    values: np.ndarray,
    config: DescriptorConfig | None = None,
) -> dict:
    """
    Compute empirical descriptor statistics for a 1D array of
    fibre measurements.

    Parameters
    ----------
    values : np.ndarray
        1D array of measurements (e.g., lengths in mm). Assumed to be
        clean: no NaN, no non-positive values, no missing entries.
        Module 2 is responsible for upstream validation.

    config : DescriptorConfig, optional
        Configuration object. If None, defaults are used.

    Returns
    -------
    dict
        Descriptor statistics with keys:
            'n'      : sample size
            'mean'   : arithmetic mean
            'median' : 50th percentile (= D50)
            'std'    : sample standard deviation (Bessel-corrected)
            'cv'     : coefficient of variation (std / mean)
            'skew'   : Fisher-Pearson bias-corrected skewness
            'D10'    : 10th percentile
            'D50'    : 50th percentile (median, repeated for clarity)
            'D90'    : 90th percentile

    Notes
    -----
    All percentiles use linear interpolation between data points
    (NumPy default). Skewness uses the bias-corrected estimator
    (scipy default with `bias=False`). Standard deviation uses
    Bessel's correction (numpy ddof=1).
    """
    if config is None:
        config = DescriptorConfig()

    values = np.asarray(values)
    n = values.size

    if n == 0:
        raise ValueError("Cannot compute descriptors on empty array.")

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    cv = std / mean if mean != 0 and not np.isnan(std) else float("nan")
    skew = float(stats.skew(values, bias=False)) if n > 2 else float("nan")
    percentiles = np.percentile(values, config.percentiles)

    return {
        "n": int(n),
        "mean": mean,
        "median": float(np.median(values)),
        "std": std,
        "cv": cv,
        "skew": skew,
        "D10": float(percentiles[0]),
        "D50": float(percentiles[1]),
        "D90": float(percentiles[2]),
    }


def compute_reconstruction_summary(
    reconstructed_fraction: np.ndarray,
    is_high_reconstruction: np.ndarray,
) -> dict:
    """
    Compute the per-batch reconstruction-quality summary.

    Parameters
    ----------
    reconstructed_fraction : np.ndarray
        1D array of reconstructed-fraction values, one per fibre.
        Values in [0, 1].

    is_high_reconstruction : np.ndarray
        1D boolean array, one per fibre. True if Module 2 flagged the
        fibre as having reconstructed_fraction above its configured
        threshold.

    Returns
    -------
    dict
        Reconstruction summary with keys:
            'mean_reconstructed_fraction' : average across all fibres
            'high_reconstruction_fraction' : fraction of fibres flagged
            'n_flagged' : count of fibres flagged
    """
    reconstructed_fraction = np.asarray(reconstructed_fraction)
    is_high_reconstruction = np.asarray(is_high_reconstruction, dtype=bool)

    n = reconstructed_fraction.size
    if n == 0:
        raise ValueError(
            "Cannot compute reconstruction summary on empty array."
        )

    return {
        "mean_reconstructed_fraction": float(
            np.mean(reconstructed_fraction)
        ),
        "high_reconstruction_fraction": float(
            np.mean(is_high_reconstruction)
        ),
        "n_flagged": int(np.sum(is_high_reconstruction)),
    }