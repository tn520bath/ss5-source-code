"""
Computes mass-weighted versions of the length descriptors using the
cylindrical mass proxy m_i = L_i * w_i^2. Mass-weighting is applied to length only:
mass-weighted width and mass-weighted aspect ratio are conceptually
ill-posed (the weight already involves the quantity being averaged).

Provides:

    compute_mass_weighted_descriptors(length, width)
        Returns a dict of mass-weighted descriptors for the length
        quantity.
"""

import numpy as np


def _weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percent: float,
) -> float:
    """
    Compute a single weighted percentile using linear interpolation
    between adjacent cumulative-weight bins.

    The midpoint normalisation (cum_w - 0.5 * w) ensures that with
    uniform weights this function reproduces np.percentile's default
    linear-interpolation result, so that 'mass-weighted' reduces
    cleanly to 'unweighted' when all fibres have equal mass.

    Parameters
    ----------
    values : np.ndarray
        1D array of values.
    weights : np.ndarray
        1D array of non-negative weights, same length as values.
    percent : float
        Percentile to compute (0 to 100).

    Returns
    -------
    float
        The weighted percentile.
    """
    sort_idx = np.argsort(values)
    v_sorted = values[sort_idx]
    w_sorted = weights[sort_idx]
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    cum_w_norm = (cum_w - 0.5 * w_sorted) / total_w
    return float(np.interp(percent / 100.0, cum_w_norm, v_sorted))


def compute_mass_weighted_descriptors(
    length: np.ndarray,
    width: np.ndarray,
) -> dict:
    """
    Compute mass-weighted descriptor statistics for fibre length.

    The per-fibre mass weight is m_i = L_i * w_i^2 (cylindrical fibre
    approximation; density and pi/4 cancel during normalisation).

    Parameters
    ----------
    length : np.ndarray
        1D array of fibre lengths in mm. Must contain only positive
        values (Module 3 enforces this).
    width : np.ndarray
        1D array of fibre widths in mm. Must contain only positive
        values, same length as `length`.

    Returns
    -------
    dict
        Mass-weighted descriptors for length, with keys:
            'n'             : number of fibres (informational)
            'total_mass'    : sum of m_i (proxy mass; arbitrary units)
            'mean_mass_w'   : mass-weighted mean length
            'std_mass_w'    : mass-weighted standard deviation
            'cv_mass_w'     : mass-weighted coefficient of variation
            'D10_mass_w'    : mass-weighted 10th percentile
            'D50_mass_w'    : mass-weighted 50th percentile
            'D90_mass_w'    : mass-weighted 90th percentile

    Raises
    ------
    ValueError
        If input arrays are empty or have mismatched lengths, or if
        the total mass is zero (cannot normalise).
    """
    length = np.asarray(length, dtype=float)
    width = np.asarray(width, dtype=float)

    if length.size == 0 or width.size == 0:
        raise ValueError(
            "Cannot compute mass-weighted descriptors on empty arrays."
        )
    if length.shape != width.shape:
        raise ValueError(
            "length and width must have the same shape; got "
            f"{length.shape} and {width.shape}."
        )

    # Cylindrical mass proxy
    weights = length * width ** 2
    total_mass = float(np.sum(weights))

    if total_mass <= 0:
        raise ValueError(
            "Total mass is non-positive; cannot normalise weights."
        )

    n = length.size

    # Weighted mean
    mean_mw = float(np.average(length, weights=weights))

    # Weighted variance and SD
    if n > 1:
        # Standard weighted variance with reliability weights:
        # sigma^2 = sum(w_i * (x_i - x_mean)^2) / sum(w_i)
        #
        # We use the population form here (no Bessel-style correction
        # for weighted samples). This is consistent with the
        # convention used in commercial fibre analysers and avoids
        # ambiguity over which "effective N" to use for the correction.
        diffs = length - mean_mw
        var_mw = float(np.sum(weights * diffs ** 2) / total_mass)
        std_mw = float(np.sqrt(var_mw))
        cv_mw = std_mw / mean_mw if mean_mw != 0 else float("nan")
    else:
        std_mw = float("nan")
        cv_mw = float("nan")

    # Weighted percentiles
    d10 = _weighted_percentile(length, weights, 10.0)
    d50 = _weighted_percentile(length, weights, 50.0)
    d90 = _weighted_percentile(length, weights, 90.0)

    return {
        "n": int(n),
        "total_mass": total_mass,
        "mean_mass_w": mean_mw,
        "std_mass_w": std_mw,
        "cv_mass_w": cv_mw,
        "D10_mass_w": d10,
        "D50_mass_w": d50,
        "D90_mass_w": d90,
    }