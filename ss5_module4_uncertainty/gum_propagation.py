"""
Computes the standard uncertainty on each fibre's length, width, and
aspect ratio by closed-form GUM (JCGM 100:2008) propagation:

    sigma^2(L_i)  = sigma_L_eff^2(i) + L_i^2 * sigma_cal_rel^2
    sigma^2(w_i)  = sigma_w_eff^2(i) + w_i^2 * sigma_cal_rel^2
    (sigma(AR_i) / AR_i)^2 = (sigma_L_eff(i) / L_i)^2
                           + (sigma_w_eff(i) / w_i)^2

where sigma_L_eff(i) = factor * sigma_L_base if fibre i is
high-reconstruction-flagged, else sigma_L_base; sigma_w_eff
analogously.

The aspect-ratio expression has no calibration term: the
multiplicative calibration factor cancels exactly in L/w. This is
a physical consequence of the form of the calibration error and
is the kind of result GUM is well suited to expose.

These per-fibre uncertainties are reported in their own right (and
are also useful for Module 5's data-confidence sub-score). The
underlying noise magnitudes drive the per-fibre perturbation step
of the Monte Carlo machinery.
"""

from dataclasses import dataclass

import numpy as np

from .config import UncertaintyConfig


@dataclass(frozen=True)
class PerFibreUncertainty:
    """
    Per-fibre standard uncertainties from GUM analytical propagation.

    All arrays are aligned to the input length/width arrays
    (one entry per fibre).

    Attributes
    ----------
    sigma_length, sigma_width : np.ndarray
        Absolute standard uncertainties in mm.
    sigma_aspect_ratio : np.ndarray
        Absolute standard uncertainty (dimensionless).
    sigma_length_relative, sigma_width_relative,
    sigma_aspect_ratio_relative : np.ndarray
        Relative (fractional) standard uncertainties: each absolute
        uncertainty divided by the corresponding measurement.
    """
    sigma_length: np.ndarray
    sigma_width: np.ndarray
    sigma_aspect_ratio: np.ndarray
    sigma_length_relative: np.ndarray
    sigma_width_relative: np.ndarray
    sigma_aspect_ratio_relative: np.ndarray


def compute_per_fibre_uncertainty(
    length: np.ndarray,
    width: np.ndarray,
    is_high_reconstruction: np.ndarray,
    config: UncertaintyConfig | None = None,
) -> PerFibreUncertainty:
    """
    Compute per-fibre GUM standard uncertainties for length, width,
    and aspect ratio.

    Parameters
    ----------
    length, width : np.ndarray
        1D arrays of fibre measurements in mm. Same length, all
        strictly positive.
    is_high_reconstruction : np.ndarray of bool
        Per-fibre flag from Module 2. True entries have their
        measurement-noise sigma inflated by
        config.reconstruction_inflation_factor.
    config : UncertaintyConfig, optional
        Configuration object. If None, defaults are used.

    Returns
    -------
    PerFibreUncertainty
        Six aligned arrays of per-fibre standard uncertainties.

    Raises
    ------
    ValueError
        If input arrays have mismatched lengths or contain
        non-positive length/width values.
    """
    if config is None:
        config = UncertaintyConfig()

    length = np.asarray(length, dtype=float)
    width = np.asarray(width, dtype=float)
    is_high_reconstruction = np.asarray(
        is_high_reconstruction, dtype=bool
    )

    n = length.size
    if width.size != n or is_high_reconstruction.size != n:
        raise ValueError(
            f"length, width, and is_high_reconstruction must have "
            f"the same length; got {length.size}, {width.size}, "
            f"{is_high_reconstruction.size}."
        )
    if np.any(length <= 0) or np.any(width <= 0):
        raise ValueError(
            "All length and width values must be strictly positive."
        )

    # Resolve effective base noise sigma values (Module 1 by default).
    sigma_L_base, sigma_w_base = config.resolved_noise_params()

    # Per-fibre inflation factor for high-reconstruction fibres.
    inflation = np.where(
        is_high_reconstruction,
        config.reconstruction_inflation_factor,
        1.0,
    )
    sigma_L_eff = sigma_L_base * inflation
    sigma_w_eff = sigma_w_base * inflation

    sigma_cal_rel = config.sigma_cal_relative

    # --- Length: additive noise + multiplicative calibration ---
    var_length = sigma_L_eff**2 + (length * sigma_cal_rel)**2
    sigma_length = np.sqrt(var_length)

    # --- Width: identical structure ---
    var_width = sigma_w_eff**2 + (width * sigma_cal_rel)**2
    sigma_width = np.sqrt(var_width)

    # --- Aspect ratio: calibration cancels ---
    rel_ar_squared = (
        (sigma_L_eff / length)**2 + (sigma_w_eff / width)**2
    )
    sigma_ar_relative = np.sqrt(rel_ar_squared)
    aspect_ratio = length / width
    sigma_aspect_ratio = sigma_ar_relative * aspect_ratio

    return PerFibreUncertainty(
        sigma_length=sigma_length,
        sigma_width=sigma_width,
        sigma_aspect_ratio=sigma_aspect_ratio,
        sigma_length_relative=sigma_length / length,
        sigma_width_relative=sigma_width / width,
        sigma_aspect_ratio_relative=sigma_ar_relative,
    )