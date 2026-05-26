"""
Reports the fraction of fibres below a configurable length threshold,
in two complementary forms:

  - Count-based: the proportion of fibres in the batch shorter than
    the threshold (the "P200" / "<1 mm" style metric).
  - Mass-based:  the proportion of total batch mass contributed by
    fibres shorter than the threshold.

The two metrics often differ substantially. A batch dominated by
short fibres in count terms can still be dominated by long fibres in
mass terms, and it is the mass-based number that more directly
reflects what is pressed into the resulting MDF board.

The default threshold is 1 mm, which aligns with SYS-PERF-01 (the
lower bound of the system's reliable measurement range). This is
currently an engineering assumption that should be calibrated once
real Fab process data becomes available.

Provides:

    compute_fines_fraction(length, width, threshold_mm=1.0)
        Returns a dict of count- and mass-based fines fractions.
"""

import numpy as np


def compute_fines_fraction(
    length: np.ndarray,
    width: np.ndarray,
    threshold_mm: float = 1.0,
) -> dict:
    """
    Compute count-based and mass-based fines fractions.

    Parameters
    ----------
    length : np.ndarray
        1D array of fibre lengths in mm. Positive values only
        (Module 2 enforces this).
    width : np.ndarray
        1D array of fibre widths in mm, same shape as `length`.
    threshold_mm : float, optional
        Length below which a fibre is classified as a "fine".
        Strict comparison (length < threshold). Defaults to 1.0 mm.

    Returns
    -------
    dict
        Fines summary with keys:
            'threshold_mm'        : threshold used (echoed for traceability)
            'n_fines'             : count of fibres below threshold
            'n_total'             : total fibre count
            'fines_fraction_count': n_fines / n_total
            'fines_fraction_mass' : mass below threshold / total mass

    Raises
    ------
    ValueError
        If input arrays are empty, mismatched in shape, or yield
        non-positive total mass.
    """
    length = np.asarray(length, dtype=float)
    width = np.asarray(width, dtype=float)

    if length.size == 0 or width.size == 0:
        raise ValueError(
            "Cannot compute fines fraction on empty arrays."
        )
    if length.shape != width.shape:
        raise ValueError(
            "length and width must have the same shape; got "
            f"{length.shape} and {width.shape}."
        )

    n_total = length.size
    is_fine = length < threshold_mm
    n_fines = int(np.sum(is_fine))

    # Mass proxy: m_i = L_i * w_i^2 (constants cancel)
    weights = length * width ** 2
    total_mass = float(np.sum(weights))

    if total_mass <= 0:
        raise ValueError(
            "Total mass is non-positive; cannot compute mass-based "
            "fines fraction."
        )

    fines_mass = float(np.sum(weights[is_fine]))

    return {
        "threshold_mm": float(threshold_mm),
        "n_fines": n_fines,
        "n_total": int(n_total),
        "fines_fraction_count": n_fines / n_total,
        "fines_fraction_mass": fines_mass / total_mass,
    }