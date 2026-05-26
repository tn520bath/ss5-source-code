"""
Measurement noise models for the synthetic data generator.

Implements additive Gaussian noise applied to true fibre dimensions
to simulate the output of Sub-system 4's image-processing pipeline.
The noise model is intended to capture the 
edge-localisation uncertainty that dominates random measurement
error.
"""

import numpy as np

from .config import MeasurementNoiseParams


def apply_measurement_noise(
    true_lengths: np.ndarray,
    true_widths: np.ndarray,
    params: MeasurementNoiseParams,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply additive Gaussian measurement noise to true fibre dimensions.

    Length and width noise are sampled independently. No post-noise
    clipping is performed: a small fraction of measured values may
    fall outside the system's specified measurement range, mirroring
    real measurement behaviour.

    Parameters
    ----------
    true_lengths : np.ndarray
        Array of true fibre lengths in mm (e.g., from sample_lengths).
    true_widths : np.ndarray
        Array of true fibre widths in mm (e.g., from sample_widths).
        Must be the same length as true_lengths.
    params : MeasurementNoiseParams
        Noise parameter values.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility. If None, a new
        one is created.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (measured_lengths, measured_widths), each of length n.

    Raises
    ------
    ValueError
        If true_lengths and true_widths have different lengths.
    """
    if len(true_lengths) != len(true_widths):
        raise ValueError(
            f"true_lengths and true_widths must have the same length, "
            f"got {len(true_lengths)} and {len(true_widths)}."
        )

    if rng is None:
        rng = np.random.default_rng()

    n = len(true_lengths)

    length_noise = rng.normal(loc=0.0, scale=params.sigma_length_mm, size=n)
    width_noise = rng.normal(loc=0.0, scale=params.sigma_width_mm, size=n)

    measured_lengths = true_lengths + length_noise
    measured_widths = true_widths + width_noise

    return measured_lengths, measured_widths