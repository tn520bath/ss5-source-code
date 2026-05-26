"""
Literature-grounded synthetic target-variable generation.

This module produces a synthetic MOR-like target value (in
MPa-equivalent units) for each image, attached to the message under
the 'synthetic_targets' key. The target serves as the training
label for the predictive modelling work in Module 6.

The descriptors used here (D50 of length, D50 of aspect ratio,
fines fraction, CV of length) are computed from the per-fibre
records of a single image so that the synthetic generator remains
self-contained. They are simpler proxies than the corresponding
descriptors Module 3 produces from a validated batch, but the two
are conceptually aligned: in deployment Module 6 would consume
Module 3's batch-level descriptors against this same target.
"""

import numpy as np

from .config import TargetGenerationParams, LiteratureGroundedTargetSpec


# -------------------------------------------------------------------
# Descriptor helpers
# -------------------------------------------------------------------

def _saturating_ramp(x: float, x_sat: float) -> float:
    """
    Saturating ramp S(x; x_sat) = min(max(x, 0), x_sat).

    Returns x for 0 <= x <= x_sat and x_sat for x > x_sat. Negative
    inputs are clipped to zero (defensive, since the descriptors
    that drive the target are positive by construction).
    """
    return float(min(max(x, 0.0), x_sat))


def _compute_descriptors(message: dict) -> dict:
    """
    Compute per-image descriptors used by the literature-grounded
    target.

    Extracts:
      - D50_length      : median fibre length (mm)
      - D50_aspect      : median aspect ratio (dimensionless)
      - fines_fraction  : count fraction of fibres below a 1 mm
                          threshold
      - CV_length       : sample SD divided by sample mean of length
                          (dimensionless dispersion)
    """
    fibres = message["char"]
    lengths = np.array([f["dimensions"]["length"] for f in fibres])
    widths = np.array([f["dimensions"]["width"] for f in fibres])
    aspect_ratios = lengths / widths

    mean_length = float(np.mean(lengths))
    sd_length = float(np.std(lengths, ddof=1)) if lengths.size > 1 else 0.0

    return {
        "D50_length": float(np.median(lengths)),
        "D50_aspect": float(np.median(aspect_ratios)),
        "fines_fraction": float(np.mean(lengths < 1.0)),
        "CV_length": sd_length / mean_length if mean_length > 0 else 0.0,
    }


# -------------------------------------------------------------------
# Deterministic part of the target
# -------------------------------------------------------------------

def _evaluate_target(descriptors: dict,
                     spec: LiteratureGroundedTargetSpec) -> float:
    """
    Compute the deterministic part of the literature-grounded
    target, before noise is added.

    Returns the noise-free MOR-like value (MPa) implied by the
    given descriptors under the given spec.
    """
    return (
        spec.beta_0
        + spec.beta_length * _saturating_ramp(
            descriptors["D50_length"], spec.length_sat_mm
        )
        + spec.beta_ar * _saturating_ramp(
            descriptors["D50_aspect"], spec.ar_sat
        )
        - spec.beta_fines * descriptors["fines_fraction"]
        - spec.beta_cv * descriptors["CV_length"]
    )


# -------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------

def attach_synthetic_targets(
    message: dict,
    params: TargetGenerationParams,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Compute the synthetic target for one image and attach it to the
    message under 'synthetic_targets'.

    Output structure:
        message['synthetic_targets'] = {
            params.spec.name: <noisy target value>,
            params.spec.name + '_truth': <noise-free target value>,
            params.spec.name + '_descriptors': {...},
        }

    The '_truth' field is included so that downstream validation
    (Module 6) can compare the model's recovered coefficients
    against the underlying noise-free relationship without having
    to re-derive the descriptors.

    If params.enabled is False, the message is returned unchanged.

    Parameters
    ----------
    message : dict
        A processing_result message produced by generate_image.
    params : TargetGenerationParams
        Target generation configuration.
    rng : np.random.Generator, optional
        NumPy random generator for reproducible noise.

    Returns
    -------
    dict
        The message with an added 'synthetic_targets' field if
        params.enabled is True; the message unchanged otherwise.
    """
    if not params.enabled:
        return message

    if rng is None:
        rng = np.random.default_rng()

    spec = params.spec
    descriptors = _compute_descriptors(message)
    truth = _evaluate_target(descriptors, spec)
    noise = float(rng.normal(loc=0.0, scale=spec.noise_std))

    message["synthetic_targets"] = {
        spec.name: truth + noise,
        spec.name + "_truth": truth,
        spec.name + "_descriptors": descriptors,
    }
    return message