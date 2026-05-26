"""
Quality-metadata simulation for the synthetic data generator.

Generates per-fibre metadata fields beyond raw geometry, matching
the agreed Sub-system 4 -> Sub-system 5 interface contract:

    - reconstructed_fraction (per fibre): float in [0, 1)
    - pixel_size_mm (per batch): float, mm per pixel

Plus identifier fields:

    - mesh_id (per fibre): integer, sequential within a batch
    - image_id (per batch): integer, currently assumed constant
      within a batch (one image per batch; multi-image batches
      are a planned extension)
    - batch_id (per batch): integer, supplied by the caller

The quality-metadata model is grounded in engineering reasoning: 
published data on overlap and reconstruction rates in mechanically 
recycled textile imaging does not exist at the resolution required, 
and the actual rates in deployment will be dominated by the performance 
of upstream sample-preparation subsystems. Defaults are stated
assumptions, deliberately exposed for sensitivity analysis.
"""

import numpy as np

from .config import QualityMetadataParams


def sample_reconstructed_fractions(
    n: int,
    params: QualityMetadataParams,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw n reconstructed_fraction values using the two-stage model.

    Each fibre is independently flagged as reconstructed with
    probability p_reconstructed. Flagged fibres draw their fraction
    from Beta(alpha, beta); unflagged fibres have fraction = 0.

    Parameters
    ----------
    n : int
        Number of fibres.
    params : QualityMetadataParams
        Quality-metadata parameter values.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of length n containing reconstructed_fraction values
        in [0, 1). Approximately (1 - p_reconstructed) * n of these
        will be exactly 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Stage 1: which fibres are reconstructed at all?
    is_reconstructed = rng.random(size=n) < params.p_reconstructed

    # Stage 2: for flagged fibres, draw the fraction from Beta
    fractions = np.zeros(n)
    n_reconstructed = int(np.sum(is_reconstructed))
    if n_reconstructed > 0:
        fractions[is_reconstructed] = rng.beta(
            a=params.beta_alpha,
            b=params.beta_beta,
            size=n_reconstructed,
        )

    return fractions


def assign_batch_metadata(
    n_fibres: int,
    batch_id: int,
    image_id: int,
    params: QualityMetadataParams,
) -> dict:
    """
    Assemble per-batch and per-fibre identifier metadata.

    Returns a dict with the per-fibre arrays and per-batch scalars
    that downstream code will assemble into Sub-system 4's nested
    output schema.

    Parameters
    ----------
    n_fibres : int
        Number of fibres in the batch.
    batch_id : int
        Identifier for this batch.
    image_id : int
        Identifier for the source image. Currently a single value
        per batch (one-image-per-batch simplification); will be
        extended in subsequent versions.
    params : QualityMetadataParams
        Quality-metadata parameters (used for pixel_size_mm).

    Returns
    -------
    dict
        Keys:
            mesh_ids       : np.ndarray of length n_fibres
            image_id       : int (per batch)
            batch_id       : int (per batch)
            pixel_size_mm  : float (per batch)
    """
    return {
        "mesh_ids": np.arange(n_fibres),
        "image_id": image_id,
        "batch_id": batch_id,
        "pixel_size_mm": params.pixel_size_mm,
    }