"""
Batch (image) assembly for the synthetic data generator.

Composes the outputs of the distribution, noise, and quality-metadata
modules into a single message matching the Sub-system 4
'processing_result' schema. The resulting message is structurally
indistinguishable from real SS4 output, allowing downstream SS5 code
to be tested against synthetic data and then handle real data with
no modification.

batch_id is intentionally not present in the message: aggregation of
images into batches is a SS5 concern, not a SS4 concern.
"""

import numpy as np

from .config import (
    LengthDistributionParams,
    WidthDistributionParams,
    MeasurementNoiseParams,
    QualityMetadataParams,
    TargetGenerationParams,
)
from .distributions import sample_lengths, sample_widths
from .noise_models import apply_measurement_noise
from .quality_metadata import sample_reconstructed_fractions
from .targets import attach_synthetic_targets


def generate_image(
    n_fibres: int = 500,
    image_id: int = 0,
    length_params: LengthDistributionParams | None = None,
    width_params: WidthDistributionParams | None = None,
    noise_params: MeasurementNoiseParams | None = None,
    quality_params: QualityMetadataParams | None = None,
    target_params: TargetGenerationParams | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Generate one synthetic 'processing_result' message representing
    one image's worth of fibres.

    Composes the four upstream components in order:
      1. Sample true lengths and widths from the configured
         distributions.
      2. Apply additive Gaussian measurement noise.
      3. Sample reconstructed_fraction values for each fibre.
      4. Assemble the per-fibre records into the SS4 schema.

    All sub-component random draws share the same generator, so
    passing a seeded rng makes the entire message reproducible.

    Parameters
    ----------
    n_fibres : int
        Number of fibres in the generated image. Default 500.
    image_id : int
        Identifier for this image. Default 0; caller increments
        for sequential images.
    length_params, width_params, noise_params, quality_params :
        Parameter dataclasses for each upstream component. If any
        is None, the default values from config.py are used.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility. If None, a new
        one is created.

    Returns
    -------
    dict
        A dictionary matching the SS4 'processing_result' schema.

    Raises
    ------
    ValueError
        If n_fibres < 1.
    """
    if n_fibres < 1:
        raise ValueError(f"n_fibres must be >= 1, got {n_fibres}.")

    if rng is None:
        rng = np.random.default_rng()

    # Use default parameters if any are not supplied
    if length_params is None:
        length_params = LengthDistributionParams()
    if width_params is None:
        width_params = WidthDistributionParams()
    if noise_params is None:
        noise_params = MeasurementNoiseParams()
    if quality_params is None:
        quality_params = QualityMetadataParams()
    if target_params is None:
        target_params = TargetGenerationParams()

    # Stage 1: true geometric values from underlying distributions
    true_lengths = sample_lengths(n_fibres, length_params, rng)
    true_widths = sample_widths(n_fibres, width_params, rng)

    # Stage 2: simulate measurement noise from SS4's segmentation
    measured_lengths, measured_widths = apply_measurement_noise(
        true_lengths, true_widths, noise_params, rng
    )

    # Stage 3: simulate quality metadata
    reconstructed_fractions = sample_reconstructed_fractions(
        n_fibres, quality_params, rng
    )

    # Stage 4: assemble per-fibre records into the SS4 schema
    char = [
        {
            "mesh_id": int(i),
            "dimensions": {
                "length": float(measured_lengths[i]),
                "width": float(measured_widths[i]),
            },
            "reconstructed_fraction": float(reconstructed_fractions[i]),
        }
        for i in range(n_fibres)
    ]

    message = {
        "image_id": int(image_id),
        "pixel_size_mm": float(quality_params.pixel_size_mm),
        "char": char,
    }

    # Stage 5: optionally attach synthetic target values
    message = attach_synthetic_targets(message, target_params, rng)

    return message