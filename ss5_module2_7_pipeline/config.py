"""
Configuration parameters for the SS5 runtime pipeline.

Separate from ss5_module1_synthetic_generator.config: the synthetic
generator and the runtime pipeline are independent concerns and
should not share parameter dataclasses.
"""

from dataclasses import dataclass


@dataclass
class IngestionParams:
    """
    Parameters for Module 2 (Data Ingestion & Validation).

    images_per_batch:
        How many incoming images make up one batch for downstream
        statistical analysis. Default 1 (one image = one batch).
        Larger values aggregate multiple images into a single
        statistical sample, useful when individual images contain
        too few fibres to support reliable distribution fitting.

    high_reconstruction_threshold:
        Fibres with reconstructed_fraction strictly greater than
        this threshold are flagged with a 'high_reconstruction'
        boolean column for downstream weighting (most directly,
        Module 4's uncertainty quantification). Default 0.5
        flags fibres where more than half the mask was
        reconstructed.
    """
    images_per_batch: int = 1
    high_reconstruction_threshold: float = 0.5