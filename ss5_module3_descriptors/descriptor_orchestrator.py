"""
Single entry point for downstream consumers (Module 4 uncertainty,
Module 6 ML, Module 5 risk index, Module 8 persistence). Takes a
Batch object emitted by Module 2 and returns a BatchDescriptors
dataclass containing all empirical descriptors, mass-weighted
descriptors, fines metrics, distribution fits, and goodness-of-fit
diagnostics for the batch.

The orchestrator is the only public interface in Module 3 that
downstream code should call. Internal components
can be called directly during testing or by Module 4's Monte Carlo
machinery, but the production data flow is:

    Module 2 -> compute_descriptors(batch) -> Module 4/5/6

Provides:

    compute_descriptors(batch, config)
        Returns a BatchDescriptors object.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import DescriptorConfig
from .empirical_descriptors import (
    compute_empirical_descriptors,
    compute_reconstruction_summary,
)
from .mass_weighted import compute_mass_weighted_descriptors
from .fines import compute_fines_fraction
from .distribution_fitting import fit_distribution, DistributionFit
from .goodness_of_fit import compute_gof, compare_aic


@dataclass(frozen=True)
class BatchDescriptors:
    """
    Statistical summary of a single fibre batch.

    Produced by compute_descriptors and consumed by
    Modules 4 (uncertainty), 5 (risk
    index), 6 (predictive modelling), and 8 (persistence).

    All quantities are point estimates. Uncertainty quantification
    is the responsibility of Module 4, which wraps the descriptors
    here with confidence intervals via Monte Carlo and bootstrap
    resampling.

    Attributes
    ----------
    batch_id : int
    n_fibres : int
    source_image_ids : list
    pixel_size_mm : float
    timestamp : Any  (datetime, kept Any to avoid coupling)

    descriptors_length, descriptors_width, descriptors_aspect_ratio : dict
        Empirical descriptors (count-weighted).
    descriptors_length_mass_weighted : dict
        Mass-weighted descriptors for length.
    fines : dict
        Count-based and mass-based fines fractions.
    reconstruction : dict
        Reconstruction-quality summary for the batch.

    fit_length_lognormal, fit_length_weibull, fit_width_lognormal : DistributionFit
        Frozen distribution fits.

    gof_length_lognormal, gof_length_weibull, gof_width_lognormal : dict
        Goodness-of-fit diagnostics for each fit.
    aic_comparison_length : dict
        Lognormal-vs-Weibull AIC comparison for the length fit.
    """
    # Traceability
    batch_id: int
    n_fibres: int
    source_image_ids: list
    pixel_size_mm: float
    timestamp: Any

    # Empirical descriptors
    descriptors_length: dict
    descriptors_width: dict
    descriptors_aspect_ratio: dict

    # Mass-weighted descriptors
    descriptors_length_mass_weighted: dict

    # Fines and reconstruction summaries
    fines: dict
    reconstruction: dict

    # Distribution fits
    fit_length_lognormal: DistributionFit
    fit_length_weibull: DistributionFit
    fit_width_lognormal: DistributionFit

    # Goodness-of-fit
    gof_length_lognormal: dict
    gof_length_weibull: dict
    gof_width_lognormal: dict
    aic_comparison_length: dict

    def to_dataframe(self) -> pd.DataFrame:
        """
        Assemble the descriptor table as a pandas DataFrame for
        report and display.

        Rows are descriptor quantities (n, mean, median, std, cv,
        skew, D10, D50, D90); columns are quantity types (length,
        width, aspect_ratio, length_mass_weighted).

        Returns
        -------
        pd.DataFrame
            Descriptor table.
        """
        # Standard descriptor keys produced by Component 1
        keys = ["n", "mean", "median", "std", "cv", "skew",
                "D10", "D50", "D90"]
        # Mass-weighted keys (subset, plus total_mass)
        mw_keys = ["n", "mean_mass_w", None, "std_mass_w", "cv_mass_w",
                   None, "D10_mass_w", "D50_mass_w", "D90_mass_w"]

        # Build columns
        cols = {
            "length": [self.descriptors_length.get(k) for k in keys],
            "width": [self.descriptors_width.get(k) for k in keys],
            "aspect_ratio": [self.descriptors_aspect_ratio.get(k)
                             for k in keys],
            "length_mass_weighted": [
                self.descriptors_length_mass_weighted.get(k)
                if k is not None else None
                for k in mw_keys
            ],
        }

        return pd.DataFrame(cols, index=keys)


def compute_descriptors(
    batch: Any,
    config: DescriptorConfig | None = None,
) -> BatchDescriptors:
    """
    Compute the full BatchDescriptors for a Module 2 Batch.

    Parameters
    ----------
    batch : Batch
        Batch object from Module 2, with attributes:
          - fibre_records : pandas DataFrame with columns 'length',
                            'width', 'reconstructed_fraction',
                            'high_reconstruction'.
          - batch_id : int
          - source_image_ids : list of int
          - pixel_size_mm : float
          - creation_timestamp : str (ISO timestamp)
    config : DescriptorConfig, optional
        Configuration object. If None, defaults are used.

    Returns
    -------
    BatchDescriptors
        Full statistical summary of the batch.

    Raises
    ------
    ValueError
        If the batch DataFrame is empty.
    """
    if config is None:
        config = DescriptorConfig()

    df = batch.fibre_records
    n = len(df)

    if n == 0:
        raise ValueError(
            "BatchDescriptors cannot be computed on an empty batch."
        )

    # Extract arrays once
    length = df["length"].to_numpy()
    width = df["width"].to_numpy()
    aspect_ratio = length / width
    reconstructed_fraction = df["reconstructed_fraction"].to_numpy()
    is_high_reconstruction = df["high_reconstruction"].to_numpy()

    # --- Components 1, 2, 3 ---
    desc_length = compute_empirical_descriptors(length, config)
    desc_width = compute_empirical_descriptors(width, config)
    desc_aspect = compute_empirical_descriptors(aspect_ratio, config)
    desc_length_mw = compute_mass_weighted_descriptors(length, width)
    fines = compute_fines_fraction(
        length, width, threshold_mm=config.fines_threshold_mm
    )
    reconstruction = compute_reconstruction_summary(
        reconstructed_fraction, is_high_reconstruction
    )

    # --- Component 4: distribution fits ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_length_lognormal = fit_distribution(
            length, "lognormal", config
        )
        fit_length_weibull = fit_distribution(
            length, "weibull", config
        )
        fit_width_lognormal = fit_distribution(
            width, "lognormal", config
        )

    if n < config.small_n_warning_threshold:
        warnings.warn(
            f"Batch {batch.batch_id} has N={n} fibres, below threshold "
            f"of {config.small_n_warning_threshold}. Distribution fits "
            "and descriptor estimates may be unreliable.",
            stacklevel=2,
        )

    # --- Component 5: goodness-of-fit ---
    gof_length_lognormal = compute_gof(fit_length_lognormal, length)
    gof_length_weibull = compute_gof(fit_length_weibull, length)
    gof_width_lognormal = compute_gof(fit_width_lognormal, width)
    aic_comparison_length = compare_aic(
        gof_length_lognormal, gof_length_weibull
    )

    return BatchDescriptors(
        batch_id=batch.batch_id,
        n_fibres=n,
        source_image_ids=list(batch.source_image_ids),
        pixel_size_mm=float(batch.pixel_size_mm),
        timestamp=batch.creation_timestamp,
        descriptors_length=desc_length,
        descriptors_width=desc_width,
        descriptors_aspect_ratio=desc_aspect,
        descriptors_length_mass_weighted=desc_length_mw,
        fines=fines,
        reconstruction=reconstruction,
        fit_length_lognormal=fit_length_lognormal,
        fit_length_weibull=fit_length_weibull,
        fit_width_lognormal=fit_width_lognormal,
        gof_length_lognormal=gof_length_lognormal,
        gof_length_weibull=gof_length_weibull,
        gof_width_lognormal=gof_width_lognormal,
        aic_comparison_length=aic_comparison_length,
    )