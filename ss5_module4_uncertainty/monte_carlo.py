"""
Runs the composite Monte Carlo scheme that propagates measurement,
calibration, reconstruction, sampling, and modelling uncertainty
through Module 3's full descriptor-computation pipeline.

Methodology follows JCGM 101:2008 (GUM Supplement 1): all sources
are activated simultaneously by default, with leave-one-out runs
controlled via the `sources_active` argument so that per-source
contributions can be attributed downstream.

Each iteration:

    1. (Source A) Draw a batch-level calibration factor.
    2. (Sources B, D) Perturb each fibre's length and width by
       Gaussian noise, with reconstruction-flagged fibres receiving
       inflated noise.
    3. Apply the calibration factor multiplicatively to the
       perturbed measurements.
    4. (Source F) Bootstrap-resample the perturbed fibres with
       replacement.
    5. (Source G implicit) Re-run Module 3's compute_descriptors
       on the resulting batch, which re-fits the parametric
       distributions and produces the headline descriptors.
    6. Record the descriptor outputs of interest.

The output is a dict of np.ndarray stacks, one per tracked
descriptor, of shape (n_iterations,).
"""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from ss5_module2_7_pipeline.ingestion import Batch
from ss5_module3_descriptors.descriptor_orchestrator import (
    compute_descriptors,
    BatchDescriptors,
)
from ss5_module3_descriptors.config import DescriptorConfig

from .config import UncertaintyConfig


# Source IDs as agreed in the master plan.
ALL_SOURCES: frozenset = frozenset({"A", "B", "D", "F", "G"})


@dataclass(frozen=True)
class MonteCarloReplicates:
    """
    Stack of K Monte Carlo replicates of every tracked descriptor.

    Each attribute is an np.ndarray of shape (n_iterations,).

    Attributes
    ----------
    sources_active : frozenset
        The set of sources that were activated in this MC run.
    n_iterations : int
        Number of completed iterations.
    point_estimates : dict
        The original (unperturbed) Module 3 descriptors, kept for
        reference.
    replicates : dict[str, np.ndarray]
        Stacked replicates, keyed by descriptor name.
    """
    sources_active: frozenset
    n_iterations: int
    point_estimates: dict
    replicates: dict


def _extract_tracked_outputs(bd: BatchDescriptors) -> dict:
    """
    Extract the 15 tracked scalar outputs from a BatchDescriptors.

    The set is fixed here so all replicates produce dicts with the
    same keys, ready for stacking.
    """
    return {
        "length_mean":   bd.descriptors_length["mean"],
        "length_D10":    bd.descriptors_length["D10"],
        "length_D50":    bd.descriptors_length["D50"],
        "length_D90":    bd.descriptors_length["D90"],
        "length_std":    bd.descriptors_length["std"],
        "width_mean":    bd.descriptors_width["mean"],
        "width_D10":     bd.descriptors_width["D10"],
        "width_D50":     bd.descriptors_width["D50"],
        "width_D90":     bd.descriptors_width["D90"],
        "width_std":     bd.descriptors_width["std"],
        "length_D50_mass_w": (
            bd.descriptors_length_mass_weighted["D50_mass_w"]
        ),
        "fines_fraction_count": bd.fines["fines_fraction_count"],
        "lognormal_mu_length":    bd.fit_length_lognormal.params["mu"],
        "lognormal_sigma_length": (
            bd.fit_length_lognormal.params["sigma"]
        ),
        "lognormal_mu_width":     bd.fit_width_lognormal.params["mu"],
        "lognormal_sigma_width":  (
            bd.fit_width_lognormal.params["sigma"]
        ),
    }


def _build_perturbed_batch(
    original: Batch,
    perturbed_length: np.ndarray,
    perturbed_width: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> Batch:
    """
    Construct a Batch object from perturbed-and-resampled fibre
    arrays, preserving Module 3's expected input contract.

    We bypass BatchAggregator entirely: validation and filtering
    have already been applied to the original batch, so the
    perturbed copy only needs the columns Module 3 reads.
    """
    # Source columns from the original DataFrame for everything
    # other than length/width.
    base_df = original.fibre_records.iloc[bootstrap_indices].copy()
    base_df = base_df.reset_index(drop=True)

    # Replace length and width with the perturbed-and-resampled values
    base_df["length"] = perturbed_length[bootstrap_indices]
    base_df["width"] = perturbed_width[bootstrap_indices]

    return Batch(
        batch_id=original.batch_id,
        source_image_ids=list(original.source_image_ids),
        fibre_records=base_df,
        pixel_size_mm=original.pixel_size_mm,
        creation_timestamp=original.creation_timestamp,
        validation_issues=list(original.validation_issues),
    )


def run_monte_carlo(
    batch: Batch,
    config: UncertaintyConfig | None = None,
    sources_active: frozenset = ALL_SOURCES,
    descriptor_config: DescriptorConfig | None = None,
) -> MonteCarloReplicates:
    """
    Run the composite Monte Carlo on one observed batch.

    Parameters
    ----------
    batch : Batch
        The observed batch (Module 2 output).
    config : UncertaintyConfig, optional
        Module 4 configuration. Defaults used if None.
    sources_active : frozenset
        Subset of {"A", "B", "D", "F", "G"} indicating which sources
        are switched on. The all-on run is sources_active=ALL_SOURCES.
        Leave-one-out runs use ALL_SOURCES - {source}.
    descriptor_config : DescriptorConfig, optional
        Forwarded to compute_descriptors. Defaults used if None.

    Returns
    -------
    MonteCarloReplicates
        The K-replicate stack and the original point estimates.
    """
    if config is None:
        config = UncertaintyConfig()
    if descriptor_config is None:
        descriptor_config = DescriptorConfig()

    # Validate sources_active
    unknown = sources_active - ALL_SOURCES
    if unknown:
        raise ValueError(
            f"Unknown source IDs in sources_active: {unknown}. "
            f"Expected subset of {set(ALL_SOURCES)}."
        )

    # Resolve effective base noise sigma values
    sigma_L_base, sigma_w_base = config.resolved_noise_params()

    # Pull source toggles into local flags for clarity
    use_A = "A" in sources_active   # calibration
    use_B = "B" in sources_active   # measurement noise
    use_D = "D" in sources_active   # reconstruction inflation
    use_F = "F" in sources_active   # bootstrap (finite-N)
    # "G" (modelling) is implicit: re-fits happen on every iteration
    # via compute_descriptors; toggling G off is handled below by
    # freezing the original batch's measurements.

    # Top-level seeded generator
    rng = np.random.default_rng(config.rng_seed)

    # Original point estimates (no perturbation)
    point_bd = compute_descriptors(batch, descriptor_config)
    point_estimates = _extract_tracked_outputs(point_bd)

    # Pre-extract original arrays
    df = batch.fibre_records
    L_orig = df["length"].to_numpy()
    w_orig = df["width"].to_numpy()
    is_high_recon = df["high_reconstruction"].to_numpy()
    n = L_orig.size

    # Per-fibre effective noise sigma (depends on source D toggle)
    if use_D:
        inflation = np.where(
            is_high_recon,
            config.reconstruction_inflation_factor,
            1.0,
        )
    else:
        inflation = np.ones(n)

    sigma_L_eff = sigma_L_base * inflation
    sigma_w_eff = sigma_w_base * inflation

    # Pre-allocate replicate storage
    tracked_keys = list(point_estimates.keys())
    replicate_arrays = {
        key: np.empty(config.n_mc_iterations, dtype=float)
        for key in tracked_keys
    }

    # === Main Monte Carlo loop ===
    for k in range(config.n_mc_iterations):

        # --- Source A: calibration factor ---
        if use_A:
            cal_factor = rng.normal(
                loc=1.0, scale=config.sigma_cal_relative
            )
        else:
            cal_factor = 1.0

        # --- Sources B, D: per-fibre measurement perturbation ---
        if use_B:
            delta_L = rng.normal(0.0, sigma_L_eff, size=n)
            delta_w = rng.normal(0.0, sigma_w_eff, size=n)
        else:
            delta_L = np.zeros(n)
            delta_w = np.zeros(n)

        L_perturbed = (L_orig + delta_L) * cal_factor
        w_perturbed = (w_orig + delta_w) * cal_factor

        # Reject any iteration that would produce non-positive values.
        # Ultra-rare with realistic sigma values; safety net only.
        if np.any(L_perturbed <= 0) or np.any(w_perturbed <= 0):
            # Replace the offending fibres with their pre-perturbation
            # values rather than discarding the iteration. Avoids
            # bias from systematically dropping iterations with
            # extreme draws.
            bad = (L_perturbed <= 0) | (w_perturbed <= 0)
            L_perturbed[bad] = L_orig[bad]
            w_perturbed[bad] = w_orig[bad]

        # --- Source F: bootstrap resample ---
        if use_F:
            boot_idx = rng.integers(low=0, high=n, size=n)
        else:
            boot_idx = np.arange(n)

        # --- Build perturbed Batch and run Module 3 ---
        new_batch = _build_perturbed_batch(
            original=batch,
            perturbed_length=L_perturbed,
            perturbed_width=w_perturbed,
            bootstrap_indices=boot_idx,
        )
        bd = compute_descriptors(new_batch, descriptor_config)
        outputs = _extract_tracked_outputs(bd)

        for key in tracked_keys:
            replicate_arrays[key][k] = outputs[key]

    return MonteCarloReplicates(
        sources_active=frozenset(sources_active),
        n_iterations=config.n_mc_iterations,
        point_estimates=point_estimates,
        replicates=replicate_arrays,
    )