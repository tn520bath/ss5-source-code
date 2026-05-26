"""
Empirically tests whether Module 4's claimed confidence intervals
have correct nominal coverage. For each of N_outer synthetic
batches generated with known ground truth, this component:

  1. Computes the population truth for each descriptor by drawing
     a very large reference sample from the same underlying
     distribution with no measurement noise applied.
  2. Generates a noisy 'observed' batch from the same generator.
  3. Runs Module 4 on the observed batch to obtain a 95% CI.
  4. Records whether the CI brackets the truth.

Empirical coverage = (#CIs containing truth) / N_outer. A
well-calibrated framework yields empirical coverage close to the
nominal 0.95.

Only available because Module 1 provides ground truth via known
generating parameters. Coverage validation is impossible against
real measured data.
"""

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from ss5_module1_synthetic_generator.batch_assembly import generate_image
from ss5_module1_synthetic_generator.config import (
    LengthDistributionParams,
    WidthDistributionParams,
    MeasurementNoiseParams,
    QualityMetadataParams,
    TargetGenerationParams,
)
from ss5_module1_synthetic_generator.distributions import (
    sample_lengths,
    sample_widths,
)
from ss5_module2_7_pipeline.ingestion import BatchAggregator
from ss5_module2_7_pipeline.config import IngestionParams

from .config import UncertaintyConfig
from .uncertainty_summary import summarise_uncertainty


# Descriptors validated by default — same set as the budget table.
DEFAULT_VALIDATED_DESCRIPTORS = [
    "length_D50",
    "length_D90",
    "length_mean",
    "width_D50",
    "width_mean",
    "length_D50_mass_w",
]


@dataclass(frozen=True)
class CoverageValidationConfig:
    """
    Configuration for coverage validation.

    Attributes
    ----------
    n_outer : int
        Number of synthetic batches generated. Each runs its own
        Module 4 analysis; total work is roughly n_outer * 6 *
        k_inner descriptor computations. Default 100.
    n_fibres_per_batch : int
        Fibre count per observed batch. Default 500.
    n_truth_fibres : int
        Sample size used to compute population truth. Should be
        large enough that the truth itself has negligible
        sampling uncertainty. Default 50000.
    parameter_jitter_scale : float
        Per-batch perturbation of the underlying mu and sigma
        parameters; non-zero values stress-test the framework
        against feedstock-to-feedstock variation. Default 0.05.
    inner_uncertainty_config : UncertaintyConfig
        Configuration passed to summarise_uncertainty for each
        observed batch. K_inner is set inside this.
    rng_seed : int
        Top-level seed. The same seed reproduces the entire
        validation run.
    """
    n_outer: int = 100
    n_fibres_per_batch: int = 500
    n_truth_fibres: int = 50_000
    parameter_jitter_scale: float = 0.05
    inner_uncertainty_config: UncertaintyConfig = field(
        default_factory=lambda: UncertaintyConfig(
            n_mc_iterations=300,
            rng_seed=None,  # overridden per-batch in the loop
        )
    )
    rng_seed: int = 42


@dataclass(frozen=True)
class CoverageValidationResult:
    """
    Output of run_coverage_validation.

    Attributes
    ----------
    coverage : dict[str, float]
        Empirical coverage per validated descriptor (fraction of
        N_outer batches for which the 95% CI contained the truth).
    per_batch : dict[str, list[dict]]
        One list per descriptor, length N_outer. Each entry contains
        'point', 'ci_lower', 'ci_upper', 'truth', 'covered' (bool).
    descriptors : list
        The descriptors validated.
    config : CoverageValidationConfig
    n_completed : int
        Number of outer iterations completed (== n_outer if no
        errors were encountered).
    """
    coverage: dict
    per_batch: dict
    descriptors: list
    config: CoverageValidationConfig
    n_completed: int

    def to_dataframe(self) -> pd.DataFrame:
        """
        Tidy summary: one row per descriptor.
        """
        records = []
        for d in self.descriptors:
            cov = self.coverage[d]
            n = self.n_completed
            # Wilson 95% CI on a binomial proportion
            z = 1.96
            denom = 1 + z**2 / n
            centre = (cov + z**2 / (2 * n)) / denom
            half = z * np.sqrt(cov * (1 - cov) / n + z**2 / (4 * n**2))
            half = half / denom
            records.append({
                "descriptor": d,
                "empirical_coverage": cov,
                "wilson_ci_lower": max(0.0, centre - half),
                "wilson_ci_upper": min(1.0, centre + half),
                "n_outer": n,
            })
        return pd.DataFrame(records).set_index("descriptor")


def _compute_truth(
    length_params: LengthDistributionParams,
    width_params: WidthDistributionParams,
    n_truth: int,
    rng: np.random.Generator,
) -> dict:
    """
    Compute population-truth values for the validated descriptors.

    Draws n_truth length and width samples from the same generating
    distributions used for the observed batch (no measurement noise
    applied), then computes each descriptor on the large sample.

    Returns
    -------
    dict[str, float]
        Truth values keyed by descriptor name (matching the names
        used in the rest of Module 4).
    """
    L = sample_lengths(n_truth, length_params, rng)
    w = sample_widths(n_truth, width_params, rng)

    # Mass-weighted D50 for length
    mass = L * w * w
    order = np.argsort(L)
    L_sorted = L[order]
    cumulative_mass = np.cumsum(mass[order])
    cumulative_mass /= cumulative_mass[-1]
    idx_50 = int(np.searchsorted(cumulative_mass, 0.5))
    idx_50 = min(idx_50, len(L_sorted) - 1)
    D50_mass_w = float(L_sorted[idx_50])

    return {
        "length_mean": float(np.mean(L)),
        "length_D10":  float(np.percentile(L, 10)),
        "length_D50":  float(np.percentile(L, 50)),
        "length_D90":  float(np.percentile(L, 90)),
        "length_std":  float(np.std(L, ddof=1)),
        "width_mean":  float(np.mean(w)),
        "width_D10":   float(np.percentile(w, 10)),
        "width_D50":   float(np.percentile(w, 50)),
        "width_D90":   float(np.percentile(w, 90)),
        "width_std":   float(np.std(w, ddof=1)),
        "length_D50_mass_w": D50_mass_w,
    }


def _jitter_parameters(
    base_length: LengthDistributionParams,
    base_width: WidthDistributionParams,
    jitter_scale: float,
    rng: np.random.Generator,
) -> tuple:
    """
    Apply mild Gaussian perturbation to the central distribution
    parameters, simulating per-batch feedstock variation.
    """
    if jitter_scale <= 0:
        return base_length, base_width

    new_length = replace(
        base_length,
        mu=base_length.mu + rng.normal(0.0, jitter_scale),
        sigma=max(0.05, base_length.sigma * (1.0 + rng.normal(0.0, jitter_scale))),
    )
    new_width = replace(
        base_width,
        mu=base_width.mu + rng.normal(0.0, jitter_scale),
        sigma=max(0.02, base_width.sigma * (1.0 + rng.normal(0.0, jitter_scale))),
    )
    return new_length, new_width


def run_coverage_validation(
    config: CoverageValidationConfig | None = None,
    descriptors: list | None = None,
    base_length_params: LengthDistributionParams | None = None,
    base_width_params: WidthDistributionParams | None = None,
    noise_params: MeasurementNoiseParams | None = None,
    quality_params: QualityMetadataParams | None = None,
    progress: bool = True,
) -> CoverageValidationResult:
    """
    Run the outer coverage-validation loop.

    Parameters
    ----------
    config : CoverageValidationConfig, optional
    descriptors : list, optional
        Descriptor names to validate. Defaults to
        DEFAULT_VALIDATED_DESCRIPTORS.
    base_length_params, base_width_params : optional
        Central generating parameters around which per-batch jitter
        is applied. Defaults from synthetic-generator config.
    noise_params, quality_params : optional
        Held constant across batches (these are upstream-pipeline
        properties, not feedstock properties).
    progress : bool
        Print one-line progress every 10 batches.

    Returns
    -------
    CoverageValidationResult
    """
    if config is None:
        config = CoverageValidationConfig()
    if descriptors is None:
        descriptors = list(DEFAULT_VALIDATED_DESCRIPTORS)
    if base_length_params is None:
        base_length_params = LengthDistributionParams()
    if base_width_params is None:
        base_width_params = WidthDistributionParams()
    if noise_params is None:
        noise_params = MeasurementNoiseParams()
    if quality_params is None:
        quality_params = QualityMetadataParams()

    top_rng = np.random.default_rng(config.rng_seed)

    per_batch = {d: [] for d in descriptors}

    for outer_idx in range(config.n_outer):
        # Per-batch sub-generator, deterministic given top seed
        sub_seed = int(top_rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(sub_seed)

        # Jitter parameters and compute truth from this batch's
        # underlying distribution
        length_params, width_params = _jitter_parameters(
            base_length_params,
            base_width_params,
            config.parameter_jitter_scale,
            sub_rng,
        )
        truth = _compute_truth(
            length_params,
            width_params,
            config.n_truth_fibres,
            sub_rng,
        )

        # Generate observed (noisy) batch with the same parameters
        msg = generate_image(
            n_fibres=config.n_fibres_per_batch,
            image_id=outer_idx,
            length_params=length_params,
            width_params=width_params,
            noise_params=noise_params,
            quality_params=quality_params,
            target_params=TargetGenerationParams(),
            rng=sub_rng,
        )
        aggregator = BatchAggregator(IngestionParams(images_per_batch=1))
        batch = aggregator.ingest(msg)
        if batch is None:
            continue

        # Run Module 4 with a per-batch RNG seed for reproducibility
        inner_cfg = replace(
            config.inner_uncertainty_config,
            rng_seed=sub_seed,
        )
        bu = summarise_uncertainty(batch, inner_cfg)

        # Record per-descriptor results
        for d in descriptors:
            entry = bu.entries[d]
            t = truth[d]
            covered = bool(entry.ci_lower_95 <= t <= entry.ci_upper_95)
            per_batch[d].append({
                "point":    entry.point_estimate,
                "ci_lower": entry.ci_lower_95,
                "ci_upper": entry.ci_upper_95,
                "truth":    t,
                "covered":  covered,
            })

        if progress and (outer_idx + 1) % 10 == 0:
            running_cov = {
                d: np.mean([r["covered"] for r in per_batch[d]])
                for d in descriptors
            }
            cov_str = ", ".join(
                f"{d.split('_')[-1]}={running_cov[d]:.2f}"
                for d in descriptors[:3]
            )
            print(f"  [{outer_idx + 1}/{config.n_outer}] "
                  f"running coverage: {cov_str}...")

    coverage = {
        d: float(np.mean([r["covered"] for r in per_batch[d]]))
        for d in descriptors
    }

    return CoverageValidationResult(
        coverage=coverage,
        per_batch=per_batch,
        descriptors=descriptors,
        config=config,
        n_completed=len(per_batch[descriptors[0]]),
    )