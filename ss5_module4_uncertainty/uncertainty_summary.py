"""
Drives the Monte Carlo engine six times - once with all
sources active and once per leave-one-out (LOO) run - and reduces
the resulting replicate stacks to:

  - Standard uncertainties from the all-on MC.
  - 95% confidence intervals via percentile method.
  - Per-source contributions by LOO variance subtraction:
        sigma_X^2(d) = sigma_all^2(d) - sigma_LOO_X^2(d)
    (negative values, which arise from finite-MC noise on
    negligible sources, are clipped to zero).

Output is a BatchUncertainty dataclass containing one
UncertaintyEntry per tracked descriptor and a method that produces
the report-ready uncertainty budget table.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ss5_module2_7_pipeline.ingestion import Batch
from ss5_module3_descriptors.config import DescriptorConfig

from .config import UncertaintyConfig
from .monte_carlo import (
    run_monte_carlo,
    ALL_SOURCES,
    MonteCarloReplicates,
)


# Default report-table column order: the six headline descriptors
# the budget table should display by default.
DEFAULT_BUDGET_DESCRIPTORS = [
    "length_D50",
    "length_D90",
    "length_mean",
    "width_D50",
    "width_mean",
    "length_D50_mass_w",
]


# Human-readable labels for sources, used in the budget table index.
SOURCE_LABELS = {
    "A": "A. Calibration",
    "B": "B. Measurement noise",
    "D": "D. Reconstruction",
    "F": "F. Finite-N sampling",
    "G": "G. Modelling (MLE)",
}


@dataclass(frozen=True)
class UncertaintyEntry:
    """
    Uncertainty information for one descriptor.
    """
    point_estimate: float
    standard_uncertainty: float
    ci_lower_95: float
    ci_upper_95: float
    contributions: dict


@dataclass(frozen=True)
class BatchUncertainty:
    """
    Headline output of Module 4 for one observed batch.

    Attributes
    ----------
    batch_id : int
    n_fibres : int
    n_mc_iterations : int
    entries : dict[str, UncertaintyEntry]
        One entry per tracked descriptor.
    sources_active_main : frozenset
        The source set used for the all-on run (= ALL_SOURCES by
        default).
    config : UncertaintyConfig
        The configuration used to produce this object.
    """
    batch_id: int
    n_fibres: int
    n_mc_iterations: int
    entries: dict
    sources_active_main: frozenset
    config: UncertaintyConfig

    def to_budget_table(
        self,
        descriptors: list | None = None,
    ) -> pd.DataFrame:
        """
        Produce the report-ready uncertainty budget table.

        Rows: uncertainty sources (A, B, D, F, G, Combined,
        plus E and H as documentary notes).
        Columns: descriptors (default DEFAULT_BUDGET_DESCRIPTORS).
        Cells: absolute standard uncertainty contributions in the
        descriptor's natural units (mm for lengths/widths,
        dimensionless for fines fraction). The 'Combined' row is
        the total standard uncertainty from the all-on MC.

        Parameters
        ----------
        descriptors : list, optional
            Subset of tracked descriptor names to include. Defaults
            to DEFAULT_BUDGET_DESCRIPTORS.

        Returns
        -------
        pd.DataFrame
        """
        if descriptors is None:
            descriptors = DEFAULT_BUDGET_DESCRIPTORS

        rows = []
        index = []
        for source_id, label in SOURCE_LABELS.items():
            row = []
            for d in descriptors:
                row.append(
                    self.entries[d].contributions.get(source_id, np.nan)
                )
            rows.append(row)
            index.append(label)

        # Combined row (= total standard uncertainty)
        rows.append(
            [self.entries[d].standard_uncertainty for d in descriptors]
        )
        index.append("Combined (quadrature)")

        # E and H documentary entries
        rows.append([np.nan] * len(descriptors))
        index.append("E. Overlap (handled by SS4 QC; not in budget)")
        rows.append([np.nan] * len(descriptors))
        index.append("H. Feedstock variability (target of measurement; "
                     "not error)")

        return pd.DataFrame(rows, columns=descriptors, index=index)

    def to_summary_dataframe(self) -> pd.DataFrame:
        """
        Produce a tidy long-form summary, one row per descriptor.

        Columns: point, sigma, ci_lower_95, ci_upper_95,
        sigma_A, sigma_B, sigma_D, sigma_F, sigma_G.
        """
        records = []
        for name, entry in self.entries.items():
            rec = {
                "descriptor": name,
                "point": entry.point_estimate,
                "sigma": entry.standard_uncertainty,
                "ci_lower_95": entry.ci_lower_95,
                "ci_upper_95": entry.ci_upper_95,
            }
            for src, value in entry.contributions.items():
                rec[f"sigma_{src}"] = value
            records.append(rec)
        return pd.DataFrame(records).set_index("descriptor")


def _percentile_ci(
    replicates: np.ndarray, confidence_level: float
) -> tuple[float, float]:
    """
    Two-sided percentile confidence interval.
    """
    alpha = 1.0 - confidence_level
    lower = np.percentile(replicates, 100.0 * alpha / 2.0)
    upper = np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _attribute_contribution(
    var_all: float,
    var_loo: float,
) -> float:
    """
    Per-source contribution: sqrt(max(var_all - var_loo, 0)).
    """
    diff = max(var_all - var_loo, 0.0)
    return float(np.sqrt(diff))


def summarise_uncertainty(
    batch: Batch,
    config: UncertaintyConfig | None = None,
    descriptor_config: DescriptorConfig | None = None,
) -> BatchUncertainty:
    """
    Run the all-on MC plus the five LOO MCs, and reduce the result
    to a BatchUncertainty.

    Parameters
    ----------
    batch : Batch
        Observed batch from Module 2.
    config : UncertaintyConfig, optional
    descriptor_config : DescriptorConfig, optional

    Returns
    -------
    BatchUncertainty
    """
    if config is None:
        config = UncertaintyConfig()

    # 1. All-on MC
    mc_all = run_monte_carlo(
        batch,
        config=config,
        sources_active=ALL_SOURCES,
        descriptor_config=descriptor_config,
    )

    # 2. Leave-one-out MCs
    mc_loo: dict[str, MonteCarloReplicates] = {}
    for source_id in ALL_SOURCES:
        loo_set = ALL_SOURCES - {source_id}
        mc_loo[source_id] = run_monte_carlo(
            batch,
            config=config,
            sources_active=loo_set,
            descriptor_config=descriptor_config,
        )

    # 3. Reduce to per-descriptor entries
    entries: dict = {}
    for name, replicates_all in mc_all.replicates.items():
        sigma_all = float(np.std(replicates_all, ddof=1))
        var_all = sigma_all ** 2
        ci_low, ci_high = _percentile_ci(
            replicates_all, config.confidence_level
        )

        contributions: dict = {}
        for source_id, mc_minus in mc_loo.items():
            sigma_loo = float(
                np.std(mc_minus.replicates[name], ddof=1)
            )
            var_loo = sigma_loo ** 2
            contributions[source_id] = _attribute_contribution(
                var_all, var_loo
            )

        entries[name] = UncertaintyEntry(
            point_estimate=float(mc_all.point_estimates[name]),
            standard_uncertainty=sigma_all,
            ci_lower_95=ci_low,
            ci_upper_95=ci_high,
            contributions=contributions,
        )

    return BatchUncertainty(
        batch_id=batch.batch_id,
        n_fibres=len(batch.fibre_records),
        n_mc_iterations=config.n_mc_iterations,
        entries=entries,
        sources_active_main=frozenset(ALL_SOURCES),
        config=config,
    )