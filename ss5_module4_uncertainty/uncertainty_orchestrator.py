"""
Single user-facing entry point for downstream consumers
(Module 6 ML, Module 5 risk index, Module 8 persistence,
direct interactive use).

Drives summarise_uncertainty, the calibration
sensitivity sweep, and plotting, packaging the
results into an UncertaintyAnalysis object that contains
everything the report and the downstream modules need.

Usage:

    analysis = analyse_uncertainty(batch)
    bu = analysis.batch_uncertainty
    figs = analysis.figures             # dict of (fig, ax)
    sweep = analysis.calibration_sweep  # list of dicts
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ss5_module2_7_pipeline.ingestion import Batch
from ss5_module3_descriptors.config import DescriptorConfig

from .config import UncertaintyConfig
from .monte_carlo import run_monte_carlo, ALL_SOURCES
from .uncertainty_summary import (
    BatchUncertainty,
    summarise_uncertainty,
)
from .plotting import (
    plot_mc_histograms,
    plot_contributions,
    plot_calibration_sweep,
)


@dataclass(frozen=True)
class UncertaintyAnalysis:
    """
    Headline container for a complete Module 4 analysis on one batch.

    Attributes
    ----------
    batch_uncertainty : BatchUncertainty
        (sigmas, CIs, contributions, budget table).
    all_on_replicates : dict
        The all-on MC replicate stacks, kept for plotting and
        downstream MC propagation through Module 6.
    calibration_sweep : list of dict
        Sigma_total and source-A contribution at each
        sigma_cal_relative in config.sigma_cal_sweep.
    figures : dict
        {'mc_histograms': (fig, axes),
         'contributions': (fig, axes),
         'calibration_sweep': (fig, ax)}
    config : UncertaintyConfig
        The configuration used.
    """
    batch_uncertainty: BatchUncertainty
    all_on_replicates: dict
    calibration_sweep: list
    figures: dict
    config: UncertaintyConfig


def _run_calibration_sweep(
    batch: Batch,
    config: UncertaintyConfig,
    descriptor_config: DescriptorConfig | None,
) -> list:
    """
    Sweep sigma_cal_relative across config.sigma_cal_sweep, recording
    total sigma and source-A contribution on length_D50 at each point.

    Each sweep point requires a single all-on MC plus one source-A
    isolated MC, which is materially cheaper than a full
    summarise_uncertainty call (no full LOO).
    """
    sweep_results = []

    for sigma_cal in config.sigma_cal_sweep:
        cfg_point = replace(config, sigma_cal_relative=sigma_cal)

        # All-on at this sigma_cal
        mc_all = run_monte_carlo(
            batch,
            config=cfg_point,
            sources_active=ALL_SOURCES,
            descriptor_config=descriptor_config,
        )
        rep = mc_all.replicates["length_D50"]
        sigma_total = float(np.std(rep, ddof=1))

        # Source-A only at this sigma_cal (isolates calibration)
        mc_a = run_monte_carlo(
            batch,
            config=cfg_point,
            sources_active=frozenset({"A"}),
            descriptor_config=descriptor_config,
        )
        sigma_cal_only = float(
            np.std(mc_a.replicates["length_D50"], ddof=1)
        )

        sweep_results.append({
            "sigma_cal_relative": sigma_cal,
            "sigma_total": sigma_total,
            "sigma_calibration": sigma_cal_only,
        })

    return sweep_results


def analyse_uncertainty(
    batch: Batch,
    config: UncertaintyConfig | None = None,
    descriptor_config: DescriptorConfig | None = None,
    save_dir: str | Path | None = None,
    show_figures: bool = False,
) -> UncertaintyAnalysis:
    """
    Run the full Module 4 analysis on one batch.

    Pipeline:
      1. Run summarise_uncertainty (6 MCs: all-on + 5 LOO) to
         produce the BatchUncertainty.
      2. Run the calibration sigma sweep at the values in
         config.sigma_cal_sweep.
      3. Generate the three report figures.
      4. Optionally save figures to save_dir as PNG at 300 dpi.

    Parameters
    ----------
    batch : Batch
        Observed batch from Module 2.
    config : UncertaintyConfig, optional
    descriptor_config : DescriptorConfig, optional
    save_dir : str or Path, optional
        If supplied, figures are saved as PNG into this directory:
            mc_histograms.png
            contributions.png
            calibration_sweep.png
        The directory is created if it does not exist.
    show_figures : bool
        If True, calls plt.show() at the end (interactive use).

    Returns
    -------
    UncertaintyAnalysis
    """
    if config is None:
        config = UncertaintyConfig()

    # === 1. Full uncertainty summary ===
    bu = summarise_uncertainty(
        batch=batch,
        config=config,
        descriptor_config=descriptor_config,
    )

    # === Re-run all-on MC once more to expose replicates for plotting ===
    # (summarise_uncertainty does not retain replicates inside the
    #  BatchUncertainty; we deliberately keep that object lean.)
    mc_all = run_monte_carlo(
        batch,
        config=config,
        sources_active=ALL_SOURCES,
        descriptor_config=descriptor_config,
    )
    all_on_replicates = mc_all.replicates

    # === 2. Calibration sigma sweep ===
    sweep = _run_calibration_sweep(
        batch=batch,
        config=config,
        descriptor_config=descriptor_config,
    )

    # === 3. Figures ===
    fig_hist, ax_hist = plot_mc_histograms(bu, all_on_replicates)
    fig_contrib, ax_contrib = plot_contributions(bu)
    fig_sweep, ax_sweep = plot_calibration_sweep(sweep)

    figures = {
        "mc_histograms": (fig_hist, ax_hist),
        "contributions": (fig_contrib, ax_contrib),
        "calibration_sweep": (fig_sweep, ax_sweep),
    }

    # === 4. Save if requested ===
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig_hist.savefig(
            save_path / "mc_histograms.png",
            dpi=300, bbox_inches="tight",
        )
        fig_contrib.savefig(
            save_path / "contributions.png",
            dpi=300, bbox_inches="tight",
        )
        fig_sweep.savefig(
            save_path / "calibration_sweep.png",
            dpi=300, bbox_inches="tight",
        )

    if show_figures:
        plt.show()

    return UncertaintyAnalysis(
        batch_uncertainty=bu,
        all_on_replicates=all_on_replicates,
        calibration_sweep=sweep,
        figures=figures,
        config=config,
    )