"""
Three top-level functions, each returning (fig, axes) so the caller
can save, display, or composite the
figures.

  - plot_mc_histograms: 2x2 panel of MC histograms for the four
    headline descriptors (length D50, length D90, width D50,
    width mean), with point estimate and 95% CI overlays.

  - plot_contributions: small-multiples horizontal bar chart of
    per-source standard-uncertainty contributions across three
    descriptors, with fixed source ordering and percentage-of-
    variance labels.

  - plot_calibration_sweep: line plot of total sigma on length_D50
    versus sigma_cal_relative across the configured sweep, with
    the calibration-only contribution overlaid.
"""

import numpy as np
import matplotlib.pyplot as plt

from .uncertainty_summary import (
    BatchUncertainty,
    SOURCE_LABELS,
)


# Source order for bar charts (fixed across panels for cross-panel
# comparison).
_SOURCE_ORDER = ["A", "B", "D", "F", "G"]


# Default panel descriptor sets
_HISTOGRAM_DESCRIPTORS = [
    "length_D50",
    "length_D90",
    "width_D50",
    "width_mean",
]
_CONTRIBUTION_DESCRIPTORS = [
    "length_D50",
    "width_D50",
    "length_D50_mass_w",
]


def _descriptor_label(name: str) -> str:
    """
    Human-readable descriptor name for axis labels and panel titles.
    """
    mapping = {
        "length_mean":          "Length, mean (mm)",
        "length_D10":           "Length, $D_{10}$ (mm)",
        "length_D50":           "Length, $D_{50}$ (mm)",
        "length_D90":           "Length, $D_{90}$ (mm)",
        "length_std":           "Length, SD (mm)",
        "width_mean":           "Width, mean (mm)",
        "width_D10":            "Width, $D_{10}$ (mm)",
        "width_D50":            "Width, $D_{50}$ (mm)",
        "width_D90":            "Width, $D_{90}$ (mm)",
        "width_std":            "Width, SD (mm)",
        "length_D50_mass_w":    "Length, $D_{50}$ mass-w. (mm)",
        "fines_fraction_count": "Fines fraction (count)",
    }
    return mapping.get(name, name)


def plot_mc_histograms(
    batch_uncertainty: BatchUncertainty,
    replicates: dict,
    descriptors: list | None = None,
) -> tuple:
    """
    2x2 panel of MC histograms for headline descriptors.

    Parameters
    ----------
    batch_uncertainty : BatchUncertainty
        Object from Component 3.
    replicates : dict[str, np.ndarray]
        The all-on MC replicate stacks (one array per descriptor).
        Sourced from Component 4 orchestrator's MC pass.
    descriptors : list, optional
        Length-4 descriptor name list. Defaults to _HISTOGRAM_DESCRIPTORS.

    Returns
    -------
    (fig, axes)
    """
    if descriptors is None:
        descriptors = _HISTOGRAM_DESCRIPTORS
    if len(descriptors) != 4:
        raise ValueError(
            f"plot_mc_histograms expects 4 descriptors; got "
            f"{len(descriptors)}."
        )

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes_flat = axes.flatten()

    for ax, name in zip(axes_flat, descriptors):
        entry = batch_uncertainty.entries[name]
        rep = replicates[name]

        ax.hist(
            rep,
            bins=40,
            density=True,
            color="#4F81BD",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
        )
        ax.axvline(
            entry.point_estimate,
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Point estimate",
        )
        ax.axvline(
            entry.ci_lower_95,
            color="firebrick",
            linestyle="--",
            linewidth=1.2,
            label="95% CI",
        )
        ax.axvline(
            entry.ci_upper_95,
            color="firebrick",
            linestyle="--",
            linewidth=1.2,
        )

        # Annotation box: sigma and relative sigma
        rel_sigma = (
            entry.standard_uncertainty / entry.point_estimate
            if entry.point_estimate > 0 else float("nan")
        )
        ax.text(
            0.97, 0.96,
            f"$\\sigma$ = {entry.standard_uncertainty:.4f}\n"
            f"$\\sigma$/x = {rel_sigma * 100:.2f}%",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="lightgray",
            ),
        )

        ax.set_xlabel(_descriptor_label(name))
        ax.set_ylabel("Density")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    return fig, axes


def plot_contributions(
    batch_uncertainty: BatchUncertainty,
    descriptors: list | None = None,
) -> tuple:
    """
    Small-multiples horizontal bar chart of per-source
    standard-uncertainty contributions.

    Parameters
    ----------
    batch_uncertainty : BatchUncertainty
    descriptors : list, optional
        Descriptor names to put in panels. Defaults to
        _CONTRIBUTION_DESCRIPTORS.

    Returns
    -------
    (fig, axes)
    """
    if descriptors is None:
        descriptors = _CONTRIBUTION_DESCRIPTORS

    n_panels = len(descriptors)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(3.6 * n_panels, 4.2),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    y_positions = np.arange(len(_SOURCE_ORDER))
    y_labels = [SOURCE_LABELS[s] for s in _SOURCE_ORDER]

    for ax, name in zip(axes, descriptors):
        entry = batch_uncertainty.entries[name]
        sigma_total = entry.standard_uncertainty
        var_total = sigma_total ** 2

        sigmas = [entry.contributions[s] for s in _SOURCE_ORDER]
        # Variance share for label (per-source variance / total variance).
        if var_total > 0:
            var_share = [(s ** 2) / var_total * 100 for s in sigmas]
        else:
            var_share = [0.0] * len(sigmas)

        bars = ax.barh(
            y_positions,
            sigmas,
            color="#4F81BD",
            edgecolor="white",
            linewidth=0.4,
        )
        # Variance-share labels
        x_max = max(sigmas) if max(sigmas) > 0 else 1.0
        for bar, share in zip(bars, var_share):
            ax.text(
                bar.get_width() + 0.03 * x_max,
                bar.get_y() + bar.get_height() / 2.0,
                f"{share:.1f}%",
                ha="left", va="center",
                fontsize=8,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"$\\sigma$ contribution ({_descriptor_label(name)})")
        ax.set_title(_descriptor_label(name), fontsize=10)
        ax.set_xlim(left=0, right=x_max * 1.25)

    fig.tight_layout()
    return fig, axes


def plot_calibration_sweep(
    sweep_results: list,
) -> tuple:
    """
    Line plot of total sigma on length_D50 vs sigma_cal_relative.

    Parameters
    ----------
    sweep_results : list of dict
        One dict per sweep point with keys:
            'sigma_cal_relative' : float
            'sigma_total'        : float (length_D50 total sigma)
            'sigma_calibration'  : float (source-A contribution)

    Returns
    -------
    (fig, ax)
    """
    if len(sweep_results) < 2:
        raise ValueError(
            f"plot_calibration_sweep needs >= 2 sweep points; got "
            f"{len(sweep_results)}."
        )

    x = np.array([r["sigma_cal_relative"] for r in sweep_results]) * 100
    y_total = np.array([r["sigma_total"] for r in sweep_results])
    y_cal = np.array([r["sigma_calibration"] for r in sweep_results])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(
        x, y_total,
        marker="o", color="#4F81BD",
        linewidth=1.5, markersize=7,
        label=r"Total $\sigma$ on $D_{50}$",
    )
    ax.plot(
        x, y_cal,
        marker="s", color="firebrick",
        linewidth=1.2, markersize=6, linestyle="--",
        label=r"Calibration-only contribution",
    )

    ax.set_xlabel(r"$\sigma_{\mathrm{cal}}/k$ (%)")
    ax.set_ylabel(r"$\sigma$ on length $D_{50}$ (mm)")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig, ax


def plot_coverage_validation(
    coverage_result,
    primary_descriptor: str = "length_D50",
) -> tuple:
    """
    Two-panel coverage-validation figure.

    Left panel: empirical coverage per descriptor with the nominal
    95% line.
    Right panel: per-batch CIs and truths for one descriptor,
    colour-coded by whether the CI covered the truth.

    Parameters
    ----------
    coverage_result : CoverageValidationResult
    primary_descriptor : str
        Descriptor used for the right-panel CI scatter. Default
        'length_D50'.

    Returns
    -------
    (fig, axes)
    """
    descriptors = coverage_result.descriptors

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [1, 1.3]},
    )

    # === Left: bar chart of coverage per descriptor ===
    ax = axes[0]
    coverages = [coverage_result.coverage[d] for d in descriptors]
    y = np.arange(len(descriptors))
    bars = ax.barh(
        y, coverages,
        color="#4F81BD",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(
        0.95, color="firebrick", linestyle="--", linewidth=1.2,
        label="Nominal (0.95)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([_descriptor_label(d) for d in descriptors])
    ax.invert_yaxis()
    ax.set_xlabel("Empirical coverage (fraction of CIs containing truth)")
    ax.set_xlim(0.7, 1.02)
    ax.legend(loc="lower right", fontsize=9)

    # Per-bar value labels
    for bar, cov in zip(bars, coverages):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2.0,
            f"{cov:.2f}",
            va="center", fontsize=8,
        )

    # === Right: per-batch CI vs truth for primary descriptor ===
    ax2 = axes[1]
    records = coverage_result.per_batch[primary_descriptor]
    n_outer = len(records)
    x = np.arange(n_outer)
    for i, rec in enumerate(records):
        colour = "#4F81BD" if rec["covered"] else "firebrick"
        ax2.vlines(
            i, rec["ci_lower"], rec["ci_upper"],
            color=colour, alpha=0.6, linewidth=0.9,
        )
        ax2.plot(
            i, rec["truth"],
            marker="x", color="black", markersize=4, alpha=0.8,
        )

    ax2.set_xlabel("Synthetic batch index")
    ax2.set_ylabel(_descriptor_label(primary_descriptor))
    ax2.set_title(
        f"Per-batch 95% CIs and population truth "
        f"({_descriptor_label(primary_descriptor)})",
        fontsize=10,
    )

    # Legend stub via proxy artists
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#4F81BD", lw=2, label="CI covers truth"),
        Line2D([0], [0], color="firebrick", lw=2, label="CI misses truth"),
        Line2D([0], [0], color="black", marker="x", lw=0,
               markersize=5, label="Truth"),
    ]
    ax2.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig, axes