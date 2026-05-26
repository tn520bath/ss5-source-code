"""
Validation script for the
top-level orchestrator + plotting + calibration sweep.

Confirms five behaviours:

  1. analyse_uncertainty returns a populated UncertaintyAnalysis
     for a synthetic batch.
  2. Calibration sweep is monotonically increasing in
     sigma_cal_relative.
  3. The three figures are generated and have the expected
     structure (axes count, non-empty data).
  4. With save_dir set, three PNG files are written to disk.
  5. The budget table renders to console without error and the
     headline length_D50 entries match the BatchUncertainty.
"""

import shutil
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for validation
import matplotlib.pyplot as plt
import numpy as np

from ss5_module1_synthetic_generator.batch_assembly import generate_image
from ss5_module2_7_pipeline.ingestion import BatchAggregator
from ss5_module2_7_pipeline.config import IngestionParams

from ss5_module4_uncertainty.config import UncertaintyConfig
from ss5_module4_uncertainty.uncertainty_orchestrator import (
    analyse_uncertainty,
    UncertaintyAnalysis,
)


def _make_batch(n_fibres: int, seed: int):
    rng = np.random.default_rng(seed)
    aggregator = BatchAggregator(IngestionParams(images_per_batch=1))
    msg = generate_image(n_fibres=n_fibres, image_id=0, rng=rng)
    batch = aggregator.ingest(msg)
    assert batch is not None
    return batch


def main():
    print("--- Validation: Component 4 (orchestrator + plots) ---\n")

    batch = _make_batch(n_fibres=400, seed=4)
    cfg = UncertaintyConfig(
        n_mc_iterations=300,
        sigma_cal_sweep=(0.005, 0.01, 0.02),
        rng_seed=4,
    )

    print("Running analyse_uncertainty (1 full summary + 1 all-on "
          "+ 3-point sweep)...")
    t0 = time.time()
    analysis = analyse_uncertainty(batch, cfg)
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s\n")

    # === Test 1: structure ===
    print("Test 1: analysis has all expected fields")
    assert isinstance(analysis, UncertaintyAnalysis)
    assert analysis.batch_uncertainty is not None
    assert len(analysis.batch_uncertainty.entries) == 16
    assert len(analysis.all_on_replicates) == 16
    assert all(
        arr.shape == (cfg.n_mc_iterations,)
        for arr in analysis.all_on_replicates.values()
    )
    assert len(analysis.calibration_sweep) == len(cfg.sigma_cal_sweep)
    assert set(analysis.figures.keys()) == {
        "mc_histograms", "contributions", "calibration_sweep"
    }
    print("  Batch uncertainty, replicates, sweep, figures: present.")
    print("  -> Passed.\n")

    # === Test 2: sweep monotonic ===
    print("Test 2: total sigma monotonic in sigma_cal_relative")
    totals = [r["sigma_total"] for r in analysis.calibration_sweep]
    cal_only = [r["sigma_calibration"]
                for r in analysis.calibration_sweep]
    print(f"  sigma_cal_relative: "
          f"{[r['sigma_cal_relative'] for r in analysis.calibration_sweep]}")
    print(f"  sigma_total:        {[round(t, 5) for t in totals]}")
    print(f"  sigma_cal_only:     {[round(c, 5) for c in cal_only]}")
    assert all(totals[i] <= totals[i+1] + 1e-4
               for i in range(len(totals) - 1)), (
        "Total sigma should be non-decreasing in sigma_cal_relative"
    )
    assert all(cal_only[i] <= cal_only[i+1] + 1e-4
               for i in range(len(cal_only) - 1)), (
        "Calibration-only sigma should be non-decreasing in "
        "sigma_cal_relative"
    )
    print("  -> Passed.\n")

    # === Test 3: figure structure ===
    print("Test 3: figures have expected structure")
    fig_hist, ax_hist = analysis.figures["mc_histograms"]
    assert ax_hist.shape == (2, 2)
    fig_contrib, ax_contrib = analysis.figures["contributions"]
    # ax_contrib is either a list or a flat ndarray of axes
    n_contrib_axes = (
        len(ax_contrib) if isinstance(ax_contrib, list)
        else ax_contrib.size
    )
    assert n_contrib_axes == 3
    fig_sweep, ax_sweep = analysis.figures["calibration_sweep"]
    n_lines = len(ax_sweep.get_lines())
    assert n_lines >= 2, (
        f"calibration_sweep figure should have at least 2 lines "
        f"(total + cal-only); got {n_lines}"
    )
    print(f"  mc_histograms: 2x2 axes")
    print(f"  contributions: {n_contrib_axes} panels")
    print(f"  calibration_sweep: {n_lines} lines")
    print("  -> Passed.\n")

    # === Test 4: save_dir writes PNGs ===
    print("Test 4: save_dir writes 3 PNGs")
    tmp = Path(tempfile.mkdtemp(prefix="ss5_module4_validate_"))
    try:
        analyse_uncertainty(
            batch,
            cfg,
            save_dir=tmp,
        )
        for filename in [
            "mc_histograms.png",
            "contributions.png",
            "calibration_sweep.png",
        ]:
            path = tmp / filename
            assert path.exists(), f"Missing PNG: {path}"
            assert path.stat().st_size > 0, f"Empty PNG: {path}"
            print(f"  {filename:30s} -> "
                  f"{path.stat().st_size / 1024:.1f} KB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  -> Passed.\n")

    # === Test 5: budget table renders + Combined matches ===
    print("Test 5: budget table consistency")
    table = analysis.batch_uncertainty.to_budget_table()
    print("\n  Budget table (length_D50, width_D50 columns):")
    print(table[["length_D50", "width_D50"]].round(6))
    print()
    bu = analysis.batch_uncertainty
    combined = table.loc["Combined (quadrature)"]
    for d in ["length_D50", "width_D50", "length_D50_mass_w"]:
        assert np.isclose(
            combined[d],
            bu.entries[d].standard_uncertainty,
            rtol=1e-12,
        )
    print("  Combined row matches BatchUncertainty entries.")
    print("  -> Passed.\n")

    # Headline summary
    print("--- Headline: length_D50 ---")
    e = bu.entries["length_D50"]
    print(f"  point estimate: {e.point_estimate:.4f} mm")
    print(f"  95% CI:         [{e.ci_lower_95:.4f}, "
          f"{e.ci_upper_95:.4f}] mm")
    print(f"  sigma:          {e.standard_uncertainty:.4f} mm "
          f"({e.standard_uncertainty / e.point_estimate * 100:.2f}%)")
    print("\n  Calibration sweep on length_D50:")
    for r in analysis.calibration_sweep:
        print(f"    sigma_cal={r['sigma_cal_relative']*100:.1f}%: "
              f"sigma_total={r['sigma_total']:.4f} mm, "
              f"calibration-only={r['sigma_calibration']:.4f} mm")

    plt.close("all")
    print("\n--- All Component 4 tests passed. ---")


if __name__ == "__main__":
    main()