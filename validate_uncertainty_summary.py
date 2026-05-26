"""
Validation script for the 
uncertainty summary and contribution attribution.

Confirms five behaviours:

  1. summarise_uncertainty produces a BatchUncertainty with
     entries for all 16 tracked descriptors.
  2. Every CI brackets its point estimate
     (ci_lower < point < ci_upper).
  3. Every descriptor's standard_uncertainty is greater than each
     of its individual contributions (basic consistency check on
     the variance decomposition).
  4. The summed-in-quadrature contributions explain "most" of the
     total variance: sum(sigma_X^2) is within +/- 30% of
     sigma_all^2 for the headline length_D50 descriptor.
     (Exact equality is not expected because LOO subtraction
     captures interaction-free contributions only; the 30%
     window reflects acceptable nonlinear interaction plus
     finite-MC noise.)
  5. to_budget_table() returns the expected DataFrame shape and
     the Combined row equals the standard uncertainties of each
     descriptor in the entries dict.
"""

import time

import numpy as np

from ss5_module1_synthetic_generator.batch_assembly import generate_image
from ss5_module2_7_pipeline.ingestion import BatchAggregator
from ss5_module2_7_pipeline.config import IngestionParams

from ss5_module4_uncertainty.config import UncertaintyConfig
from ss5_module4_uncertainty.uncertainty_summary import (
    summarise_uncertainty,
    DEFAULT_BUDGET_DESCRIPTORS,
    SOURCE_LABELS,
)


def _make_batch(n_fibres: int, seed: int):
    rng = np.random.default_rng(seed)
    aggregator = BatchAggregator(IngestionParams(images_per_batch=1))
    msg = generate_image(n_fibres=n_fibres, image_id=0, rng=rng)
    batch = aggregator.ingest(msg)
    assert batch is not None
    return batch


def main():
    print("--- Validation: Component 3 (uncertainty summary) ---\n")

    batch = _make_batch(n_fibres=400, seed=3)
    cfg = UncertaintyConfig(
        n_mc_iterations=400,   # enough to stabilise the LOO subtraction
        rng_seed=3,
    )

    print("Running 6 MCs (all-on + 5 LOO) at 400 iterations each...")
    t0 = time.time()
    bu = summarise_uncertainty(batch, cfg)
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s\n")

    # === Test 1: structure ===
    print("Test 1: BatchUncertainty has 16 entries")
    assert len(bu.entries) == 16
    assert bu.batch_id == batch.batch_id
    assert bu.n_fibres == len(batch.fibre_records)
    assert bu.n_mc_iterations == cfg.n_mc_iterations
    print(f"  batch_id={bu.batch_id}, n_fibres={bu.n_fibres}, "
          f"K={bu.n_mc_iterations}")
    print("  -> Passed.\n")

    # === Test 2: CIs bracket point estimates ===
    print("Test 2: CIs bracket point estimates")
    for name, e in bu.entries.items():
        assert e.ci_lower_95 <= e.point_estimate <= e.ci_upper_95, (
            f"CI does not bracket point for {name}: "
            f"[{e.ci_lower_95}, {e.ci_upper_95}], "
            f"point={e.point_estimate}"
        )
    print(f"  All 16 descriptors: ci_lower <= point <= ci_upper.")
    print("  -> Passed.\n")

    # === Test 3: contributions vs standard_uncertainty ===
    print("Test 3: each contribution <= standard_uncertainty")
    for name, e in bu.entries.items():
        for src, val in e.contributions.items():
            assert val <= e.standard_uncertainty + 1e-12, (
                f"Contribution {src}={val} exceeds sigma_total="
                f"{e.standard_uncertainty} for descriptor {name}"
            )
    print("  All per-source contributions <= total sigma.")
    print("  -> Passed.\n")

    # === Test 4: quadrature sum vs total variance for length_D50 ===
    print("Test 4: sum of squared contributions ~ total variance "
          "(length_D50)")
    e = bu.entries["length_D50"]
    var_total = e.standard_uncertainty ** 2
    var_sum = sum(v ** 2 for v in e.contributions.values())
    ratio = var_sum / var_total if var_total > 0 else 0.0
    print(f"  sigma_total^2  = {var_total:.6e}")
    print(f"  sum sigma_X^2  = {var_sum:.6e}")
    print(f"  ratio          = {ratio:.3f}")
    assert 0.7 <= ratio <= 1.3, (
        f"Quadrature sum / total variance = {ratio} outside "
        f"[0.7, 1.3] window"
    )
    print("  -> Passed.\n")

    # === Test 5: budget table shape and Combined row ===
    print("Test 5: budget table shape + Combined row consistency")
    table = bu.to_budget_table()
    print("\nBudget table preview (length_D50 column):")
    print(table[["length_D50", "width_D50"]].round(6))
    print()
    expected_n_rows = len(SOURCE_LABELS) + 1 + 2  # sources + Combined + E + H
    assert table.shape == (expected_n_rows, len(DEFAULT_BUDGET_DESCRIPTORS))
    # Combined row equals the total sigma per descriptor
    combined_row = table.loc["Combined (quadrature)"]
    for d in DEFAULT_BUDGET_DESCRIPTORS:
        assert np.isclose(
            combined_row[d],
            bu.entries[d].standard_uncertainty,
            rtol=1e-12,
        )
    print(f"  Shape = {table.shape} as expected.")
    print("  Combined row matches per-descriptor sigma values.")
    print("  -> Passed.\n")

    # === Bonus: print the headline descriptor's full breakdown ===
    print("--- Headline summary: length_D50 ---")
    e = bu.entries["length_D50"]
    print(f"  point estimate: {e.point_estimate:.4f} mm")
    print(f"  sigma:          {e.standard_uncertainty:.4f} mm "
          f"({e.standard_uncertainty / e.point_estimate * 100:.2f}%)")
    print(f"  95% CI:         [{e.ci_lower_95:.4f}, "
          f"{e.ci_upper_95:.4f}] mm")
    print("  Contributions:")
    for src in ["A", "B", "D", "F", "G"]:
        v = e.contributions[src]
        pct = v / e.standard_uncertainty * 100 if e.standard_uncertainty > 0 else 0
        print(f"    {SOURCE_LABELS[src]:35s} {v:.4f} mm  "
              f"({pct:5.1f}% of total sigma)")

    print("\n--- All Component 3 tests passed. ---")


if __name__ == "__main__":
    main()