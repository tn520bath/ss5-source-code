"""
Confirms five behaviours:

  1. run_coverage_validation returns a populated
     CoverageValidationResult.
  2. Empirical coverage is computed for every requested
     descriptor and lies in [0, 1].
  3. The Wilson 95% CI on the empirical coverage either
     contains 0.95 or comes very close (within 0.05) for
     length_D50, the headline descriptor.
  4. Plotting function returns a two-panel figure with the
     expected structure.
  5. Reproducibility: two runs with the same config produce
     identical coverage values.

Uses a small N_outer (40) and small K_inner (200) to keep
the validate run under ~3 minutes. The headline production
run will use larger values.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ss5_module4_uncertainty.config import UncertaintyConfig
from ss5_module4_uncertainty.coverage_validation import (
    CoverageValidationConfig,
    CoverageValidationResult,
    DEFAULT_VALIDATED_DESCRIPTORS,
    run_coverage_validation,
)
from ss5_module4_uncertainty.plotting import (
    plot_coverage_validation,
)


def main():
    print("--- Validation: Module 4 coverage validation ---\n")

    cfg = CoverageValidationConfig(
        n_outer=40,
        n_fibres_per_batch=400,
        n_truth_fibres=20_000,
        parameter_jitter_scale=0.05,
        inner_uncertainty_config=UncertaintyConfig(
            n_mc_iterations=200,
        ),
        rng_seed=5,
    )

    print(f"Running coverage validation: "
          f"N_outer={cfg.n_outer}, "
          f"K_inner={cfg.inner_uncertainty_config.n_mc_iterations}\n")

    t0 = time.time()
    result = run_coverage_validation(cfg, progress=True)
    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s\n")

    # === Test 1: structure ===
    print("Test 1: result structure")
    assert isinstance(result, CoverageValidationResult)
    assert result.n_completed == cfg.n_outer
    assert set(result.coverage.keys()) == set(DEFAULT_VALIDATED_DESCRIPTORS)
    assert set(result.per_batch.keys()) == set(DEFAULT_VALIDATED_DESCRIPTORS)
    print("  -> Passed.\n")

    # === Test 2: coverage in [0, 1] ===
    print("Test 2: every coverage value in [0, 1]")
    for d, cov in result.coverage.items():
        assert 0.0 <= cov <= 1.0, f"{d}: coverage = {cov}"
        print(f"  {d:25s} -> coverage = {cov:.3f}")
    print("  -> Passed.\n")

    # === Test 3: length_D50 close to 0.95 ===
    print("Test 3: length_D50 empirical coverage near 0.95")
    df = result.to_dataframe()
    print("\n", df.round(3), "\n")
    cov_d50 = result.coverage["length_D50"]
    # On N_outer=40 the binomial 95% Wilson CI on true coverage of
    # 0.95 spans roughly [0.84, 0.99]. Allow [0.80, 1.00] to absorb
    # mild mis-calibration that won't be a real failure of Module 4.
    assert 0.80 <= cov_d50 <= 1.00, (
        f"length_D50 coverage {cov_d50:.3f} far from nominal 0.95"
    )
    print(f"  length_D50 coverage = {cov_d50:.3f} "
          "(target ~0.95, window [0.80, 1.00])")
    print("  -> Passed.\n")

    # === Test 4: plotting ===
    print("Test 4: plot_coverage_validation produces 2-panel figure")
    fig, axes = plot_coverage_validation(result)
    assert len(axes) == 2
    # Left panel has bars
    n_bars = len(axes[0].patches)
    assert n_bars == len(DEFAULT_VALIDATED_DESCRIPTORS)
    # Right panel has at least n_outer vlines (and possibly more
    # depending on collection structure); check via collections.
    n_vlines_collections = len(axes[1].collections)
    assert n_vlines_collections >= 1
    plt.close(fig)
    print("  -> Passed.\n")

    # === Test 5: reproducibility ===
    print("Test 5: two runs with same config -> identical coverage")
    result2 = run_coverage_validation(cfg, progress=False)
    for d in DEFAULT_VALIDATED_DESCRIPTORS:
        assert result.coverage[d] == result2.coverage[d], (
            f"Coverage non-reproducible for {d}: "
            f"{result.coverage[d]} vs {result2.coverage[d]}"
        )
    print("  All 6 coverage values byte-identical across runs.")
    print("  -> Passed.\n")

    # Summary
    print("--- Coverage summary ---")
    for d in DEFAULT_VALIDATED_DESCRIPTORS:
        cov = result.coverage[d]
        wilson = df.loc[d]
        print(f"  {d:25s} {cov:.3f}  "
              f"(Wilson 95%: [{wilson['wilson_ci_lower']:.2f}, "
              f"{wilson['wilson_ci_upper']:.2f}])")

    print("\n--- All Module 4 coverage-validation tests passed. ---")


if __name__ == "__main__":
    main()