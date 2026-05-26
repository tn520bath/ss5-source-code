"""
Confirms five behaviours:

  1. Output shape: replicates dict has all 16 keys, each a 1D
     array of length n_iterations.
  2. Reproducibility: two runs with the same config and input
     produce byte-identical replicate arrays.
  3. Zero-source run: with sources_active=frozenset(), the
     replicates are all equal to the point estimates.
  4. Calibration-only run: scaling sigma_cal_relative by 2x
     scales the spread of length_D50 by approximately 2x.
  5. Bootstrap-only run: the spread of length_D50 follows the
     classical 1/sqrt(N) decay across two batch sizes.
"""

import time

import numpy as np

from ss5_module1_synthetic_generator.batch_assembly import generate_image
from ss5_module2_7_pipeline.ingestion import BatchAggregator
from ss5_module2_7_pipeline.config import IngestionParams

from ss5_module4_uncertainty.config import UncertaintyConfig
from ss5_module4_uncertainty.monte_carlo import (
    run_monte_carlo,
    ALL_SOURCES,
)


def _make_batch(n_fibres: int, seed: int):
    rng = np.random.default_rng(seed)
    aggregator = BatchAggregator(IngestionParams(images_per_batch=1))
    msg = generate_image(n_fibres=n_fibres, image_id=0, rng=rng)
    batch = aggregator.ingest(msg)
    assert batch is not None
    return batch


def main():
    print("--- Validation: Component 2 (Monte Carlo engine) ---\n")

    # === Test 1: shape ===
    print("Test 1: replicate arrays have correct shape")
    batch = _make_batch(n_fibres=300, seed=1)
    cfg = UncertaintyConfig(n_mc_iterations=50, rng_seed=1)
    t0 = time.time()
    mc = run_monte_carlo(batch, cfg, ALL_SOURCES)
    elapsed = time.time() - t0
    expected_keys = {
        "length_mean", "length_D10", "length_D50", "length_D90",
        "length_std",
        "width_mean", "width_D10", "width_D50", "width_D90",
        "width_std",
        "length_D50_mass_w", "fines_fraction_count",
        "lognormal_mu_length", "lognormal_sigma_length",
        "lognormal_mu_width", "lognormal_sigma_width",
    }
    assert set(mc.replicates.keys()) == expected_keys
    for arr in mc.replicates.values():
        assert arr.shape == (50,)
    print(f"  16 tracked descriptors, 50 replicates each.")
    print(f"  Elapsed: {elapsed:.2f}s "
          f"({elapsed / 50 * 1000:.1f} ms/iter)")
    print("  -> Passed.\n")

    # === Test 2: reproducibility ===
    print("Test 2: same seed -> identical replicates")
    mc1 = run_monte_carlo(batch, cfg, ALL_SOURCES)
    mc2 = run_monte_carlo(batch, cfg, ALL_SOURCES)
    for key in mc1.replicates:
        assert np.array_equal(mc1.replicates[key], mc2.replicates[key])
    print("  All 16 arrays byte-identical across runs.")
    print("  -> Passed.\n")

    # === Test 3: empty source set -> point estimates ===
    print("Test 3: sources_active=empty -> replicates == point est.")
    cfg_quick = UncertaintyConfig(n_mc_iterations=10, rng_seed=1)
    mc_zero = run_monte_carlo(batch, cfg_quick, frozenset())
    for key, point in mc_zero.point_estimates.items():
        assert np.allclose(mc_zero.replicates[key], point, rtol=1e-12)
    print("  All replicates equal the point estimates.")
    print("  -> Passed.\n")

    # === Test 4: calibration scaling ===
    print("Test 4: 2x sigma_cal -> ~2x spread in length_D50")
    cfg_low = UncertaintyConfig(
        n_mc_iterations=500,
        sigma_cal_relative=0.01,
        rng_seed=7,
    )
    cfg_high = UncertaintyConfig(
        n_mc_iterations=500,
        sigma_cal_relative=0.02,
        rng_seed=7,
    )
    mc_low = run_monte_carlo(batch, cfg_low, frozenset({"A"}))
    mc_high = run_monte_carlo(batch, cfg_high, frozenset({"A"}))
    sd_low = np.std(mc_low.replicates["length_D50"])
    sd_high = np.std(mc_high.replicates["length_D50"])
    ratio = sd_high / sd_low
    print(f"  SD(D50) at sigma_cal=1%: {sd_low:.5f}")
    print(f"  SD(D50) at sigma_cal=2%: {sd_high:.5f}")
    print(f"  ratio = {ratio:.3f}  (expect ~2.0)")
    assert 1.7 < ratio < 2.3, f"Ratio {ratio} not in [1.7, 2.3]"
    print("  -> Passed.\n")

    # === Test 5: 1/sqrt(N) scaling ===
    print("Test 5: bootstrap-only, SD(D50) shrinks like 1/sqrt(N)")
    cfg_boot = UncertaintyConfig(n_mc_iterations=500, rng_seed=11)
    batch_small = _make_batch(n_fibres=200, seed=2)
    batch_large = _make_batch(n_fibres=800, seed=2)
    mc_small = run_monte_carlo(batch_small, cfg_boot, frozenset({"F"}))
    mc_large = run_monte_carlo(batch_large, cfg_boot, frozenset({"F"}))
    sd_small = np.std(mc_small.replicates["length_D50"])
    sd_large = np.std(mc_large.replicates["length_D50"])
    ratio = sd_small / sd_large
    expected = np.sqrt(800 / 200)  # = 2
    print(f"  SD(D50) at N=200:  {sd_small:.5f}")
    print(f"  SD(D50) at N=800:  {sd_large:.5f}")
    print(f"  ratio = {ratio:.3f}  (expect ~{expected:.3f})")
    assert 1.5 < ratio < 2.5, f"Ratio {ratio} not in [1.5, 2.5]"
    print("  -> Passed.\n")

    print("--- All Component 2 tests passed. ---")


if __name__ == "__main__":
    main()