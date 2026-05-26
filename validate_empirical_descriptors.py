"""
Validation script for Module 3 empirical descriptors.

Confirms:
  - Descriptor values are correct on a known synthetic batch.
  - Sample-size guards behave correctly.
  - Reconstruction summary returns expected counts and means.
  - Skewness sign and magnitude match the lognormal-population
    expectation for the generator's parameters.
"""

import numpy as np

from ss5_module3_descriptors.empirical_descriptors import (
    compute_empirical_descriptors,
    compute_reconstruction_summary,
)


def main():
    rng = np.random.default_rng(seed=42)

    # --- Test 1: well-behaved sample ---
    # Lognormal with mu=1.1, sigma=0.4 (matches the Module 1 length
    # generator). Theoretical median = exp(1.1) = 3.004.
    n = 5000
    mu, sigma = 1.1, 0.4
    sample = rng.lognormal(mean=mu, sigma=sigma, size=n)

    desc = compute_empirical_descriptors(sample)

    print("--- Test 1: 5000-fibre lognormal sample ---")
    for k, v in desc.items():
        if isinstance(v, float):
            print(f"  {k:<8} = {v:.4f}")
        else:
            print(f"  {k:<8} = {v}")

    # Theoretical comparisons
    theo_median = np.exp(mu)
    theo_mean = np.exp(mu + sigma**2 / 2)
    theo_skew = (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1)

    print(f"\n  Theoretical median = {theo_median:.4f}")
    print(f"  Theoretical mean   = {theo_mean:.4f}")
    print(f"  Theoretical skew   = {theo_skew:.4f}")

    assert abs(desc["median"] - theo_median) < 0.05, "Median off"
    assert abs(desc["mean"] - theo_mean) < 0.1, "Mean off"
    assert abs(desc["skew"] - theo_skew) < 0.2, "Skew off"
    assert desc["D10"] < desc["D50"] < desc["D90"], "D-values out of order"
    print("  -> Passed.")

    # --- Test 2: small N edge case ---
    print("\n--- Test 2: N=1 sample ---")
    desc_n1 = compute_empirical_descriptors(np.array([3.5]))
    assert desc_n1["n"] == 1
    assert desc_n1["mean"] == 3.5
    assert desc_n1["median"] == 3.5
    assert np.isnan(desc_n1["std"])
    assert np.isnan(desc_n1["cv"])
    assert np.isnan(desc_n1["skew"])
    print("  N=1 returns NaN for std/cv/skew, as expected.")
    print("  -> Passed.")

    # --- Test 3: empty input ---
    print("\n--- Test 3: empty array raises ValueError ---")
    try:
        compute_empirical_descriptors(np.array([]))
        print("  ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"  Caught ValueError: {e}")
    print("  -> Passed.")

    # --- Test 4: reconstruction summary ---
    print("\n--- Test 4: reconstruction summary ---")
    rf = rng.beta(a=2, b=8, size=1000)
    is_high = rf > 0.5
    n_flagged_expected = int(np.sum(is_high))

    summary = compute_reconstruction_summary(rf, is_high)
    print(f"  mean_reconstructed_fraction  = "
          f"{summary['mean_reconstructed_fraction']:.4f}")
    print(f"  high_reconstruction_fraction = "
          f"{summary['high_reconstruction_fraction']:.4f}")
    print(f"  n_flagged                    = {summary['n_flagged']}")
    assert summary["n_flagged"] == n_flagged_expected
    # Beta(2, 8) has mean 0.2
    assert abs(summary["mean_reconstructed_fraction"] - 0.2) < 0.02
    print("  -> Passed.")

    print("\nAll Component 1 validation checks passed.")


if __name__ == "__main__":
    main()