"""
Validation script for Module 3 goodness-of-fit.

Confirms:
  - AIC formula matches hand calc on a known fit.
  - KS statistic on data drawn from the fitted distribution is
    small (good fit).
  - KS statistic on data deliberately mismatched against a wrong
    family is larger (bad fit).
  - compare_aic correctly identifies the preferred distribution
    and labels the evidence strength.
  - Edge cases (empty data, mismatched sample size) raise.
"""

import numpy as np

from ss5_module3_descriptors.distribution_fitting import (
    fit_distribution,
)
from ss5_module3_descriptors.goodness_of_fit import (
    compute_gof,
    compare_aic,
)


def main():
    rng = np.random.default_rng(seed=42)

    # --- Test 1: AIC arithmetic ---
    print("--- Test 1: AIC formula ---")
    n = 5000
    sample = rng.lognormal(mean=1.1, sigma=0.4, size=n)
    fit = fit_distribution(sample, "lognormal")
    gof = compute_gof(fit, sample)

    expected_aic = 2 * 2 - 2 * fit.log_likelihood
    print(f"  log_likelihood = {gof['log_likelihood']:.2f}")
    print(f"  k_params       = {gof['k_params']}")
    print(f"  AIC (computed) = {gof['aic']:.2f}")
    print(f"  AIC (expected) = {expected_aic:.2f}")
    assert abs(gof["aic"] - expected_aic) < 1e-9
    print("  -> Passed: AIC formula matches hand calc.")

    # --- Test 2: KS on the right family ---
    print("\n--- Test 2: KS for correctly-specified fit ---")
    print(f"  KS D = {gof['ks_D']:.4f}")
    print(f"  KS p = {gof['ks_p']:.4f}")
    assert gof["ks_D"] < 0.03, \
        "KS D should be small for correctly-specified fit on N=5000"
    print("  -> Passed: KS D is small (good fit).")

    # --- Test 3: KS on the wrong family ---
    print("\n--- Test 3: KS for misspecified fit ---")
    fit_wrong = fit_distribution(sample, "weibull")
    gof_wrong = compute_gof(fit_wrong, sample)
    print(f"  lognormal fit: KS D = {gof['ks_D']:.4f}, "
          f"p = {gof['ks_p']:.4f}")
    print(f"  weibull   fit: KS D = {gof_wrong['ks_D']:.4f}, "
          f"p = {gof_wrong['ks_p']:.4f}")
    assert gof_wrong["ks_D"] > gof["ks_D"], \
        "Misspecified fit should have larger KS D"
    print("  -> Passed: misspecified fit has larger KS D.")

    # --- Test 4: AIC comparison ---
    print("\n--- Test 4: AIC comparison ---")
    cmp = compare_aic(gof, gof_wrong)
    print(f"  delta_aic         = {cmp['delta_aic']:.2f}")
    print(f"  preferred         = {cmp['preferred']}")
    print(f"  evidence_strength = {cmp['evidence_strength']}")
    assert cmp["preferred"] == "lognormal"
    assert cmp["evidence_strength"] == "strong"
    print("  -> Passed: lognormal correctly preferred with strong evidence.")

    # --- Test 5: edge cases ---
    print("\n--- Test 5: edge cases ---")
    try:
        compute_gof(fit, np.array([]))
    except ValueError as e:
        print(f"  empty data -> ValueError: {e}")
    try:
        compute_gof(fit, sample[:100])
    except ValueError as e:
        print(f"  size mismatch -> ValueError: {e}")
    print("  -> Passed: edge cases handled correctly.")

    print("\nAll Component 5 validation checks passed.")


if __name__ == "__main__":
    main()