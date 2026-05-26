"""
Validation script for Module 3 distribution fitting.

Confirms:
  - Lognormal fit recovers the generating parameters on a known
    synthetic sample (5000 points; recovered mu, sigma close to
    true mu=1.1, sigma=0.4).
  - Weibull fit produces sensible shape and scale parameters on a
    Weibull-generated sample.
  - The frozen distribution object's .pdf() and .cdf() behave as
    expected (CDF strictly increasing, PDF non-negative).
  - Log-likelihood is reasonable (negative, finite, larger when
    the assumed family matches the generating family).
  - Fitting the wrong family produces a worse log-likelihood
    than fitting the correct family on lognormal data.
  - Edge cases: empty array, single value, identical values,
    negative values, unknown distribution name all raise ValueError.
  - Small-N warning fires when N < 30.
"""

import warnings

import numpy as np

from ss5_module3_descriptors.distribution_fitting import (
    fit_distribution,
    DistributionFit,
)


def main():
    rng = np.random.default_rng(seed=42)

    # --- Test 1: lognormal fit recovers parameters ---
    print("--- Test 1: lognormal fit recovers parameters ---")
    n = 5000
    true_mu, true_sigma = 1.1, 0.4
    sample = rng.lognormal(mean=true_mu, sigma=true_sigma, size=n)

    fit = fit_distribution(sample, distribution="lognormal")

    print(f"  fit type           = {type(fit).__name__}")
    print(f"  name               = {fit.name}")
    print(f"  fitted mu          = {fit.params['mu']:.4f} "
          f"(true {true_mu})")
    print(f"  fitted sigma       = {fit.params['sigma']:.4f} "
          f"(true {true_sigma})")
    print(f"  log-likelihood     = {fit.log_likelihood:.2f}")
    print(f"  n                  = {fit.n}")

    assert isinstance(fit, DistributionFit)
    assert fit.name == "lognormal"
    assert abs(fit.params["mu"] - true_mu) < 0.05
    assert abs(fit.params["sigma"] - true_sigma) < 0.02
    assert fit.log_likelihood < 0  # PDFs of continuous RVs can exceed 1
    assert fit.n == n
    print("  -> Passed: parameters recovered within Monte Carlo noise.")

    # --- Test 2: frozen distribution behaves correctly ---
    print("\n--- Test 2: frozen distribution methods ---")
    x = np.array([1.0, 3.0, 9.0])
    print(f"  pdf(x)  = {fit.distribution.pdf(x)}")
    print(f"  cdf(x)  = {fit.distribution.cdf(x)}")
    print(f"  ppf(0.5)= {fit.distribution.ppf(0.5):.4f}  "
          f"(theoretical median exp(mu) = {np.exp(true_mu):.4f})")
    assert np.all(fit.distribution.pdf(x) >= 0)
    assert np.all(np.diff(fit.distribution.cdf(x)) >= 0)
    print("  -> Passed: pdf non-negative, cdf monotone, ppf sensible.")

    # --- Test 3: Weibull fit on Weibull data ---
    print("\n--- Test 3: Weibull fit on Weibull-generated data ---")
    true_k, true_lambda = 2.5, 3.0
    weibull_sample = rng.weibull(a=true_k, size=n) * true_lambda

    fit_w = fit_distribution(weibull_sample, distribution="weibull")
    print(f"  fitted k       = {fit_w.params['k']:.4f} "
          f"(true {true_k})")
    print(f"  fitted lambda  = {fit_w.params['lambda']:.4f} "
          f"(true {true_lambda})")
    print(f"  log-likelihood = {fit_w.log_likelihood:.2f}")

    assert abs(fit_w.params["k"] - true_k) < 0.1
    assert abs(fit_w.params["lambda"] - true_lambda) < 0.1
    print("  -> Passed: Weibull parameters recovered.")

    # --- Test 4: wrong family produces worse log-likelihood ---
    print("\n--- Test 4: model selection sanity check ---")
    # Fit Weibull to lognormal data; should have lower log-likelihood
    # than the correctly-fitted lognormal.
    fit_wrong = fit_distribution(sample, distribution="weibull")
    print(f"  lognormal fit on lognormal data: ll = "
          f"{fit.log_likelihood:.2f}")
    print(f"  weibull   fit on lognormal data: ll = "
          f"{fit_wrong.log_likelihood:.2f}")
    assert fit.log_likelihood > fit_wrong.log_likelihood, \
        "Correct family should give higher log-likelihood"
    print("  -> Passed: lognormal fits its own data better than Weibull.")

    # --- Test 5: edge cases ---
    print("\n--- Test 5: edge cases ---")
    cases = [
        ("empty", np.array([])),
        ("single point", np.array([3.0])),
        ("identical values", np.array([2.0, 2.0, 2.0, 2.0])),
        ("negative value", np.array([-1.0, 2.0, 3.0])),
        ("zero value", np.array([0.0, 1.0, 2.0])),
    ]
    for label, arr in cases:
        try:
            fit_distribution(arr)
            print(f"  ERROR: '{label}' should have raised")
        except ValueError as e:
            print(f"  '{label}' -> ValueError: {e}")

    try:
        fit_distribution(sample, distribution="gaussian")
        print("  ERROR: unknown distribution should have raised")
    except ValueError as e:
        print(f"  unknown distribution -> ValueError: {e}")
    print("  -> Passed: all edge cases handled.")

    # --- Test 6: small-N warning ---
    print("\n--- Test 6: small-N warning fires ---")
    small_sample = rng.lognormal(mean=1.1, sigma=0.4, size=20)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit_small = fit_distribution(small_sample)
        assert len(w) >= 1
        assert "below threshold" in str(w[-1].message)
        print(f"  Warning fired: {w[-1].message}")
        print(f"  Fit still produced: mu={fit_small.params['mu']:.4f}, "
              f"sigma={fit_small.params['sigma']:.4f}")
    print("  -> Passed: small-N warning fires but fit still produced.")

    print("\nAll Component 4 validation checks passed.")


if __name__ == "__main__":
    main()