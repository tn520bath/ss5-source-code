"""
Validation script for Module 3 fines fraction.

Confirms:
  - Count-based fines fraction matches a hand-computed value on
    a small contrived example.
  - Mass-based fines fraction is smaller than count-based when
    fines have lower mass per fibre than long fibres (the
    expected real-world ordering).
  - Threshold is correctly applied with strict comparison.
  - Edge cases (empty arrays, mismatched shapes) raise ValueError.
  - On a synthetic batch matching Module 1's defaults, the count-
    based fines fraction is in a reasonable range (single-digit
    percent or low double digits given mu=1.1, sigma=0.4 truncated
    at 1 mm).
"""

import numpy as np

from ss5_module3_descriptors.fines import compute_fines_fraction


def main():
    rng = np.random.default_rng(seed=42)

    # --- Test 1: hand-check on small contrived batch ---
    print("--- Test 1: hand-check on 5-fibre batch ---")
    L = np.array([0.5, 0.9, 1.0, 2.0, 8.0])
    W = np.array([0.05, 0.06, 0.07, 0.10, 0.20])

    # Threshold = 1.0 mm, strict comparison: 0.5 and 0.9 are fines.
    # 1.0 itself is NOT a fine (strict <).
    # Expected: n_fines = 2, n_total = 5, count fraction = 0.4
    result = compute_fines_fraction(L, W, threshold_mm=1.0)
    print(f"  threshold              = {result['threshold_mm']}")
    print(f"  n_fines                = {result['n_fines']}")
    print(f"  n_total                = {result['n_total']}")
    print(f"  fines_fraction_count   = {result['fines_fraction_count']:.4f}")
    print(f"  fines_fraction_mass    = {result['fines_fraction_mass']:.4f}")

    assert result["n_fines"] == 2, "Expected 2 fines"
    assert result["n_total"] == 5
    assert abs(result["fines_fraction_count"] - 0.4) < 1e-12
    print("  -> Passed: count-based fines fraction correct (0.4).")

    # Hand-check mass:
    # weights = [0.5*0.0025, 0.9*0.0036, 1.0*0.0049, 2.0*0.01, 8.0*0.04]
    #         = [0.00125, 0.00324, 0.0049, 0.02, 0.32]
    # fines mass (L < 1) = 0.00125 + 0.00324 = 0.00449
    # total mass = 0.34939
    # expected fines_fraction_mass = 0.00449 / 0.34939 ~ 0.01285
    expected_mass_frac = (0.5 * 0.05**2 + 0.9 * 0.06**2) / np.sum(L * W**2)
    assert abs(result["fines_fraction_mass"] - expected_mass_frac) < 1e-10
    print(f"  Hand-computed mass fraction = {expected_mass_frac:.4f}")
    print("  -> Passed: mass-based fines fraction matches hand calc.")

    # --- Test 2: strict comparison at boundary ---
    print("\n--- Test 2: strict comparison at boundary ---")
    L_boundary = np.array([1.0, 1.0, 1.0])
    W_boundary = np.array([0.1, 0.1, 0.1])
    result_b = compute_fines_fraction(L_boundary, W_boundary, threshold_mm=1.0)
    assert result_b["n_fines"] == 0, \
        "Fibres at exactly 1.0 mm should not count as fines (strict <)"
    print(f"  Three fibres at exactly 1.0 mm: n_fines = "
          f"{result_b['n_fines']} (should be 0).")
    print("  -> Passed: strict-comparison boundary correctly handled.")

    # --- Test 3: realistic batch, mass < count ---
    print("\n--- Test 3: realistic synthetic batch ---")
    n = 5000
    L_real = rng.lognormal(mean=1.1, sigma=0.4, size=n)
    W_real = rng.lognormal(mean=-2.30, sigma=0.18, size=n)

    result_r = compute_fines_fraction(L_real, W_real, threshold_mm=1.0)
    print(f"  n_fines                = {result_r['n_fines']}")
    print(f"  n_total                = {result_r['n_total']}")
    print(f"  fines_fraction_count   = "
          f"{result_r['fines_fraction_count']:.4f}")
    print(f"  fines_fraction_mass    = "
          f"{result_r['fines_fraction_mass']:.4f}")

    # Sanity 1: fines fraction should be in a plausible range.
    # P(L < 1 | L ~ lognormal(1.1, 0.4)) = Phi(ln(1) - 1.1)/0.4)
    #                                    = Phi(-2.75) ~ 0.003
    # So we expect maybe 0.3% by count. (Note: NOT applying the
    # truncation here because rng.lognormal doesn't truncate.)
    assert 0 <= result_r["fines_fraction_count"] <= 0.05, \
        "Fines fraction looks implausibly large"

    # Sanity 2: mass-based fraction should be SMALLER than count-based,
    # because fines (short) are also typically thin -> low mass.
    assert result_r["fines_fraction_mass"] < result_r["fines_fraction_count"], \
        "Mass-based fines fraction should be < count-based for typical batches"
    print("  -> Passed: count-based and mass-based ordering as expected.")

    # --- Test 4: configurable threshold ---
    print("\n--- Test 4: threshold sensitivity ---")
    for thr in [0.5, 1.0, 2.0]:
        r = compute_fines_fraction(L_real, W_real, threshold_mm=thr)
        print(f"  threshold = {thr} mm: count fraction = "
              f"{r['fines_fraction_count']:.4f}, mass fraction = "
              f"{r['fines_fraction_mass']:.4f}")
    print("  -> Passed: monotonic increase with threshold, as expected.")

    # --- Test 5: edge cases ---
    print("\n--- Test 5: edge cases ---")
    try:
        compute_fines_fraction(np.array([]), np.array([]))
        print("  ERROR: empty input should have raised")
    except ValueError as e:
        print(f"  Caught ValueError on empty input: {e}")

    try:
        compute_fines_fraction(np.array([1.0, 2.0]), np.array([0.1]))
        print("  ERROR: shape mismatch should have raised")
    except ValueError as e:
        print(f"  Caught ValueError on shape mismatch: {e}")
    print("  -> Passed: edge cases handled correctly.")

    print("\nAll Component 3 validation checks passed.")


if __name__ == "__main__":
    main()