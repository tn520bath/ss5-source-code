"""
Confirms five behaviours:

  1. Output array shapes match the input length n.
  2. With sigma_cal=0 and no reconstruction flag, sigma(L) reduces
     to the base segmentation sigma. (Pure noise limit.)
  3. With sigma_L=sigma_w=0 and sigma_cal>0, sigma(AR)=0.
     (Calibration-cancellation property.)
  4. A flagged fibre's sigma(L) scales linearly with the
     reconstruction inflation factor.
  5. The full closed-form expression matches a hand calculation
     on a representative case.
"""

import numpy as np

from ss5_module4_uncertainty.config import UncertaintyConfig
from ss5_module4_uncertainty.gum_propagation import (
    compute_per_fibre_uncertainty,
)


def main():
    print("--- Validation: Component 1 (per-fibre GUM) ---\n")

    # === Test 1: shape ===
    print("Test 1: output shapes match input length")
    n = 100
    rng = np.random.default_rng(42)
    L = rng.uniform(1.0, 8.0, size=n)
    w = rng.uniform(0.05, 0.15, size=n)
    flag = rng.random(size=n) < 0.2

    u = compute_per_fibre_uncertainty(L, w, flag, UncertaintyConfig())
    assert u.sigma_length.shape == (n,)
    assert u.sigma_width.shape == (n,)
    assert u.sigma_aspect_ratio.shape == (n,)
    assert u.sigma_length_relative.shape == (n,)
    print("  -> Passed.\n")

    # === Test 2: sigma_cal=0 -> sigma(L)=sigma_L_base ===
    print("Test 2: sigma_cal=0, unflagged -> sigma(L) = sigma_L_base")
    cfg = UncertaintyConfig(sigma_cal_relative=0.0)
    u = compute_per_fibre_uncertainty(
        np.array([5.0]), np.array([0.1]), np.array([False]), cfg
    )
    sigma_L_base, _ = cfg.resolved_noise_params()
    assert np.isclose(u.sigma_length[0], sigma_L_base, rtol=1e-9)
    print(f"  sigma_L_base   = {sigma_L_base}")
    print(f"  sigma(L)       = {u.sigma_length[0]}")
    print("  -> Passed.\n")

    # === Test 3: noise=0, sigma_cal>0 -> sigma(AR)=0 ===
    print("Test 3: noise=0, sigma_cal>0 -> sigma(AR)=0 "
          "(calibration cancels)")
    cfg = UncertaintyConfig(
        sigma_cal_relative=0.02,
        sigma_length_mm=0.0,
        sigma_width_mm=0.0,
        use_module9_noise_params=False,
    )
    u = compute_per_fibre_uncertainty(
        np.array([5.0]), np.array([0.1]), np.array([False]), cfg
    )
    assert np.isclose(u.sigma_aspect_ratio[0], 0.0, atol=1e-12)
    print(f"  sigma(AR) = {u.sigma_aspect_ratio[0]:.3e}")
    print("  -> Passed.\n")

    # === Test 4: reconstruction inflation ===
    print("Test 4: high_reconstruction inflates sigma(L) by factor")
    cfg = UncertaintyConfig(
        sigma_cal_relative=0.0,           # isolate noise term
        reconstruction_inflation_factor=2.0,
    )
    L = np.array([5.0, 5.0])
    w = np.array([0.1, 0.1])
    flag = np.array([False, True])
    u = compute_per_fibre_uncertainty(L, w, flag, cfg)
    ratio = u.sigma_length[1] / u.sigma_length[0]
    assert np.isclose(ratio, 2.0, rtol=1e-9)
    print(f"  sigma(L) flagged / unflagged = {ratio}")
    print("  -> Passed.\n")

    # === Test 5: representative numerical example ===
    print("Test 5: representative case, hand-calculated check")
    cfg = UncertaintyConfig(sigma_cal_relative=0.01)
    sigma_L_base, sigma_w_base = cfg.resolved_noise_params()
    L = np.array([5.0])
    w = np.array([0.1])
    u = compute_per_fibre_uncertainty(L, w, np.array([False]), cfg)

    # Hand calculations
    sigma_L_hand = np.sqrt(sigma_L_base**2 + (5.0 * 0.01)**2)
    sigma_w_hand = np.sqrt(sigma_w_base**2 + (0.1 * 0.01)**2)
    sigma_AR_rel_hand = np.sqrt(
        (sigma_L_base / 5.0)**2 + (sigma_w_base / 0.1)**2
    )

    print(f"  L=5 mm, w=0.1 mm, sigma_cal=1%, no flag")
    print(f"  sigma(L)        hand={sigma_L_hand:.6f}, "
          f"got={u.sigma_length[0]:.6f}")
    print(f"  sigma(w)        hand={sigma_w_hand:.6f}, "
          f"got={u.sigma_width[0]:.6f}")
    print(f"  sigma(AR)/AR    hand={sigma_AR_rel_hand:.6f}, "
          f"got={u.sigma_aspect_ratio_relative[0]:.6f}")
    assert np.isclose(u.sigma_length[0], sigma_L_hand, rtol=1e-9)
    assert np.isclose(u.sigma_width[0], sigma_w_hand, rtol=1e-9)
    assert np.isclose(
        u.sigma_aspect_ratio_relative[0],
        sigma_AR_rel_hand,
        rtol=1e-9,
    )
    print("  -> Passed.\n")

    print("--- All Component 1 tests passed. ---")


if __name__ == "__main__":
    main()