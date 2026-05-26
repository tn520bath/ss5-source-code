"""
Validation of the literature-grounded synthetic target generator.

Confirms:
  - When target generation is disabled (default), messages do not
    contain a synthetic_targets field.
  - When enabled, each message gains a 'synthetic_targets' dict
    with the configured target name, a '_truth' counterpart, and
    a '_descriptors' record.
  - Saturation behaves as designed: increasing length above
    length_sat_mm does NOT increase the target; doubling length
    below saturation DOES increase the target.
  - Sign of each term is correct (positive for length and aspect
    ratio; negative for fines and CV).
  - Across a 200-image dataset, the target lies in a realistic
    MOR-like range (roughly 0-30 MPa).
  - The deterministic _truth value reconstructs exactly from the
    saved descriptors and the spec's coefficients.
  - Reproducibility: same seed -> identical targets and noise.
"""

import numpy as np

from ss5_module1_synthetic_generator.config import (
    DatasetGenerationParams,
    TargetGenerationParams,
    LiteratureGroundedTargetSpec,
)
from ss5_module1_synthetic_generator.batch_assembly import generate_image
from ss5_module1_synthetic_generator.dataset_generator import generate_dataset
from ss5_module1_synthetic_generator.targets import (
    _evaluate_target,
    _compute_descriptors,
)


def main():
    print("=== Test 1: disabled by default ===")
    rng = np.random.default_rng(seed=42)
    msg = generate_image(n_fibres=200, rng=rng)
    assert "synthetic_targets" not in msg, \
        "Default behaviour should produce no synthetic_targets field"
    print("  Default disabled: synthetic_targets absent. OK.")

    print("\n=== Test 2: enabled with default literature-grounded spec ===")
    rng = np.random.default_rng(seed=42)
    target_params = TargetGenerationParams(enabled=True)
    msg = generate_image(
        n_fibres=200, target_params=target_params, rng=rng
    )
    assert "synthetic_targets" in msg
    assert "synthetic_MOR" in msg["synthetic_targets"]
    assert "synthetic_MOR_truth" in msg["synthetic_targets"]
    assert "synthetic_MOR_descriptors" in msg["synthetic_targets"]
    print(f"  synthetic_targets keys: {list(msg['synthetic_targets'].keys())}")
    print(f"  noisy value : {msg['synthetic_targets']['synthetic_MOR']:.3f} MPa")
    print(f"  truth value : {msg['synthetic_targets']['synthetic_MOR_truth']:.3f} MPa")
    print(f"  descriptors : {msg['synthetic_targets']['synthetic_MOR_descriptors']}")

    print("\n=== Test 3: target reconstructs exactly from saved descriptors ===")
    descriptors = msg["synthetic_targets"]["synthetic_MOR_descriptors"]
    saved_truth = msg["synthetic_targets"]["synthetic_MOR_truth"]
    spec = target_params.spec
    reconstructed = _evaluate_target(descriptors, spec)
    assert abs(reconstructed - saved_truth) < 1e-9, \
        f"Truth ({saved_truth}) != reconstruction ({reconstructed})"
    print(f"  saved truth        = {saved_truth:.6f}")
    print(f"  reconstructed truth = {reconstructed:.6f}")
    print("  Exact reconstruction. OK.")

    print("\n=== Test 4: saturation behaviour on length ===")
    base_descriptors = {
        "D50_length": 3.0,
        "D50_aspect": 30.0,
        "fines_fraction": 0.05,
        "CV_length": 0.4,
    }
    base_y = _evaluate_target(base_descriptors, spec)

    longer = dict(base_descriptors); longer["D50_length"] = 5.0
    longer_y = _evaluate_target(longer, spec)

    saturated = dict(base_descriptors); saturated["D50_length"] = 6.0
    saturated_y = _evaluate_target(saturated, spec)

    super_saturated = dict(base_descriptors); super_saturated["D50_length"] = 9.0
    super_saturated_y = _evaluate_target(super_saturated, spec)

    print(f"  D50_length=3.0 -> {base_y:.3f} MPa")
    print(f"  D50_length=5.0 -> {longer_y:.3f} MPa  (should exceed base)")
    print(f"  D50_length=6.0 -> {saturated_y:.3f} MPa  (saturation onset)")
    print(f"  D50_length=9.0 -> {super_saturated_y:.3f} MPa  (above saturation)")

    assert longer_y > base_y, "Length increase below saturation should raise target"
    assert abs(super_saturated_y - saturated_y) < 1e-9, \
        "Length increase above saturation should NOT raise target"
    print("  Saturation correctly active. OK.")

    print("\n=== Test 5: sign of each term ===")
    # Length: positive
    d = dict(base_descriptors); d["D50_length"] = 2.0
    y_short = _evaluate_target(d, spec)
    assert y_short < base_y, "Shorter length should reduce target"
    # Aspect ratio: positive
    d = dict(base_descriptors); d["D50_aspect"] = 15.0
    y_low_ar = _evaluate_target(d, spec)
    assert y_low_ar < base_y, "Lower AR should reduce target"
    # Fines: negative
    d = dict(base_descriptors); d["fines_fraction"] = 0.30
    y_high_fines = _evaluate_target(d, spec)
    assert y_high_fines < base_y, "Higher fines should reduce target"
    # CV: negative
    d = dict(base_descriptors); d["CV_length"] = 0.7
    y_high_cv = _evaluate_target(d, spec)
    assert y_high_cv < base_y, "Higher CV should reduce target"

    print(f"  base                       = {base_y:.3f} MPa")
    print(f"  shorter length (2.0 mm)    = {y_short:.3f} MPa  (lower, OK)")
    print(f"  lower AR (15)              = {y_low_ar:.3f} MPa  (lower, OK)")
    print(f"  higher fines (0.30)        = {y_high_fines:.3f} MPa  (lower, OK)")
    print(f"  higher CV (0.7)            = {y_high_cv:.3f} MPa  (lower, OK)")
    print("  All signs correct. OK.")

    print("\n=== Test 6: realistic range across a 200-image dataset ===")
    rng = np.random.default_rng(seed=42)
    dataset_params = DatasetGenerationParams(
        n_images=200, target_n_fibres=500, variation_scale=0.3
    )
    messages = generate_dataset(
        dataset_params=dataset_params,
        target_params=TargetGenerationParams(enabled=True),
        rng=rng,
    )
    targets_noisy = np.array(
        [m["synthetic_targets"]["synthetic_MOR"] for m in messages]
    )
    targets_truth = np.array(
        [m["synthetic_targets"]["synthetic_MOR_truth"] for m in messages]
    )
    print(f"  truth: mean={targets_truth.mean():.2f} MPa, "
          f"sd={targets_truth.std():.2f}, "
          f"range=[{targets_truth.min():.2f}, {targets_truth.max():.2f}]")
    print(f"  noisy: mean={targets_noisy.mean():.2f} MPa, "
          f"sd={targets_noisy.std():.2f}, "
          f"range=[{targets_noisy.min():.2f}, {targets_noisy.max():.2f}]")

    # Sanity: noisy SD should exceed truth SD by approximately
    # spec.noise_std (in quadrature). Confirms noise is being added
    # at the right scale.
    extra_sd = np.sqrt(max(targets_noisy.var() - targets_truth.var(), 0.0))
    print(f"  empirical noise SD ~= {extra_sd:.3f} MPa "
          f"(spec.noise_std = {spec.noise_std})")
    assert 0.5 * spec.noise_std < extra_sd < 2.0 * spec.noise_std, \
        "Noise level inconsistent with spec.noise_std"
    print("  Noise level matches spec. OK.")

    # Sanity: range should cover roughly 4-26 MPa as designed.
    assert targets_truth.min() > 0.0, "Target should be positive"
    assert targets_truth.max() < 35.0, "Target should not exceed ~30 MPa"
    print("  Realistic MOR-like range. OK.")

    print("\n=== Test 7: reproducibility ===")
    rng_a = np.random.default_rng(seed=999)
    rng_b = np.random.default_rng(seed=999)
    msg_a = generate_image(
        n_fibres=100,
        target_params=TargetGenerationParams(enabled=True),
        rng=rng_a,
    )
    msg_b = generate_image(
        n_fibres=100,
        target_params=TargetGenerationParams(enabled=True),
        rng=rng_b,
    )
    assert msg_a == msg_b, "Identical seeds should produce identical messages"
    print("  Same seed produces identical targets. OK.")

    print("\nAll Component 1 (literature-grounded target) checks passed.")


if __name__ == "__main__":
    main()