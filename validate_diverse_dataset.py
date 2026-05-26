"""
Validation of generate_diverse_dataset.

Confirms:
  - Function returns the configured number of messages with correct
    structural fields (image_id, char list, pixel_size_mm).
  - Per-image regime sampling produces measurable variation in all
    four target-driving descriptors: D50_length, D50_aspect,
    fines_fraction, CV_length.
  - Descriptor coverage is materially wider than the standard
    generate_dataset (with variation_scale = 0.3) on the descriptor
    that matters most for Module 6: fines_fraction.
  - Targets attach correctly when target generation is enabled, and
    the synthetic_MOR values span a wider range than under the
    default generator.
  - Reproducibility: identical seeds produce identical datasets.
  - Latin Hypercube path is exercised when scipy is available and
    the fallback uniform path is exercised when it is not.
"""

import numpy as np

from ss5_module1_synthetic_generator.config import (
    DatasetGenerationParams,
    DiverseDatasetParams,
    TargetGenerationParams,
)
from ss5_module1_synthetic_generator.dataset_generator import (
    generate_dataset,
    generate_diverse_dataset,
)
from ss5_module1_synthetic_generator.targets import _compute_descriptors


def _per_image_descriptors(messages):
    """Helper: extract per-image (D50_L, D50_AR, fines, CV_L) tuples."""
    rows = []
    for msg in messages:
        d = _compute_descriptors(msg)
        rows.append((
            d["D50_length"],
            d["D50_aspect"],
            d["fines_fraction"],
            d["CV_length"],
        ))
    return np.array(rows)


def main():
    print("=== Test 1: structural sanity ===")
    rng = np.random.default_rng(seed=42)
    messages = generate_diverse_dataset(
        diverse_params=DiverseDatasetParams(n_images=50),
        rng=rng,
    )
    assert len(messages) == 50
    assert all("char" in m for m in messages)
    assert all("pixel_size_mm" in m for m in messages)
    assert [m["image_id"] for m in messages] == list(range(50))
    print(f"  Generated {len(messages)} images with sequential IDs. OK.")

    print("\n=== Test 2: descriptor variation across the dataset ===")
    descriptors = _per_image_descriptors(messages)
    names = ["D50_length (mm)", "D50_aspect", "fines_fraction", "CV_length"]
    for col, name in enumerate(names):
        vals = descriptors[:, col]
        rng_str = f"[{vals.min():.3f}, {vals.max():.3f}]"
        print(f"  {name:18s}: mean={vals.mean():.3f}  "
              f"sd={vals.std():.3f}  range={rng_str}")
        assert vals.std() > 0, f"{name} should vary"
    print("  All descriptors vary. OK.")

    print("\n=== Test 3: coverage comparison vs standard generator ===")
    # Standard generator with the same n_images and a healthy
    # variation_scale, for comparison.
    rng_std = np.random.default_rng(seed=42)
    standard_messages = generate_dataset(
        dataset_params=DatasetGenerationParams(
            n_images=50, target_n_fibres=500, variation_scale=0.3
        ),
        rng=rng_std,
    )
    standard_desc = _per_image_descriptors(standard_messages)

    print("  Standard generator (variation_scale=0.3):")
    for col, name in enumerate(names):
        vals = standard_desc[:, col]
        rng_str = f"[{vals.min():.3f}, {vals.max():.3f}]"
        print(f"    {name:18s}: mean={vals.mean():.3f}  "
              f"sd={vals.std():.3f}  range={rng_str}")

    print("\n  Coverage ratio (diverse SD / standard SD):")
    for col, name in enumerate(names):
        ratio = descriptors[:, col].std() / max(standard_desc[:, col].std(), 1e-9)
        print(f"    {name:18s}: ratio = {ratio:.2f}x")

    # The headline check: fines_fraction should be much more
    # variable in the diverse dataset.
    fines_diverse = descriptors[:, 2]
    fines_standard = standard_desc[:, 2]
    print(f"\n  Diverse fines_fraction max:  {fines_diverse.max():.4f}")
    print(f"  Standard fines_fraction max: {fines_standard.max():.4f}")
    assert fines_diverse.max() > 5 * max(fines_standard.max(), 1e-4), \
        "Diverse generator should produce meaningfully higher fines"
    print("  Diverse generator produces materially higher fines. OK.")

    print("\n=== Test 4: target variation ===")
    rng = np.random.default_rng(seed=42)
    messages_with_targets = generate_diverse_dataset(
        diverse_params=DiverseDatasetParams(n_images=200),
        target_params=TargetGenerationParams(enabled=True),
        rng=rng,
    )
    truths = np.array([
        m["synthetic_targets"]["synthetic_MOR_truth"]
        for m in messages_with_targets
    ])
    print(f"  synthetic_MOR_truth: mean={truths.mean():.2f} MPa, "
          f"sd={truths.std():.2f}")
    print(f"    range=[{truths.min():.2f}, {truths.max():.2f}] MPa")

    rng = np.random.default_rng(seed=42)
    standard_targets_msgs = generate_dataset(
        dataset_params=DatasetGenerationParams(
            n_images=200, target_n_fibres=500, variation_scale=0.3
        ),
        target_params=TargetGenerationParams(enabled=True),
        rng=rng,
    )
    std_truths = np.array([
        m["synthetic_targets"]["synthetic_MOR_truth"]
        for m in standard_targets_msgs
    ])
    print(f"\n  Standard generator (variation_scale=0.3): "
          f"mean={std_truths.mean():.2f} MPa, sd={std_truths.std():.2f}")
    print(f"    range=[{std_truths.min():.2f}, {std_truths.max():.2f}] MPa")

    print(f"\n  Target SD ratio (diverse/standard): "
          f"{truths.std() / std_truths.std():.2f}x")
    assert truths.std() > 1.5 * std_truths.std(), \
        "Diverse generator should produce wider target spread"
    print("  Diverse target spread exceeds standard. OK.")

    print("\n=== Test 5: reproducibility ===")
    rng_a = np.random.default_rng(seed=1234)
    rng_b = np.random.default_rng(seed=1234)
    msgs_a = generate_diverse_dataset(
        diverse_params=DiverseDatasetParams(n_images=20),
        target_params=TargetGenerationParams(enabled=True),
        rng=rng_a,
    )
    msgs_b = generate_diverse_dataset(
        diverse_params=DiverseDatasetParams(n_images=20),
        target_params=TargetGenerationParams(enabled=True),
        rng=rng_b,
    )
    assert msgs_a == msgs_b, "Identical seeds must produce identical datasets"
    print("  Same seed produces identical datasets. OK.")

    print("\n=== Test 6: LHS-vs-uniform sampling paths ===")
    # LHS path
    rng = np.random.default_rng(seed=7)
    msgs_lhs = generate_diverse_dataset(
        diverse_params=DiverseDatasetParams(
            n_images=50, use_latin_hypercube=True
        ),
        rng=rng,
    )
    desc_lhs = _per_image_descriptors(msgs_lhs)

    # Uniform path
    rng = np.random.default_rng(seed=7)
    msgs_uni = generate_diverse_dataset(
        diverse_params=DiverseDatasetParams(
            n_images=50, use_latin_hypercube=False
        ),
        rng=rng,
    )
    desc_uni = _per_image_descriptors(msgs_uni)

    print("  LHS path (D50_length sd):     "
          f"{desc_lhs[:, 0].std():.4f}")
    print("  Uniform path (D50_length sd): "
          f"{desc_uni[:, 0].std():.4f}")
    # Both should produce valid datasets; LHS typically gives
    # slightly tighter coverage statistics, but at n=50 the
    # difference is small. We just confirm both run cleanly.
    assert desc_lhs[:, 0].std() > 0
    assert desc_uni[:, 0].std() > 0
    print("  Both sampling paths produce valid datasets. OK.")

    print("\nAll diverse-dataset checks passed.")


if __name__ == "__main__":
    main()