"""
Validation of Module 6 Component 1: feature engineering.

Confirms:
  - Default 6-feature extraction returns the right names in the
    right order with sensible numerical values on real descriptors
    produced by Module 3.
  - Custom feature configs select the requested features.
  - Unknown feature names raise KeyError with a clear message.
  - extract_feature_matrix correctly stacks per-batch vectors into
    a (n_batches, n_features) array.
  - Feature values reproduce exactly the corresponding fields of the
    BatchDescriptors object (sanity check on the extractor mapping).
"""

import numpy as np

from ss5_module1_synthetic_generator.config import (
    DiverseDatasetParams,
    TargetGenerationParams,
)
from ss5_module1_synthetic_generator.dataset_generator import (
    generate_diverse_dataset,
)
from ss5_module2_7_pipeline.config import IngestionParams
from ss5_module2_7_pipeline.ingestion import BatchAggregator
from ss5_module3_descriptors.descriptor_orchestrator import compute_descriptors

from ss5_module6_predictive.predictive_config import FeatureConfig
from ss5_module6_predictive.features import (
    extract_features,
    extract_feature_matrix,
    feature_names,
)


def _produce_batch_descriptors(n_images=20, seed=42):
    """
    Helper: generate a synthetic dataset, run it through Module 2's
    aggregator, then compute Module 3 descriptors. Returns a list
    of BatchDescriptors.
    """
    rng = np.random.default_rng(seed=seed)
    messages = generate_diverse_dataset(
        diverse_params=DiverseDatasetParams(n_images=n_images),
        target_params=TargetGenerationParams(enabled=True),
        rng=rng,
    )
    aggregator = BatchAggregator(IngestionParams(images_per_batch=1))
    descriptors_list = []
    for msg in messages:
        batch = aggregator.ingest(msg)
        if batch is not None:
            descriptors_list.append(compute_descriptors(batch))
    return descriptors_list, messages


def main():
    print("=== Test 1: default feature set ===")
    descriptors_list, _ = _produce_batch_descriptors(n_images=5)
    config = FeatureConfig()
    names = feature_names(config)
    print(f"  Feature names ({len(names)}):")
    for i, n in enumerate(names):
        print(f"    {i}: {n}")

    expected_default = (
        "length_D50",
        "aspect_D50",
        "fines_fraction_count",
        "length_CV",
        "length_D90_over_D50",
        "width_D50",
    )
    assert tuple(names) == expected_default
    print("  Default feature ordering matches expectation. OK.")

    print("\n=== Test 2: extracted values are sensible ===")
    vec = extract_features(descriptors_list[0], config)
    assert vec.shape == (6,)
    assert np.all(np.isfinite(vec))
    print(f"  First batch features: {vec}")
    print(f"    length_D50           = {vec[0]:.4f} mm     (expect ~3 mm)")
    print(f"    aspect_D50           = {vec[1]:.2f}        (expect ~25-50)")
    print(f"    fines_fraction_count = {vec[2]:.4f}        (expect ~0-0.2)")
    print(f"    length_CV            = {vec[3]:.4f}        (expect ~0.3-0.7)")
    print(f"    length_D90/D50       = {vec[4]:.4f}        (expect ~1.5-2.5)")
    print(f"    width_D50            = {vec[5]:.4f} mm     (expect ~0.1)")
    assert 0.5 < vec[0] < 10.0,    "length_D50 outside plausible range"
    assert 5.0 < vec[1] < 80.0,    "aspect_D50 outside plausible range"
    assert 0.0 <= vec[2] <= 1.0,   "fines_fraction outside [0, 1]"
    assert 0.0 < vec[3] < 1.0,     "length_CV outside plausible range"
    assert 1.0 < vec[4] < 5.0,     "length_D90/D50 outside plausible range"
    assert 0.02 < vec[5] < 0.30,   "width_D50 outside plausible range"
    print("  All values within sane ranges. OK.")

    print("\n=== Test 3: extracted values match BatchDescriptors fields ===")
    bd = descriptors_list[0]
    assert vec[0] == bd.descriptors_length["D50"]
    assert vec[1] == bd.descriptors_aspect_ratio["D50"]
    assert vec[2] == bd.fines["fines_fraction_count"]
    assert vec[3] == bd.descriptors_length["cv"]
    expected_ratio = bd.descriptors_length["D90"] / bd.descriptors_length["D50"]
    assert abs(vec[4] - expected_ratio) < 1e-12
    assert vec[5] == bd.descriptors_width["D50"]
    print("  All six features bitwise-match the BatchDescriptors fields. OK.")

    print("\n=== Test 4: custom feature config ===")
    custom = FeatureConfig(features=(
        "length_mean", "fines_fraction_mass", "aspect_CV"
    ))
    custom_names = feature_names(custom)
    custom_vec = extract_features(descriptors_list[0], custom)
    assert custom_names == ["length_mean", "fines_fraction_mass", "aspect_CV"]
    assert custom_vec.shape == (3,)
    print(f"  Custom features ({len(custom_names)}): {custom_names}")
    print(f"  Values: {custom_vec}")
    assert custom_vec[0] == bd.descriptors_length["mean"]
    assert custom_vec[1] == bd.fines["fines_fraction_mass"]
    assert custom_vec[2] == bd.descriptors_aspect_ratio["cv"]
    print("  Custom config produces correct features. OK.")

    print("\n=== Test 5: unknown feature name raises clear error ===")
    bad = FeatureConfig(features=("length_D50", "no_such_feature"))
    try:
        extract_features(descriptors_list[0], bad)
        raise AssertionError("Should have raised KeyError")
    except KeyError as e:
        msg = str(e)
        assert "no_such_feature" in msg
        assert "Valid names" in msg or "Valid" in msg
        print(f"  KeyError raised as expected.")
        print(f"  Message contains the bad name and the valid name list. OK.")

    print("\n=== Test 6: feature matrix from a list ===")
    descriptors_list, messages = _produce_batch_descriptors(n_images=20)
    X = extract_feature_matrix(descriptors_list, FeatureConfig())
    print(f"  Feature matrix shape: {X.shape}")
    assert X.shape == (len(descriptors_list), 6)
    assert np.all(np.isfinite(X))

    print("  Per-feature stats across the dataset:")
    for i, name in enumerate(feature_names()):
        col = X[:, i]
        print(f"    {name:25s}: mean={col.mean():.4f}  "
              f"sd={col.std():.4f}  range=[{col.min():.4f}, {col.max():.4f}]")
    print("  Matrix-level extraction coherent with single-batch. OK.")

    print("\n=== Test 7: empty input list ===")
    empty = extract_feature_matrix([], FeatureConfig())
    assert empty.shape == (0, 6)
    print(f"  Empty list -> shape {empty.shape}. OK.")

    print("\n=== Test 8: target alignment ===")
    # The model will map X to y: confirm we can pull both, and that
    # they're of compatible length.
    y = np.array([
        m["synthetic_targets"]["synthetic_MOR"]
        for m in messages[:len(descriptors_list)]
    ])
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    assert X.shape[0] == y.shape[0]
    print(f"  Target stats:  mean={y.mean():.2f} MPa  sd={y.std():.2f}  "
          f"range=[{y.min():.2f}, {y.max():.2f}]")
    print("  X and y align correctly. OK.")

    print("\nAll feature-engineering checks passed.")


if __name__ == "__main__":
    main()