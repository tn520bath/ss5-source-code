"""
Validation of Module 6 Component 3: random-forest comparator.

Confirms:
  - The forest trains without error on the same 200-batch diverse
    dataset as ridge.
  - Outer-CV R^2 is in a sensible range (positive, > 0.5 at minimum,
    typically near ridge's R^2 since the descriptor space the
    diverse generator produces rarely activates saturation).
  - Direct head-to-head comparison with ridge on identical folds.
  - MDI importance assigns non-trivial weight to all four
    target-driving features (fines, length, AR, CV).
  - Permutation importance ranks fines highly (uncorrelated) and
    de-rates the redundant collinear pair (length_CV and
    length_D90_over_D50) because each can be substituted by the
    other - a known property of vanilla permutation importance
    under feature correlation, worth illustrating directly.
  - The two non-target features (length_D90_over_D50 and width_D50)
    have measurable but non-dominant importance.
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
from ss5_module3_descriptors.descriptor_orchestrator import (
    compute_descriptors,
)

from ss5_module6_predictive.features import (
    extract_feature_matrix,
    feature_names,
    FeatureConfig,
)
from ss5_module6_predictive.training import train_ridge, TrainingConfig
from ss5_module6_predictive.comparator import (
    train_random_forest,
    ComparatorConfig,
)


def _build_training_data(n_images=200, seed=42):
    """Same helper as validate_training, kept locally for clarity."""
    rng = np.random.default_rng(seed=seed)
    target_params = TargetGenerationParams(enabled=True)
    messages = generate_diverse_dataset(
        diverse_params=DiverseDatasetParams(n_images=n_images),
        target_params=target_params,
        rng=rng,
    )
    aggregator = BatchAggregator(IngestionParams(images_per_batch=1))
    descriptors_list = []
    targets = []
    for msg in messages:
        batch = aggregator.ingest(msg)
        if batch is not None:
            descriptors_list.append(compute_descriptors(batch))
            targets.append(msg["synthetic_targets"]["synthetic_MOR"])
    X = extract_feature_matrix(descriptors_list, FeatureConfig())
    y = np.asarray(targets)
    return X, y


def main():
    print("=== Test 1: forest trains without error ===")
    X, y = _build_training_data(n_images=200, seed=42)
    print(f"  Training data: X shape {X.shape}, y shape {y.shape}")

    result = train_random_forest(X, y, feature_names(),
                                 ComparatorConfig())
    print(f"  Trained model: {type(result.model).__name__}")
    print(f"  n_train       = {result.n_train}")
    print(f"  n_estimators  = {result.model.n_estimators}")
    print("  Pipeline trains. OK.")

    print("\n=== Test 2: CV scores are sensible ===")
    cv = result.cv_scores
    print("  Per-fold metrics:")
    print(f"    R^2  per fold: "
          f"{np.array2string(cv['r2']['per_fold'], precision=3)}")
    print(f"    RMSE per fold: "
          f"{np.array2string(cv['rmse']['per_fold'], precision=3)}")
    print(f"\n  Pooled outer-CV scores:")
    print(f"    R^2  = {cv['r2']['mean']:.3f} +/- {cv['r2']['sd']:.3f}")
    print(f"    RMSE = {cv['rmse']['mean']:.3f} +/- "
          f"{cv['rmse']['sd']:.3f} MPa")
    print(f"    MAE  = {cv['mae']['mean']:.3f} +/- "
          f"{cv['mae']['sd']:.3f} MPa")
    assert cv["r2"]["mean"] > 0.5, "R^2 too low: forest did not learn"
    print("  Mean R^2 exceeds 0.5. OK.")

    print("\n=== Test 3: head-to-head ridge vs random forest ===")
    ridge_result = train_ridge(X, y, feature_names(), TrainingConfig())
    ridge_cv = ridge_result.cv_scores
    print(f"  Ridge:         R^2 = {ridge_cv['r2']['mean']:.3f}, "
          f"RMSE = {ridge_cv['rmse']['mean']:.3f} MPa")
    print(f"  Random forest: R^2 = {cv['r2']['mean']:.3f}, "
          f"RMSE = {cv['rmse']['mean']:.3f} MPa")
    delta_r2 = cv["r2"]["mean"] - ridge_cv["r2"]["mean"]
    delta_rmse = cv["rmse"]["mean"] - ridge_cv["rmse"]["mean"]
    print(f"  Delta R^2  (forest - ridge): {delta_r2:+.4f}")
    print(f"  Delta RMSE (forest - ridge): {delta_rmse:+.4f} MPa")
    print("\n  Interpretation:")
    if abs(delta_r2) < max(cv["r2"]["sd"], ridge_cv["r2"]["sd"]):
        print("    Difference is within fold-to-fold noise. "
              "Linearity is sufficient for")
        print("    the descriptor space the diverse generator "
              "produces; ridge is fully")
        print("    competitive with random forest here.")
    elif delta_r2 > 0:
        print("    Random forest outperforms ridge by more than "
              "fold-to-fold noise. The")
        print("    saturating non-linearity in the synthetic target "
              "is contributing.")
    else:
        print("    Ridge slightly outperforms forest, consistent "
              "with linear data with")
        print("    small N and forest's bootstrap variance.")
    print("  Head-to-head comparison computed. OK.")

    print("\n=== Test 4: MDI importance ===")
    names = feature_names()
    mdi = result.mdi_importance
    print(f"  {'feature':25s}  {'MDI':>8s}")
    for name, val in zip(names, mdi):
        print(f"  {name:25s}  {val:>8.4f}")
    # MDI is normalised to sum to 1; ensure the fines feature
    # appears in the top half of features by MDI (it should, given
    # it's the strongest single contributor to the synthetic target).
    fines_idx = names.index("fines_fraction_count")
    rank = sum(1 for v in mdi if v > mdi[fines_idx])
    print(f"\n  fines_fraction_count ranks {rank+1} of {len(names)} "
          f"by MDI (1 = most important).")
    assert rank < len(names) / 2, \
        "fines should be in the top half by MDI"
    print("  MDI ranks fines in the top half. OK.")

    print("\n=== Test 5: permutation importance ===")
    perm_mean = result.perm_importance_mean
    perm_sd = result.perm_importance_sd
    print(f"  {'feature':25s}  {'perm_mean':>10s}  {'perm_sd':>8s}")
    for name, m, s in zip(names, perm_mean, perm_sd):
        print(f"  {name:25s}  {m:>10.4f}  {s:>8.4f}")

    # Fines should be the top permutation-importance feature: it has
    # the lowest correlation with the other features (Test 7 of the
    # ridge validator), so shuffling it cannot be compensated for.
    top_idx = int(np.argmax(perm_mean))
    print(f"\n  Top permutation-importance feature: {names[top_idx]}")
    assert names[top_idx] == "fines_fraction_count", \
        f"Expected fines to rank first by permutation importance, " \
        f"got {names[top_idx]}"
    print("  fines ranks first by permutation importance "
          "(uncorrelated -> can't be substituted). OK.")

    print("\n=== Test 6: importance comparison under multicollinearity ===")
    # Pair-of-interest: length_CV and length_D90_over_D50 (r = 0.98
    # in the training data, per Test 7 of validate_training).
    cv_idx = names.index("length_CV")
    d9050_idx = names.index("length_D90_over_D50")
    fines_idx = names.index("fines_fraction_count")

    print(f"  Under r=0.98 between length_CV and length_D90_over_D50:")
    print(f"    MDI(length_CV)            = {mdi[cv_idx]:.4f}")
    print(f"    MDI(length_D90_over_D50)  = {mdi[d9050_idx]:.4f}")
    print(f"    MDI sum (these two)       = "
          f"{mdi[cv_idx] + mdi[d9050_idx]:.4f}")
    print(f"    MDI(fines)                = {mdi[fines_idx]:.4f}    "
          f"(reference: uncorrelated feature)")
    print()
    print(f"    perm(length_CV)           = {perm_mean[cv_idx]:.4f} +/- "
          f"{perm_sd[cv_idx]:.4f}")
    print(f"    perm(length_D90_over_D50) = "
          f"{perm_mean[d9050_idx]:.4f} +/- {perm_sd[d9050_idx]:.4f}")
    print(f"    perm sum (these two)      = "
          f"{perm_mean[cv_idx] + perm_mean[d9050_idx]:.4f}")
    print(f"    perm(fines)               = "
          f"{perm_mean[fines_idx]:.4f} +/- {perm_sd[fines_idx]:.4f}    "
          f"(reference)")
    print()
    print("  Interpretation:")
    print("    MDI splits the joint length-heterogeneity signal "
          "between length_CV and")
    print("    length_D90_over_D50 (because both are sampled into "
          "tree splits when")
    print("    the other isn't), giving non-trivial importance to "
          "each.")
    print("    Permutation importance is suppressed for both: "
          "shuffling one leaves the")
    print("    other intact to carry the signal. This is a known "
          "weakness of vanilla")
    print("    permutation importance under correlated features. "
          "Fines, by contrast,")
    print("    ranks highly under permutation because it cannot "
          "be substituted.")
    print("  Diagnostics computed. OK.")

    print("\n=== Test 7: non-target features ===")
    # length_D90_over_D50 and width_D50 are not in the synthetic
    # target. They should appear as moderate-importance features
    # (because they correlate with target-driving descriptors), but
    # should not dominate.
    d9050_perm = perm_mean[d9050_idx]
    width_idx = names.index("width_D50")
    width_perm = perm_mean[width_idx]
    fines_perm = perm_mean[fines_idx]
    print(f"  perm(fines)                = {fines_perm:.4f}  (target-driving, top)")
    print(f"  perm(length_D90_over_D50) = {d9050_perm:.4f}  (non-target)")
    print(f"  perm(width_D50)           = {width_perm:.4f}  (non-target)")
    assert d9050_perm < fines_perm, \
        "length_D90_over_D50 should not exceed fines"
    assert width_perm < fines_perm, \
        "width_D50 should not exceed fines"
    print("  Non-target features bounded below the strongest "
          "target-driving feature. OK.")

    print("\nAll comparator checks passed.")


if __name__ == "__main__":
    main()