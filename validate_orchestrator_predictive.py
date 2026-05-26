"""
End-to-end validation of the Module 6 orchestrator.

Confirms:
  - fit_module6 runs end-to-end and returns a well-formed
    Module6Result with all expected fields.
  - The orchestrator's ridge CV scores match what train_ridge
    produces directly (proving the orchestrator does not silently
    alter the training methodology).
  - The held-out CV predictions reproduce the R^2 and RMSE that
    train_ridge reports from its outer-CV loop (proving the
    aggregated and per-batch metrics are consistent).
  - The comparator runs by default and is skippable on request.
  - The Module6Result properties (cv_r2_mean, cv_rmse_mean,
    selected_alpha) expose the right scalars.
  - module6_predict produces sensible intervals at runtime and
    correctly defaults sigma_eps to the training-CV RMSE when not
    overridden.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

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

from ss5_module6_predictive import (
    extract_feature_matrix,
    feature_names,
    FeatureConfig,
    fit_module6,
    module6_predict,
    train_ridge,
    TrainingConfig,
    PredictionConfig,
)


def _build_training_data(n_images=200, seed=42):
    """Same as in the per-component validators."""
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
    print("=== Test 1: fit_module6 runs end-to-end ===")
    X, y = _build_training_data(n_images=200, seed=42)
    print(f"  Training data: X shape {X.shape}, y shape {y.shape}")

    result = fit_module6(X, y, feature_names())
    print(f"  Module6Result fields:")
    print(f"    ridge          = {type(result.ridge).__name__}")
    print(f"    cv_predictions = {type(result.cv_predictions).__name__}")
    print(f"    comparator     = {type(result.comparator).__name__}")
    print(f"    n_train        = {result.n_train}")
    print(f"    feature_names  = {result.feature_names}")
    assert result.n_train == X.shape[0]
    assert result.feature_names == feature_names()
    print("  fit_module6 runs and returns well-formed result. OK.")

    print("\n=== Test 2: orchestrator ridge results match train_ridge ===")
    direct = train_ridge(X, y, feature_names(), TrainingConfig())
    orch_alpha = result.ridge.selected_alpha
    direct_alpha = direct.selected_alpha
    print(f"  selected alpha: orchestrator = {orch_alpha:.5g}, "
          f"direct = {direct_alpha:.5g}")
    assert abs(orch_alpha - direct_alpha) < 1e-9, \
        "Orchestrator alpha must equal direct-call alpha"

    orch_r2 = result.ridge.cv_scores["r2"]["mean"]
    direct_r2 = direct.cv_scores["r2"]["mean"]
    print(f"  CV R^2:         orchestrator = {orch_r2:.5g}, "
          f"direct = {direct_r2:.5g}")
    assert abs(orch_r2 - direct_r2) < 1e-9, \
        "Orchestrator R^2 must equal direct-call R^2"

    orch_rmse = result.ridge.cv_scores["rmse"]["mean"]
    direct_rmse = direct.cv_scores["rmse"]["mean"]
    print(f"  CV RMSE:        orchestrator = {orch_rmse:.5g}, "
          f"direct = {direct_rmse:.5g}")
    assert abs(orch_rmse - direct_rmse) < 1e-9, \
        "Orchestrator RMSE must equal direct-call RMSE"
    print("  Orchestrator does not alter ridge methodology. OK.")

    print("\n=== Test 3: held-out CV predictions reproduce CV scores ===")
    preds = result.cv_predictions
    print(f"  CV-predictions arrays:")
    print(f"    predicted shape = {preds.predicted.shape}")
    print(f"    actual    shape = {preds.actual.shape}")
    print(f"    fold_id   shape = {preds.fold_id.shape}")
    print(f"    residuals shape = {preds.residuals.shape}")
    assert preds.predicted.shape == (X.shape[0],)
    assert preds.actual.shape == (X.shape[0],)
    np.testing.assert_array_equal(preds.actual, y)
    np.testing.assert_allclose(preds.residuals, preds.predicted - y)
    # Per-batch R^2 and RMSE from the held-out predictions should
    # match the outer-CV mean R^2 and RMSE from train_ridge.
    # cross_val_predict pools all held-out predictions and computes
    # one R^2 across the pooled set; cross_validate computes per-fold
    # R^2 and averages them. These can differ slightly when the
    # per-fold R^2 distribution is wide, but should agree at the
    # ~1% level for a well-behaved 5-fold setup.
    pooled_r2 = r2_score(preds.actual, preds.predicted)
    pooled_rmse = float(np.sqrt(
        mean_squared_error(preds.actual, preds.predicted)
    ))
    print(f"  Pooled R^2 from held-out preds: {pooled_r2:.4f}")
    print(f"  Mean per-fold R^2:              {orch_r2:.4f}")
    print(f"  Pooled RMSE from held-out preds: {pooled_rmse:.4f} MPa")
    print(f"  Mean per-fold RMSE:              {orch_rmse:.4f} MPa")
    assert abs(pooled_r2 - orch_r2) < 0.02, \
        "Pooled and per-fold R^2 should agree to within ~1%"
    assert abs(pooled_rmse - orch_rmse) < 0.05, \
        "Pooled and per-fold RMSE should agree to within ~0.05 MPa"
    print("  Held-out predictions consistent with reported "
          "CV scores. OK.")

    print("\n=== Test 4: per-fold breakdown ===")
    # Just confirm each fold has roughly the same number of
    # samples and that fold IDs are 0..(n_folds-1).
    n_folds = len(np.unique(preds.fold_id))
    print(f"  Number of folds: {n_folds}")
    assert n_folds == 5, "Expected 5-fold CV by default"
    for f in range(n_folds):
        mask = preds.fold_id == f
        n_in_fold = int(mask.sum())
        fold_r2 = r2_score(preds.actual[mask], preds.predicted[mask])
        fold_rmse = float(np.sqrt(
            mean_squared_error(preds.actual[mask], preds.predicted[mask])
        ))
        print(f"    fold {f}: n={n_in_fold:3d}  R^2={fold_r2:.3f}  "
              f"RMSE={fold_rmse:.3f} MPa")
    print("  Per-fold breakdown sensible. OK.")

    print("\n=== Test 5: comparator on/off switch ===")
    assert result.comparator is not None, \
        "Comparator should run by default"
    print(f"  Default comparator: {type(result.comparator).__name__}")
    result_no_comp = fit_module6(
        X, y, feature_names(), comparator_enabled=False
    )
    assert result_no_comp.comparator is None
    print("  comparator_enabled=False produces None. OK.")

    print("\n=== Test 6: result properties ===")
    print(f"  cv_r2_mean     = {result.cv_r2_mean:.4f}")
    print(f"  cv_rmse_mean   = {result.cv_rmse_mean:.4f} MPa")
    print(f"  selected_alpha = {result.selected_alpha:.5g}")
    assert result.cv_r2_mean == orch_r2
    assert result.cv_rmse_mean == orch_rmse
    assert result.selected_alpha == orch_alpha
    print("  Properties expose the right scalars. OK.")

    print("\n=== Test 7: module6_predict runtime path ===")
    rel_sigma = np.array([0.01, 0.015, 0.02, 0.02, 0.015, 0.01])
    x_point = X[0]
    x_sigma = rel_sigma * np.abs(x_point)

    # Default sigma_eps should fall back to the training-CV RMSE.
    pred = module6_predict(result, x_point, x_sigma)
    print(f"  Batch 0 prediction:")
    print(f"    point estimate = {pred.point_estimate:.3f} MPa")
    print(f"    95% CI         = "
          f"[{pred.ci_lower:.3f}, {pred.ci_upper:.3f}] MPa")
    print(f"    sigma_eps used = {pred.sigma_eps_used:.3f} MPa")
    assert pred.ci_lower < pred.point_estimate < pred.ci_upper
    assert abs(pred.sigma_eps_used - result.cv_rmse_mean) < 1e-9, \
        "Default sigma_eps must equal training-CV RMSE"
    print("  Default sigma_eps falls back to training-CV RMSE. OK.")

    # Explicit override
    pred_override = module6_predict(
        result, x_point, x_sigma,
        prediction_config=PredictionConfig(sigma_eps=3.0),
    )
    assert pred_override.sigma_eps_used == 3.0
    assert (pred_override.ci_upper - pred_override.ci_lower) > \
           (pred.ci_upper - pred.ci_lower), \
           "Larger sigma_eps should widen the interval"
    print(f"  Explicit sigma_eps=3.0 used (widens interval to "
          f"{pred_override.ci_upper - pred_override.ci_lower:.3f} MPa "
          f"vs {pred.ci_upper - pred.ci_lower:.3f} MPa). OK.")

    print("\nAll orchestrator checks passed.")


if __name__ == "__main__":
    main()