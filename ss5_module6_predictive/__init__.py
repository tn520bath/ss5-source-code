"""
Module 6: Predictive Modelling (ML-Ready Framework).

Trains a regularised regression model on Module 3 batch descriptors
to predict a synthetic MOR-like target produced by Module 1. The
package is intentionally separate from ss5_module2_7_pipeline (the runtime
data-flow handlers): Module 6 is a model-training and prediction
concern that operates at a different layer.

Public API:

    fit_module6(X, y, feature_names, ...)
        Training-time entry point. Returns a Module6Result holding
        the fitted ridge model, the random-forest comparator, the
        held-out CV predictions, and all diagnostic information.

    module6_predict(result, x_point, x_sigma, ...)
        Runtime entry point. Returns a BatchPrediction with 95%
        Monte Carlo prediction interval.

    extract_features, extract_feature_matrix, feature_names
        Translate Module 3 BatchDescriptors into a feature vector.

    FeatureConfig, TrainingConfig, ComparatorConfig, PredictionConfig
        Configuration dataclasses, all re-exported from predictive_config.py.

Files:

    features.py             : feature extraction
    training.py             : ridge regression with nested CV
    comparator.py           : random-forest comparator
    validation.py           : CV-fold reporting helpers
    prediction.py           : Monte Carlo prediction intervals
    orchestrator.py         : fit_module6 and module6_predict entry points
    predictive_config.py    : re-exports of all configuration dataclasses
"""

from .features import (
    extract_features,
    extract_feature_matrix,
    feature_names,
    FeatureConfig,
)
from .training import (
    train_ridge,
    TrainingConfig,
    TrainingResult,
)
from .comparator import (
    train_random_forest,
    ComparatorConfig,
    ComparatorResult,
)
from .validation import (
    cv_holdout_predictions,
    CVPredictions,
)
from .prediction import (
    predict_with_uncertainty,
    predict_batch_with_uncertainty,
    PredictionConfig,
    BatchPrediction,
)
from .orchestrator import (
    fit_module6,
    module6_predict,
    Module6Result,
)

__all__ = [
    # Top-level entry points
    "fit_module6", "module6_predict", "Module6Result",
    # Feature extraction
    "extract_features", "extract_feature_matrix", "feature_names",
    "FeatureConfig",
    # Training
    "train_ridge", "TrainingConfig", "TrainingResult",
    # Comparator
    "train_random_forest", "ComparatorConfig", "ComparatorResult",
    # Validation helpers
    "cv_holdout_predictions", "CVPredictions",
    # Prediction
    "predict_with_uncertainty", "predict_batch_with_uncertainty",
    "PredictionConfig", "BatchPrediction",
]