"""
Component 1 of Module 6: Feature engineering.

Translates a Module 3 BatchDescriptors object into a flat feature
vector suitable for the ridge-regression and random-forest models
that follow.

The default feature set is six descriptors, motivated by the Module 5
evidence base and the structural form of the synthetic target.
The first four features map directly
onto the four target-driving terms: length D50, aspect-ratio D50,
fines fraction, and length CV. The remaining two (length D90/D50
and width D50) are deliberately included as descriptors that the
target does not depend on, so that ridge's regularisation can
demonstrate near-zero coefficient assignment to genuinely
non-contributing features.

Provides:

    extract_features(descriptors, config)
        Returns a 1D NumPy array of feature values in a stable order.

    feature_names(config)
        Returns the corresponding list of feature names, same order.

    FeatureConfig
        Dataclass holding the active feature list, used by both
        functions to ensure consistent ordering and selection.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Default ordering. Defined at module scope so feature_names and
# extract_features cannot drift apart.
_DEFAULT_FEATURES = (
    "length_D50",            # length saturation term in synthetic target
    "aspect_D50",            # AR saturation term
    "fines_fraction_count",  # fines penalty term
    "length_CV",             # heterogeneity penalty term
    "length_D90_over_D50",   # upper-tail proxy (non-target feature)
    "width_D50",             # width central tendency (non-target feature)
)


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for feature extraction.

    The default feature set is the six-feature reduced set described
    above. Custom subsets or supersets can be specified by passing a
    different tuple of names; valid names are any of:

        length_mean, length_D10, length_D50, length_D90,
        length_D90_over_D50, length_CV, length_skew,
        width_mean, width_D10, width_D50, width_D90, width_CV,
        aspect_mean, aspect_D50, aspect_CV,
        fines_fraction_count, fines_fraction_mass,
        length_mass_weighted_D50, length_mass_weighted_mean,

    The function extract_features will raise KeyError for unknown
    names, surfacing typos at call time rather than producing a
    silently-wrong feature vector.
    """
    features: tuple = field(default_factory=lambda: _DEFAULT_FEATURES)


# -------------------------------------------------------------------
# Feature extractors
# -------------------------------------------------------------------
# Each entry maps a feature name to a function that pulls the value
# out of the BatchDescriptors object. Adding a new feature only
# requires adding one line here; the rest of the module finds it
# through the registry.
# -------------------------------------------------------------------

def _ratio(numerator, denominator):
    """Safe ratio. Returns NaN if denominator is zero or non-finite."""
    if denominator is None or denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return numerator / denominator


_FEATURE_REGISTRY = {
    # Length descriptors
    "length_mean":         lambda b: b.descriptors_length["mean"],
    "length_D10":          lambda b: b.descriptors_length["D10"],
    "length_D50":          lambda b: b.descriptors_length["D50"],
    "length_D90":          lambda b: b.descriptors_length["D90"],
    "length_D90_over_D50": lambda b: _ratio(
        b.descriptors_length["D90"], b.descriptors_length["D50"]
    ),
    "length_CV":           lambda b: b.descriptors_length["cv"],
    "length_skew":         lambda b: b.descriptors_length["skew"],

    # Width descriptors
    "width_mean":          lambda b: b.descriptors_width["mean"],
    "width_D10":           lambda b: b.descriptors_width["D10"],
    "width_D50":           lambda b: b.descriptors_width["D50"],
    "width_D90":           lambda b: b.descriptors_width["D90"],
    "width_CV":            lambda b: b.descriptors_width["cv"],

    # Aspect-ratio descriptors
    "aspect_mean":         lambda b: b.descriptors_aspect_ratio["mean"],
    "aspect_D50":          lambda b: b.descriptors_aspect_ratio["D50"],
    "aspect_CV":           lambda b: b.descriptors_aspect_ratio["cv"],

    # Fines
    "fines_fraction_count": lambda b: b.fines["fines_fraction_count"],
    "fines_fraction_mass":  lambda b: b.fines["fines_fraction_mass"],

    # Mass-weighted length
    "length_mass_weighted_D50":  lambda b: b.descriptors_length_mass_weighted["D50"],
    "length_mass_weighted_mean": lambda b: b.descriptors_length_mass_weighted["mean"],
}


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def feature_names(config: FeatureConfig | None = None) -> list:
    """
    Return the active feature names in the order they will appear
    in extract_features's output array.
    """
    if config is None:
        config = FeatureConfig()
    return list(config.features)


def extract_features(
    descriptors: Any,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """
    Extract a feature vector from a Module 3 BatchDescriptors object.

    Parameters
    ----------
    descriptors : BatchDescriptors
        The output of compute_descriptors. Only the fields listed in
        config.features are accessed; missing or NaN fields produce
        a NaN in the output (model code is responsible for handling
        these or rejecting the example upstream).
    config : FeatureConfig, optional
        Feature configuration. If None, the default 6-feature reduced
        set is used.

    Returns
    -------
    np.ndarray
        A 1D float array of length len(config.features), with values
        in the order specified by config.features.

    Raises
    ------
    KeyError
        If any name in config.features is not in the feature registry.
    """
    if config is None:
        config = FeatureConfig()

    values = []
    for name in config.features:
        if name not in _FEATURE_REGISTRY:
            raise KeyError(
                f"Unknown feature name '{name}'. Valid names are: "
                f"{sorted(_FEATURE_REGISTRY.keys())}"
            )
        try:
            value = _FEATURE_REGISTRY[name](descriptors)
        except (KeyError, AttributeError, TypeError):
            value = float("nan")
        values.append(float(value) if value is not None else float("nan"))

    return np.asarray(values, dtype=float)


def extract_feature_matrix(
    descriptors_list: list,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """
    Vectorised convenience: extract features for many batches.

    Parameters
    ----------
    descriptors_list : list of BatchDescriptors
        One entry per batch.
    config : FeatureConfig, optional
        Feature configuration.

    Returns
    -------
    np.ndarray
        2D array of shape (len(descriptors_list), n_features).
    """
    if not descriptors_list:
        return np.zeros((0, len(feature_names(config))), dtype=float)
    rows = [extract_features(d, config) for d in descriptors_list]
    return np.vstack(rows)