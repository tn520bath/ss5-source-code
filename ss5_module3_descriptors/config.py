"""
Configuration parameters for Module 3 (Descriptor Computation &
Distribution Modelling).

All thresholds, defaults, and tunable values live here so they can be
inspected, overridden, and reported in the audit trail without hunting
through the implementation files.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptorConfig:
    """
    Top-level configuration for descriptor computation.
    """
    # Fines threshold (mm). Defaults to 1 mm, the lower bound of
    # SYS-PERF-01. Configurable for sensitivity analysis.
    fines_threshold_mm: float = 1.0

    # Warn if batch contains fewer than this many fibres. MLE fits and
    # descriptor estimates become unreliable below this size. Configurable.
    small_n_warning_threshold: int = 30

    # Percentile levels to compute. The standard particle-science trio.
    percentiles: tuple = (10.0, 50.0, 90.0)