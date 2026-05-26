"""
Configuration parameters for Module 4 (Uncertainty Quantification).

Centralises every uncertainty source's magnitude, the Monte Carlo
controls, and the reproducibility seed. All thresholds and default
values live here so they can be inspected, overridden, and reported
in the audit trail without hunting through the implementation files.

Source A (calibration) is parameterised by sigma_cal_relative; a
sensitivity sweep is provided through sigma_cal_sweep.

Source B (measurement noise) by default sources its sigma values
from the synthetic generator's MeasurementNoiseParams, ensuring that
the noise magnitude assumed in the uncertainty budget is the same
magnitude that drives the Monte Carlo perturbations and the
synthetic-data validation experiments. Override values can be
supplied via sigma_length_mm and sigma_width_mm.

Source D (reconstruction bias) inflates per-fibre noise on
high-reconstruction fibres by reconstruction_inflation_factor.

Sources F (finite-N sampling) and G (modelling) carry no parameters;
both are handled by bootstrap inside the Monte Carlo loop.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Top-level configuration for Module 4.
    """
    # === Source A: Calibration ===
    sigma_cal_relative: float = 0.01
    sigma_cal_sweep: tuple = (0.005, 0.01, 0.02)

    # === Source B: Measurement noise ===
    use_module9_noise_params: bool = True
    sigma_length_mm: float | None = None
    sigma_width_mm: float | None = None

    # === Source D: Reconstruction bias ===
    reconstruction_inflation_factor: float = 2.0

    # === Monte Carlo controls (used by Component 2) ===
    n_mc_iterations: int = 1000
    confidence_level: float = 0.95
    rng_seed: int | None = 42

    def resolved_noise_params(self) -> tuple[float, float]:
        """
        Resolve the effective measurement-noise sigma values.

        Resolution order:
          1. If sigma_length_mm / sigma_width_mm are supplied
             explicitly, use those.
          2. Otherwise, if use_module1_noise_params is True, source
             from ss5_module1_synthetic_generator.config.MeasurementNoiseParams.
          3. Otherwise, raise ValueError.

        Returns
        -------
        tuple[float, float]
            (sigma_length_mm, sigma_width_mm) in mm.
        """
        # Length
        if self.sigma_length_mm is not None:
            sigma_L = float(self.sigma_length_mm)
        elif self.use_module9_noise_params:
            from ss5_module1_synthetic_generator.config import (
                MeasurementNoiseParams,
            )
            sigma_L = float(MeasurementNoiseParams().sigma_length_mm)
        else:
            raise ValueError(
                "sigma_length_mm is None and "
                "use_module9_noise_params is False; cannot resolve "
                "effective length-noise magnitude."
            )

        # Width
        if self.sigma_width_mm is not None:
            sigma_w = float(self.sigma_width_mm)
        elif self.use_module9_noise_params:
            from ss5_module1_synthetic_generator.config import (
                MeasurementNoiseParams,
            )
            sigma_w = float(MeasurementNoiseParams().sigma_width_mm)
        else:
            raise ValueError(
                "sigma_width_mm is None and "
                "use_module9_noise_params is False; cannot resolve "
                "effective width-noise magnitude."
            )

        return sigma_L, sigma_w