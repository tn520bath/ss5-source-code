"""
Visual + numerical validation of the measurement-noise model.

Demonstrates that the noise model:
  - Preserves the underlying length and width distributions (mean
    and shape).
  - Adds Gaussian-distributed deviations of the configured magnitude.
  - Produces fractional uncertainties consistent with the
    SYS-PERF-02 (+/- 5%) and SYS-PERF-03 (+/- 15%) tolerances
    for length and width respectively.
"""

import numpy as np
import matplotlib.pyplot as plt

from ss5_module1_synthetic_generator.config import (
    LengthDistributionParams,
    WidthDistributionParams,
    MeasurementNoiseParams,
)
from ss5_module1_synthetic_generator.distributions import sample_lengths, sample_widths
from ss5_module1_synthetic_generator.noise_models import apply_measurement_noise


def main():
    rng = np.random.default_rng(seed=42)

    length_params = LengthDistributionParams()
    width_params = WidthDistributionParams()
    noise_params = MeasurementNoiseParams()

    print(f"Noise parameters:")
    print(f"  sigma_length = {noise_params.sigma_length_mm} mm")
    print(f"  sigma_width  = {noise_params.sigma_width_mm} mm")

    n = 5000
    true_lengths = sample_lengths(n, length_params, rng)
    true_widths = sample_widths(n, width_params, rng)

    measured_lengths, measured_widths = apply_measurement_noise(
        true_lengths, true_widths, noise_params, rng
    )

    # Per-fibre measurement errors
    length_errors = measured_lengths - true_lengths
    width_errors = measured_widths - true_widths

    # Per-fibre fractional errors (relative to true value)
    length_frac_errors = length_errors / true_lengths
    width_frac_errors = width_errors / true_widths

    print(f"\nLength error statistics (n = {n}):")
    print(f"  Mean error          : {np.mean(length_errors):.5f} mm")
    print(f"  Std of error        : {np.std(length_errors):.5f} mm")
    print(f"  Mean fractional err : {np.mean(length_frac_errors) * 100:.3f} %")
    print(f"  Std fractional err  : {np.std(length_frac_errors) * 100:.3f} %")
    print(f"  95% within          : +/- {np.percentile(np.abs(length_frac_errors), 95) * 100:.2f} %")
    print(f"  SYS-PERF-02 spec    : +/- 5 %")

    print(f"\nWidth error statistics (n = {n}):")
    print(f"  Mean error          : {np.mean(width_errors):.6f} mm")
    print(f"  Std of error        : {np.std(width_errors):.6f} mm")
    print(f"  Mean fractional err : {np.mean(width_frac_errors) * 100:.3f} %")
    print(f"  Std fractional err  : {np.std(width_frac_errors) * 100:.3f} %")
    print(f"  95% within          : +/- {np.percentile(np.abs(width_frac_errors), 95) * 100:.2f} %")
    print(f"  SYS-PERF-03 spec    : +/- 15 %")

    # Plots: comparison of true and measured distributions, plus error histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top-left: true vs measured lengths
    axes[0, 0].hist(true_lengths, bins=50, alpha=0.5, label="True", density=True)
    axes[0, 0].hist(measured_lengths, bins=50, alpha=0.5, label="Measured", density=True)
    axes[0, 0].set_xlabel("Length (mm)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Length: true vs measured")
    axes[0, 0].legend()

    # Top-right: true vs measured widths
    axes[0, 1].hist(true_widths, bins=50, alpha=0.5, label="True", density=True)
    axes[0, 1].hist(measured_widths, bins=50, alpha=0.5, label="Measured", density=True)
    axes[0, 1].set_xlabel("Width (mm)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Width: true vs measured")
    axes[0, 1].legend()

    # Bottom-left: length error distribution (should be Gaussian, centred at zero)
    axes[1, 0].hist(length_errors, bins=50, density=True, edgecolor="black")
    axes[1, 0].set_xlabel("Length error (mm)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Length measurement error (should be Gaussian)")
    axes[1, 0].axvline(0, color="red", linestyle="--", label="Zero")
    axes[1, 0].legend()

    # Bottom-right: width error distribution (should be Gaussian, centred at zero)
    axes[1, 1].hist(width_errors, bins=50, density=True, edgecolor="black")
    axes[1, 1].set_xlabel("Width error (mm)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Width measurement error (should be Gaussian)")
    axes[1, 1].axvline(0, color="red", linestyle="--", label="Zero")
    axes[1, 1].legend()
    
    # Add tolerance bands for SYS-PERF specs
    length_tol = 0.05 * np.mean(true_lengths)  # +/- 5% of mean length
    axes[1, 0].axvline(-length_tol, color="orange", linestyle=":", label="SYS-PERF-02 (+/- 5%)")
    axes[1, 0].axvline(length_tol, color="orange", linestyle=":")

    width_tol = 0.15 * np.mean(true_widths)  # +/- 15% of mean width
    axes[1, 1].axvline(-width_tol, color="orange", linestyle=":", label="SYS-PERF-03 (+/- 15%)")
    axes[1, 1].axvline(width_tol, color="orange", linestyle=":")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()