"""
Visual + numerical validation that the width sampler produces
plausible fibre width distributions.
"""

import numpy as np
import matplotlib.pyplot as plt

from ss5_module1_synthetic_generator.config import WidthDistributionParams
from ss5_module1_synthetic_generator.distributions import sample_widths


def main():
    rng = np.random.default_rng(seed=42)

    params = WidthDistributionParams()
    print(f"Distribution: lognormal (width)")
    print(f"mu = {params.mu}, sigma = {params.sigma}")
    print(f"Truncation: [{params.min_width_mm}, {params.max_width_mm}] mm")

    n = 5000
    widths = sample_widths(n, params, rng)

    print(f"\nSummary statistics for n = {n}:")
    print(f"  Mean   : {np.mean(widths):.4f} mm")
    print(f"  Median : {np.median(widths):.4f} mm")
    print(f"  Std    : {np.std(widths):.4f} mm")
    print(f"  CV     : {np.std(widths) / np.mean(widths) * 100:.2f} %")
    print(f"  D10    : {np.percentile(widths, 10):.4f} mm")
    print(f"  D50    : {np.percentile(widths, 50):.4f} mm")
    print(f"  D90    : {np.percentile(widths, 90):.4f} mm")
    print(f"  Min    : {np.min(widths):.4f} mm")
    print(f"  Max    : {np.max(widths):.4f} mm")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale histogram
    axes[0].hist(widths, bins=60, density=True, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Width (mm)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Lognormal width distribution (n = {n})")
    axes[0].axvline(np.median(widths), color="red", linestyle="--", label="Median")
    axes[0].legend()

    # Log-transformed: should be approximately Gaussian
    log_widths = np.log(widths)
    axes[1].hist(log_widths, bins=60, density=True, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("ln(Width)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Log-transformed (should be Gaussian)")

    plt.tight_layout()
    plt.savefig("width_distribution_validation.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()