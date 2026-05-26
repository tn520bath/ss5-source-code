"""
Visual + numerical validation that the length sampler produces
plausible fibre length distributions.
"""

import numpy as np
import matplotlib.pyplot as plt

from ss5_module1_synthetic_generator.config import LengthDistributionParams
from ss5_module1_synthetic_generator.distributions import sample_lengths


def main():
    # Reproducible random generator
    rng = np.random.default_rng(seed=42)

    # Default parameters
    params = LengthDistributionParams()
    print(f"Distribution: {params.distribution}")
    print(f"mu = {params.mu}, sigma = {params.sigma}")
    print(f"Truncation: [{params.min_length_mm}, {params.max_length_mm}] mm")

    # Generate a large batch
    n = 5000
    lengths = sample_lengths(n, params, rng)

    # Print summary statistics
    print(f"\nSummary statistics for n = {n}:")
    print(f"  Mean   : {np.mean(lengths):.3f} mm")
    print(f"  Median : {np.median(lengths):.3f} mm")
    print(f"  Std    : {np.std(lengths):.3f} mm")
    print(f"  D10    : {np.percentile(lengths, 10):.3f} mm")
    print(f"  D50    : {np.percentile(lengths, 50):.3f} mm")
    print(f"  D90    : {np.percentile(lengths, 90):.3f} mm")
    print(f"  Min    : {np.min(lengths):.3f} mm")
    print(f"  Max    : {np.max(lengths):.3f} mm")

    # Plot histogram with theoretical density overlay
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale histogram
    axes[0].hist(lengths, bins=60, density=True, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Length (mm)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Lognormal length distribution (n = {n})")
    axes[0].axvline(np.median(lengths), color="red", linestyle="--", label="Median")
    axes[0].legend()

    # Log scale to verify lognormal shape (should look approximately Gaussian)
    log_lengths = np.log(lengths)
    axes[1].hist(log_lengths, bins=60, density=True, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("ln(Length)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Log-transformed (should be Gaussian)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()