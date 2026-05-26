"""
Visual + numerical validation of the quality-metadata model.

Confirms the two-stage reconstructed_fraction model produces:
  - The expected proportion of reconstructed vs non-reconstructed
    fibres.
  - A right-skewed Beta-distributed fraction among reconstructed
    fibres, with mean and shape matching the configured parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

from ss5_module1_synthetic_generator.config import QualityMetadataParams
from ss5_module1_synthetic_generator.quality_metadata import (
    sample_reconstructed_fractions,
    assign_batch_metadata,
)


def main():
    rng = np.random.default_rng(seed=42)

    params = QualityMetadataParams()
    print(f"Quality metadata parameters:")
    print(f"  p_reconstructed = {params.p_reconstructed}")
    print(f"  Beta(alpha={params.beta_alpha}, beta={params.beta_beta})")
    print(f"  pixel_size_mm  = {params.pixel_size_mm}")

    n = 5000
    fractions = sample_reconstructed_fractions(n, params, rng)

    n_zero = int(np.sum(fractions == 0))
    n_nonzero = n - n_zero
    nonzero_fractions = fractions[fractions > 0]

    print(f"\nReconstructed-fraction statistics (n = {n}):")
    print(f"  Total fibres                 : {n}")
    print(f"  With reconstruction          : {n_nonzero} "
          f"({100 * n_nonzero / n:.1f} %)")
    print(f"  Without reconstruction       : {n_zero} "
          f"({100 * n_zero / n:.1f} %)")
    print(f"  Expected reconstructed share : "
          f"{100 * params.p_reconstructed:.1f} %")
    print(f"\nAmong reconstructed fibres only:")
    print(f"  Mean fraction   : {np.mean(nonzero_fractions):.3f}")
    print(f"  Median fraction : {np.median(nonzero_fractions):.3f}")
    print(f"  Max fraction    : {np.max(nonzero_fractions):.3f}")
    expected_mean = params.beta_alpha / (params.beta_alpha + params.beta_beta)
    print(f"  Expected mean   : {expected_mean:.3f}")

    # Demonstrate batch-metadata assignment
    metadata = assign_batch_metadata(
        n_fibres=10, batch_id=0, image_id=0, params=params
    )
    print(f"\nExample batch metadata for n_fibres=10:")
    print(f"  mesh_ids       : {metadata['mesh_ids']}")
    print(f"  batch_id       : {metadata['batch_id']}")
    print(f"  image_id       : {metadata['image_id']}")
    print(f"  pixel_size_mm  : {metadata['pixel_size_mm']}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: full distribution including the spike at zero
    axes[0].hist(fractions, bins=50, density=False, edgecolor="black")
    axes[0].set_xlabel("reconstructed_fraction")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"All fibres (n = {n})\n"
        f"Spike at 0 represents the {100 * (1 - params.p_reconstructed):.0f}% "
        f"of fibres with no reconstruction"
    )

    # Right: only reconstructed fibres, showing the Beta shape
    axes[1].hist(
        nonzero_fractions, bins=40, density=True, alpha=0.7,
        edgecolor="black", label="Empirical"
    )

    # Overlay the theoretical Beta PDF for visual check
    from scipy import stats as scipy_stats
    x = np.linspace(0.001, 0.999, 200)
    theoretical_pdf = scipy_stats.beta.pdf(
        x, a=params.beta_alpha, b=params.beta_beta
    )
    axes[1].plot(x, theoretical_pdf, "r-", linewidth=2,
                 label=f"Beta({params.beta_alpha:.0f}, {params.beta_beta:.0f}) PDF")
    axes[1].set_xlabel("reconstructed_fraction")
    axes[1].set_ylabel("Density")
    axes[1].set_title(
        f"Reconstructed fibres only (n = {n_nonzero})"
    )
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()