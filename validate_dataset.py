"""
Validation of multi-image dataset generation.

Generates three datasets covering the full range of variation_scale
behaviour and confirms that:

  - At variation_scale = 0, all images have identical n_fibres and
    statistically identical underlying distributions (up to sampling
    noise).
  - At variation_scale > 0, images vary in n_fibres and in
    per-image distribution statistics.
  - The reconstruction rate is stable across all images regardless
    of variation_scale (because reconstruction parameters are not
    varied).
  - The dataset is reproducible: identical seeds produce identical
    datasets.
"""

import numpy as np
import matplotlib.pyplot as plt

from ss5_module1_synthetic_generator.config import DatasetGenerationParams
from ss5_module1_synthetic_generator.dataset_generator import generate_dataset


def _summarise(messages: list[dict]) -> dict:
    """Compute per-image summary statistics across a dataset."""
    n_fibres_per_image = [len(m["char"]) for m in messages]
    mean_lengths = [
        np.mean([f["dimensions"]["length"] for f in m["char"]])
        for m in messages
    ]
    mean_widths = [
        np.mean([f["dimensions"]["width"] for f in m["char"]])
        for m in messages
    ]
    rec_rates = [
        np.mean([f["reconstructed_fraction"] > 0 for f in m["char"]])
        for m in messages
    ]
    return {
        "n_fibres": np.array(n_fibres_per_image),
        "mean_length": np.array(mean_lengths),
        "mean_width": np.array(mean_widths),
        "rec_rate": np.array(rec_rates),
    }


def _print_summary(label: str, summary: dict):
    print(f"\n--- {label} ---")
    print(f"n_fibres        : "
          f"mean={np.mean(summary['n_fibres']):.1f}, "
          f"std={np.std(summary['n_fibres']):.1f}, "
          f"range=[{np.min(summary['n_fibres'])}, "
          f"{np.max(summary['n_fibres'])}]")
    print(f"mean length (mm): "
          f"mean={np.mean(summary['mean_length']):.3f}, "
          f"std={np.std(summary['mean_length']):.3f}")
    print(f"mean width (mm) : "
          f"mean={np.mean(summary['mean_width']):.4f}, "
          f"std={np.std(summary['mean_width']):.4f}")
    print(f"rec rate (%)    : "
          f"mean={100*np.mean(summary['rec_rate']):.1f}, "
          f"std={100*np.std(summary['rec_rate']):.1f}")


def main():
    # Three datasets at different variation scales
    cases = [
        ("variation_scale = 0.0  (clean)", 0.0),
        ("variation_scale = 0.1  (mild)", 0.1),
        ("variation_scale = 0.5  (moderate)", 0.5),
    ]

    summaries = []
    for label, scale in cases:
        rng = np.random.default_rng(seed=42)
        params = DatasetGenerationParams(
            n_images=50, target_n_fibres=500, variation_scale=scale
        )
        messages = generate_dataset(dataset_params=params, rng=rng)
        summary = _summarise(messages)
        summaries.append((label, summary))
        _print_summary(label, summary)

    # Reproducibility check
    rng_a = np.random.default_rng(seed=123)
    rng_b = np.random.default_rng(seed=123)
    params = DatasetGenerationParams(
        n_images=10, target_n_fibres=100, variation_scale=0.3
    )
    msgs_a = generate_dataset(dataset_params=params, rng=rng_a)
    msgs_b = generate_dataset(dataset_params=params, rng=rng_b)
    assert msgs_a == msgs_b, "Same seed should produce identical datasets"
    print("\nReproducibility   : OK (same seed -> identical datasets)")

    # Visualise per-image variation across the three cases
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for col, (label, summary) in enumerate(summaries):
        axes[0, col].scatter(
            range(len(summary["n_fibres"])),
            summary["n_fibres"],
            alpha=0.7,
        )
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("Image index")
        axes[0, col].set_ylabel("n_fibres in image")
        axes[0, col].axhline(500, color="red", linestyle="--",
                             label="Target")
        axes[0, col].legend()

        axes[1, col].scatter(
            summary["mean_length"],
            summary["mean_width"],
            alpha=0.7,
        )
        axes[1, col].set_xlabel("Per-image mean length (mm)")
        axes[1, col].set_ylabel("Per-image mean width (mm)")
        axes[1, col].set_title("Per-image distribution centres")
        
        # Fix axis ranges so all three columns are directly comparable
        axes[0, col].set_ylim(430, 600)
        axes[1, col].set_xlim(2.0, 5.5)
        axes[1, col].set_ylim(0.07, 0.13)

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("dataset_variation_demonstration.png",
                dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()