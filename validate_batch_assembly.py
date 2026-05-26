"""
Validation of the batch assembly function.

Confirms that generate_image:
  - Produces a message structurally matching the SS4 schema.
  - Contains the correct number of fibres.
  - Carries pixel_size_mm at the message top level.
  - Has reconstructed_fraction at the per-fibre level (alongside
    dimensions, not inside it).
  - Reproduces deterministically when given the same seed.
"""

import json

import numpy as np

from ss5_module1_synthetic_generator.batch_assembly import generate_image


def main():
    rng = np.random.default_rng(seed=42)

    # Generate a small message for inspection
    message = generate_image(n_fibres=5, image_id=42, rng=rng)

    print("Generated message structure (n_fibres=5):")
    print(json.dumps(message, indent=2))

    # Schema checks
    print("\n--- Schema validation ---")
    assert "image_id" in message, "Missing top-level image_id"
    assert "pixel_size_mm" in message, "Missing top-level pixel_size_mm"
    assert "char" in message, "Missing top-level char"
    assert "batch_id" not in message, "batch_id should NOT be in the message"
    print("Top-level fields  : OK (image_id, pixel_size_mm, char present; batch_id absent)")

    assert isinstance(message["char"], list), "char must be a list"
    assert len(message["char"]) == 5, "char length should match n_fibres"
    print(f"char list length  : OK ({len(message['char'])} fibres)")

    fibre = message["char"][0]
    assert "mesh_id" in fibre, "Missing per-fibre mesh_id"
    assert "dimensions" in fibre, "Missing per-fibre dimensions"
    assert "reconstructed_fraction" in fibre, \
        "Missing per-fibre reconstructed_fraction"
    assert "length" in fibre["dimensions"], "Missing length in dimensions"
    assert "width" in fibre["dimensions"], "Missing width in dimensions"
    assert "reconstructed_fraction" not in fibre["dimensions"], \
        "reconstructed_fraction should NOT be inside dimensions"
    print("Per-fibre schema  : OK (mesh_id, dimensions[length, width], "
          "reconstructed_fraction at correct levels)")

    # Reproducibility check
    rng_a = np.random.default_rng(seed=123)
    rng_b = np.random.default_rng(seed=123)
    msg_a = generate_image(n_fibres=10, image_id=0, rng=rng_a)
    msg_b = generate_image(n_fibres=10, image_id=0, rng=rng_b)

    assert msg_a == msg_b, "Same seed should produce identical messages"
    print("Reproducibility   : OK (same seed -> identical message)")

    # Population-level sanity check on a larger image
    rng_pop = np.random.default_rng(seed=42)
    large = generate_image(n_fibres=5000, image_id=0, rng=rng_pop)

    lengths = np.array([f["dimensions"]["length"] for f in large["char"]])
    widths = np.array([f["dimensions"]["width"] for f in large["char"]])
    rec_fracs = np.array(
        [f["reconstructed_fraction"] for f in large["char"]]
    )

    print(f"\n--- Population statistics (n_fibres=5000) ---")
    print(f"Lengths : mean = {np.mean(lengths):.3f} mm, "
          f"median = {np.median(lengths):.3f} mm, "
          f"std = {np.std(lengths):.3f} mm")
    print(f"Widths  : mean = {np.mean(widths):.4f} mm, "
          f"median = {np.median(widths):.4f} mm, "
          f"std = {np.std(widths):.4f} mm")
    print(f"Reconstructed fraction : "
          f"{100 * np.mean(rec_fracs > 0):.1f} % of fibres reconstructed, "
          f"mean fraction (when nonzero) = "
          f"{np.mean(rec_fracs[rec_fracs > 0]):.3f}")
    print(f"pixel_size_mm : {large['pixel_size_mm']}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()