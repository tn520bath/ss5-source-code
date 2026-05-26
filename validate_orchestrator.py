"""
Confirms end-to-end behaviour by feeding the orchestrator a
Batch produced by Module 1 (synthetic generator) and Module 2
(BatchAggregator).
"""

import warnings

import numpy as np

from ss5_module1_synthetic_generator.batch_assembly import generate_image
from ss5_module2_7_pipeline.ingestion import BatchAggregator
from ss5_module2_7_pipeline.config import IngestionParams
from ss5_module3_descriptors.descriptor_orchestrator import (
    compute_descriptors,
    BatchDescriptors,
)


def main():
    rng = np.random.default_rng(seed=42)

    # --- Build a Batch via Module 1 + Module 2 ---
    aggregator = BatchAggregator(IngestionParams(images_per_batch=1))

    msg = generate_image(n_fibres=500, image_id=0, rng=rng)
    batch = aggregator.ingest(msg)
    assert batch is not None, "Aggregator should have emitted a batch"

    print("--- Constructed batch from Module 1 + Module 2 ---")
    print(f"  batch_id    = {batch.batch_id}")
    print(f"  n_fibres    = {len(batch.fibre_records)}")
    print(f"  pixel size  = {batch.pixel_size_mm}")
    print(f"  source IDs  = {batch.source_image_ids}")
    print(f"  timestamp   = {batch.creation_timestamp}")

    # --- Test 1: orchestrator produces a populated BatchDescriptors ---
    print("\n--- Test 1: orchestrator end-to-end ---")
    bd = compute_descriptors(batch)

    assert isinstance(bd, BatchDescriptors)
    assert bd.batch_id == batch.batch_id
    assert bd.n_fibres == len(batch.fibre_records)
    assert bd.descriptors_length["mean"] > 0
    assert bd.descriptors_width["mean"] > 0
    assert bd.descriptors_aspect_ratio["mean"] > 0
    assert bd.descriptors_length_mass_weighted["mean_mass_w"] > 0
    assert bd.fines["fines_fraction_count"] >= 0
    assert bd.fit_length_lognormal.name == "lognormal"
    assert bd.fit_length_weibull.name == "weibull"
    assert bd.fit_width_lognormal.name == "lognormal"
    print(f"  length mean         = {bd.descriptors_length['mean']:.4f}")
    print(f"  length D50          = {bd.descriptors_length['D50']:.4f}")
    print(f"  length D50 (mass-w) = "
          f"{bd.descriptors_length_mass_weighted['D50_mass_w']:.4f}")
    print(f"  width mean          = {bd.descriptors_width['mean']:.4f}")
    print(f"  fines (count)       = "
          f"{bd.fines['fines_fraction_count']:.4f}")
    print(f"  fines (mass)        = "
          f"{bd.fines['fines_fraction_mass']:.4f}")
    print(f"  reconstruction mean = "
          f"{bd.reconstruction['mean_reconstructed_fraction']:.4f}")
    print(f"  AIC lognormal       = "
          f"{bd.gof_length_lognormal['aic']:.2f}")
    print(f"  AIC weibull         = "
          f"{bd.gof_length_weibull['aic']:.2f}")
    print(f"  preferred (length)  = "
          f"{bd.aic_comparison_length['preferred']}")
    print(f"  evidence            = "
          f"{bd.aic_comparison_length['evidence_strength']}")
    print("  -> Passed: BatchDescriptors fully populated.")

    # --- Test 2: descriptor table renders ---
    print("\n--- Test 2: descriptor table to DataFrame ---")
    table = bd.to_dataframe()
    print(table)
    assert table.shape == (9, 4), \
        f"Expected (9, 4) table, got {table.shape}"
    assert "length" in table.columns
    assert "width" in table.columns
    assert "aspect_ratio" in table.columns
    assert "length_mass_weighted" in table.columns
    print("  -> Passed: table shape and columns correct.")

    # --- Test 3: length AIC comparison favours lognormal ---
    print("\n--- Test 3: AIC comparison favours lognormal ---")
    assert bd.aic_comparison_length["preferred"] == "lognormal"
    print(f"  preferred = {bd.aic_comparison_length['preferred']} "
          f"(delta_aic = {bd.aic_comparison_length['delta_aic']:.2f})")
    print("  -> Passed: lognormal preferred for length on lognormal data.")

    # --- Test 4: small-N warning fires (single consolidated) ---
    print("\n--- Test 4: small-N warning on tiny batch ---")
    msg_small = generate_image(n_fibres=20, image_id=99, rng=rng)
    aggregator2 = BatchAggregator(IngestionParams(images_per_batch=1))
    batch_small = aggregator2.ingest(msg_small)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bd_small = compute_descriptors(batch_small)
        relevant = [ww for ww in w if "below threshold" in str(ww.message)]
        assert len(relevant) == 1, \
            f"Expected one consolidated warning, got {len(relevant)}"
        print(f"  Warning: {relevant[0].message}")
    print("  -> Passed: single consolidated small-N warning fired.")

    print("\nAll Module 3 orchestrator validation checks passed.")


if __name__ == "__main__":
    main()