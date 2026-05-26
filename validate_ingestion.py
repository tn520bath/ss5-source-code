"""
Validation of Module 2 (Data Ingestion & Validation).

Tests:
  - Single-image-per-batch (default) emits a batch on every
    ingest call.
  - Multi-image-per-batch correctly aggregates and emits only
    on completion.
  - Schema validation rejects malformed messages without
    blocking the pipeline.
  - Per-fibre filters correctly drop bad fibres and flag
    high-reconstruction ones.
  - The Batch dataclass carries the expected metadata.
  - End-to-end: synthetic generator output -> Module 2
    completes a batch with valid contents.
"""

import numpy as np

from ss5_module1_synthetic_generator.config import (
    DatasetGenerationParams,
)
from ss5_module1_synthetic_generator.batch_assembly import generate_image
from ss5_module1_synthetic_generator.dataset_generator import generate_dataset

from ss5_module2_7_pipeline.config import IngestionParams
from ss5_module2_7_pipeline.ingestion import BatchAggregator


def main():
    rng = np.random.default_rng(seed=42)

    # Case 1: single-image-per-batch (default), end-to-end with
    # the synthetic generator
    print("--- Case 1: single image per batch ---")
    aggregator = BatchAggregator()
    message = generate_image(n_fibres=100, image_id=0, rng=rng)
    batch = aggregator.ingest(message)

    assert batch is not None, "Expected immediate batch emission"
    assert batch.batch_id == 0, f"Expected batch_id 0, got {batch.batch_id}"
    assert len(batch.source_image_ids) == 1
    assert batch.fibre_records.shape[0] <= 100, \
        "Filter may have removed some fibres; cannot exceed 100"
    assert "high_reconstruction" in batch.fibre_records.columns
    print(f"Batch emitted: {batch.fibre_records.shape[0]} fibres "
          f"after filtering (from 100 ingested)")
    print(f"High-reconstruction count: "
          f"{batch.fibre_records['high_reconstruction'].sum()}")

    # Case 2: multi-image-per-batch
    print("\n--- Case 2: 3 images per batch ---")
    params = IngestionParams(images_per_batch=3)
    aggregator = BatchAggregator(params)

    msg_1 = generate_image(n_fibres=100, image_id=0, rng=rng)
    msg_2 = generate_image(n_fibres=100, image_id=1, rng=rng)
    msg_3 = generate_image(n_fibres=100, image_id=2, rng=rng)

    assert aggregator.ingest(msg_1) is None, \
        "Should not emit on image 1"
    assert aggregator.ingest(msg_2) is None, \
        "Should not emit on image 2"
    batch = aggregator.ingest(msg_3)
    assert batch is not None, "Should emit on image 3"
    assert len(batch.source_image_ids) == 3
    print(f"Batch emitted after 3 images: "
          f"{batch.fibre_records.shape[0]} fibres total")

    # Case 3: schema validation rejects malformed messages
    print("\n--- Case 3: malformed message handling ---")
    aggregator = BatchAggregator()

    bad_messages = [
        {},  # empty
        {"image_id": "not_an_int", "pixel_size_mm": 0.001, "char": []},
        {"image_id": 0, "pixel_size_mm": -1, "char": [{}]},
        {"image_id": 0, "pixel_size_mm": 0.001, "char": []},  # empty char
    ]
    for bad in bad_messages:
        result = aggregator.ingest(bad)
        assert result is None, "Bad message should not produce a batch"

    print(f"Rejected {len(aggregator.validation_issues)} malformed messages")
    for issue in aggregator.validation_issues:
        print(f"  image_id={issue.image_id}: {issue.description}")

    # Confirm a good message still works after the bad ones
    good = generate_image(n_fibres=10, image_id=42, rng=rng)
    batch = aggregator.ingest(good)
    assert batch is not None, "Should recover from earlier rejections"
    print("Good message after rejections produces a batch correctly.")

    # Case 4: filters
    print("\n--- Case 4: filter behaviour ---")
    aggregator = BatchAggregator(
        IngestionParams(high_reconstruction_threshold=0.3)
    )

    # Construct a message with deliberately bad fibres
    crafted = {
        "image_id": 99,
        "pixel_size_mm": 0.000977,
        "char": [
            {"mesh_id": 0, "dimensions": {"length": 3.0, "width": 0.1},
             "reconstructed_fraction": 0.0},      # OK
            {"mesh_id": 1, "dimensions": {"length": -0.5, "width": 0.1},
             "reconstructed_fraction": 0.0},      # negative length
            {"mesh_id": 2, "dimensions": {"length": 0.05, "width": 0.1},
             "reconstructed_fraction": 0.0},      # width >= length
            {"mesh_id": 3, "dimensions": {"length": 5.0, "width": 0.08},
             "reconstructed_fraction": 0.4},      # high reconstruction (>0.3)
        ],
    }
    batch = aggregator.ingest(crafted)
    assert batch is not None
    df = batch.fibre_records
    assert df.shape[0] == 2, \
        f"Expected 2 fibres after filtering, got {df.shape[0]}"
    assert df["high_reconstruction"].sum() == 1, \
        "Expected exactly one high-reconstruction flag"
    print(f"Filters work correctly: 2/4 fibres kept, "
          f"1 flagged as high-reconstruction")

    # Case 5: end-to-end with multi-image dataset
    print("\n--- Case 5: end-to-end with synthetic dataset ---")
    rng = np.random.default_rng(seed=42)
    dataset_params = DatasetGenerationParams(
        n_images=12, target_n_fibres=200, variation_scale=0.2
    )
    messages = generate_dataset(dataset_params=dataset_params, rng=rng)

    aggregator = BatchAggregator(IngestionParams(images_per_batch=4))
    completed_batches = []
    for msg in messages:
        b = aggregator.ingest(msg)
        if b is not None:
            completed_batches.append(b)

    print(f"Ingested {len(messages)} messages, "
          f"emitted {len(completed_batches)} batches")
    assert len(completed_batches) == 3
    for b in completed_batches:
        print(f"  Batch {b.batch_id}: "
              f"{b.fibre_records.shape[0]} fibres, "
              f"images {b.source_image_ids}, "
              f"pixel_size={b.pixel_size_mm}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()