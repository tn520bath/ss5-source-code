"""
Module 2 - Data Ingestion & Validation.

The front door of SS5's analytical pipeline. Receives
processing_result messages from SS4 (or, during
development, from the synthetic data generator), validates their
schema, flattens them into an internal representation, applies
light per-fibre filtering, and aggregates them into batches
according to the user's configured aggregation rule.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import IngestionParams


# -----------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------

@dataclass
class ValidationIssue:
    """
    One recorded validation failure.

    image_id is None if the message was so malformed that even
    image_id could not be extracted.
    """
    image_id: int | None
    issue_type: str
    description: str


@dataclass
class Batch:
    """
    The data structure emitted by Module 2 when a batch is complete.

    Downstream modules (3 onwards) operate on Batch objects. They
    contain the per-fibre records as a flat pandas DataFrame plus
    metadata supporting the traceability requirement (SYS-FUNC-06).
    """
    batch_id: int
    source_image_ids: list
    fibre_records: pd.DataFrame
    pixel_size_mm: float
    creation_timestamp: str
    validation_issues: list = field(default_factory=list)


# -----------------------------------------------------------------
# Validation
# -----------------------------------------------------------------

def _validate_message(message: dict) -> tuple[bool, str | None]:
    """
    Check that a processing_result message conforms to the agreed
    schema. Returns (is_valid, reason) where reason is None on
    success or a short description of the failure.

    Validation rules:
      - Top-level keys image_id, pixel_size_mm, char all present.
      - image_id is an int.
      - pixel_size_mm is a positive finite float.
      - char is a non-empty list.
      - Every per-fibre record has mesh_id, dimensions, and
        reconstructed_fraction.
      - dimensions contains finite length and width values.
      - reconstructed_fraction is in [0, 1).
    """
    # Top-level structure
    for key in ("image_id", "pixel_size_mm", "char"):
        if key not in message:
            return False, f"missing top-level key: {key}"

    if not isinstance(message["image_id"], int):
        return False, "image_id is not an int"

    psz = message["pixel_size_mm"]
    if not isinstance(psz, (int, float)) or not np.isfinite(psz) or psz <= 0:
        return False, "pixel_size_mm is not a positive finite number"

    if not isinstance(message["char"], list) or len(message["char"]) == 0:
        return False, "char is missing or empty"

    # Per-fibre structure
    for i, fibre in enumerate(message["char"]):
        if not isinstance(fibre, dict):
            return False, f"fibre {i} is not a dict"
        for key in ("mesh_id", "dimensions", "reconstructed_fraction"):
            if key not in fibre:
                return False, f"fibre {i} missing key: {key}"

        dims = fibre["dimensions"]
        if not isinstance(dims, dict):
            return False, f"fibre {i} dimensions is not a dict"
        for dim_key in ("length", "width"):
            if dim_key not in dims:
                return False, f"fibre {i} missing dimension: {dim_key}"
            val = dims[dim_key]
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                return False, f"fibre {i} {dim_key} is not finite"

        rf = fibre["reconstructed_fraction"]
        if not isinstance(rf, (int, float)) or not (0 <= rf < 1):
            return False, f"fibre {i} reconstructed_fraction not in [0, 1)"

    return True, None


# -----------------------------------------------------------------
# Flattening
# -----------------------------------------------------------------

def _flatten_message(message: dict) -> pd.DataFrame:
    """
    Convert a validated processing_result message's per-fibre
    records into a flat pandas DataFrame, one row per fibre.
    """
    records = []
    for fibre in message["char"]:
        records.append({
            "image_id": message["image_id"],
            "mesh_id": fibre["mesh_id"],
            "length": fibre["dimensions"]["length"],
            "width": fibre["dimensions"]["width"],
            "reconstructed_fraction": fibre["reconstructed_fraction"],
        })
    return pd.DataFrame(records)


# -----------------------------------------------------------------
# Per-fibre filtering
# -----------------------------------------------------------------

def _apply_filters(
    df: pd.DataFrame,
    params: IngestionParams,
) -> pd.DataFrame:
    """
    Apply per-fibre filters to a flattened DataFrame.

    Filter 1: drop fibres with non-positive length or width.
    Filter 2: drop fibres where width >= length (geometric
              implausibility, almost certainly a segmentation error).
    Filter 3: flag (do not drop) fibres where reconstructed_fraction
              exceeds the configured threshold. Adds a boolean
              'high_reconstruction' column for downstream use.

    Filter 1 and Filter 2 reduce the row count; Filter 3 only adds
    a column.
    """
    # Filter 1: positive geometry
    keep = (df["length"] > 0) & (df["width"] > 0)

    # Filter 2: aspect ratio plausibility
    keep &= df["width"] < df["length"]

    df_filtered = df[keep].copy()

    # Filter 3: high-reconstruction flagging (no drop)
    df_filtered["high_reconstruction"] = (
        df_filtered["reconstructed_fraction"]
        > params.high_reconstruction_threshold
    )

    return df_filtered


# -----------------------------------------------------------------
# The Aggregator
# -----------------------------------------------------------------

class BatchAggregator:
    """
    Stateful batch aggregator for SS5.

    Receives one processing_result message at a time via ingest().
    Validates, flattens, filters, and accumulates fibres for the
    current in-flight batch. When the configured number of images
    have been ingested into a batch, ingest() returns a completed
    Batch object; otherwise it returns None.

    Malformed messages are recorded in self.validation_issues and
    skipped (permissive mode). They do not contribute to the
    in-flight batch and do not advance the image count.

    Usage
    -----
        aggregator = BatchAggregator(IngestionParams())
        for message in incoming_messages:
            completed = aggregator.ingest(message)
            if completed is not None:
                # Pass the batch to Module 3 etc.
                ...
    """

    def __init__(self, params: IngestionParams | None = None):
        self.params = params or IngestionParams()
        self._next_batch_id = 0
        self._reset_in_flight()
        self.validation_issues: list = []

    def _reset_in_flight(self):
        self._in_flight_dfs: list = []
        self._in_flight_image_ids: list = []
        self._in_flight_pixel_size: float | None = None
        self._in_flight_validation: list = []

    def ingest(self, message: dict) -> Batch | None:
        """
        Ingest one processing_result message.

        Returns a completed Batch if this message triggers batch
        completion; otherwise returns None. Malformed messages
        are skipped (logged but not counted toward the batch).
        """
        # Validate
        is_valid, reason = _validate_message(message)
        if not is_valid:
            image_id = (
                message.get("image_id")
                if isinstance(message, dict)
                else None
            )
            issue = ValidationIssue(
                image_id=image_id,
                issue_type="schema",
                description=reason,
            )
            self.validation_issues.append(issue)
            self._in_flight_validation.append(issue)
            return None

        # Flatten and filter
        df_raw = _flatten_message(message)
        df_filtered = _apply_filters(df_raw, self.params)

        # Accumulate
        self._in_flight_dfs.append(df_filtered)
        self._in_flight_image_ids.append(message["image_id"])

        # First image of a batch sets the calibration
        if self._in_flight_pixel_size is None:
            self._in_flight_pixel_size = message["pixel_size_mm"]

        # Has the batch reached its target size?
        if len(self._in_flight_image_ids) >= self.params.images_per_batch:
            return self._emit_batch()

        return None

    def _emit_batch(self) -> Batch:
        """Concatenate the in-flight DataFrames and emit a Batch."""
        all_records = pd.concat(self._in_flight_dfs, ignore_index=True)

        batch = Batch(
            batch_id=self._next_batch_id,
            source_image_ids=self._in_flight_image_ids.copy(),
            fibre_records=all_records,
            pixel_size_mm=self._in_flight_pixel_size,
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            validation_issues=self._in_flight_validation.copy(),
        )

        self._next_batch_id += 1
        self._reset_in_flight()
        return batch

    def force_emit(self) -> Batch | None:
        """
        Force the current in-flight batch to be emitted, even if
        it has not reached the target size. Returns None if no
        images are in flight. Useful at end-of-stream.
        """
        if not self._in_flight_image_ids:
            return None
        return self._emit_batch()