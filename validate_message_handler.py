"""
Validation of Module 7 (Message Handler).

Tests the MessageHandler in isolation by mocking the team's Node
class. This avoids the need to spin up the full multiprocessing
broker just to verify wiring.

Tests:
  - Incoming processing_result messages are correctly extracted
    and handed to the aggregator.
  - Ready signals are sent to SS3 and SS4 after every ingested
    message, regardless of whether a batch was emitted.
  - The on_batch_ready callback is invoked exactly when a batch
    is completed.
  - The default callback is used when no callback is supplied
    (batches are logged and discarded).
"""

import asyncio
from datetime import datetime, timezone

import numpy as np

from ss5_module1_synthetic_generator.batch_assembly import generate_image
from ss5_module2_7_pipeline.config import IngestionParams
from ss5_module2_7_pipeline.ingestion import BatchAggregator, Batch


# -----------------------------------------------------------------
# Mock Node implementation
# -----------------------------------------------------------------

class MockNode:
    """
    A minimal stand-in for common.Node that records all calls
    instead of actually putting things on multiprocessing queues.

    Has the same interface as the real Node so MessageHandler
    cannot tell the difference.
    """

    def __init__(self, name="ss5"):
        self.name = name
        self._handlers = {}
        self.sent_messages = []

    def on(self, message_type, handler):
        self._handlers[message_type] = handler

    def send(self, target, message_type, data):
        self.sent_messages.append({
            "target": target,
            "type": message_type,
            "data": data,
        })

    async def poll(self):
        # No-op for testing: messages are dispatched manually
        # by inject_message() rather than drained from a queue.
        pass

    async def inject_message(self, msg_type, msg):
        """Helper to manually invoke a registered handler."""
        handler = self._handlers.get(msg_type)
        if handler:
            await handler(msg)


# -----------------------------------------------------------------
# Patching cprint to be quiet during tests
# -----------------------------------------------------------------

# We don't want to depend on common.py being on the path during
# standalone testing, so we monkey-patch ss5_module2_7_pipeline.message_handler
# to use MockNode instead of the real one. We also stub out cprint
# to keep the test output clean.

import ss5_module2_7_pipeline.message_handler as mh_module
mh_module.Node = MockNode  # Replace the imported Node with our mock
mh_module.cprint = lambda *args, **kwargs: None  # Silence prints

from ss5_module2_7_pipeline.message_handler import MessageHandler


# -----------------------------------------------------------------
# Tests
# -----------------------------------------------------------------

def _make_processing_result_msg(image_id: int, n_fibres: int,
                                 rng: np.random.Generator) -> dict:
    """
    Wrap a synthetic generator output in the broker's envelope
    format (matching what Node.send produces and what MessageHandler
    expects to receive).
    """
    payload = generate_image(n_fibres=n_fibres, image_id=image_id, rng=rng)
    return {
        "type": "processing_result",
        "sender": "ss4",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": {"result": payload},
    }


async def main():
    rng = np.random.default_rng(seed=42)

    # Case 1: single-image-per-batch, custom callback
    print("--- Case 1: single image per batch, callback wired up ---")
    node = MockNode()
    aggregator = BatchAggregator()  # default 1 image per batch
    received_batches = []

    handler = MessageHandler(
        node=node,
        aggregator=aggregator,
        on_batch_ready=lambda b: received_batches.append(b),
    )
    handler.register_handlers()
    handler.start()

    # start() should have sent ready signals to SS3 and SS4
    assert len(node.sent_messages) == 2
    assert {m["target"] for m in node.sent_messages} == {"ss3", "ss4"}
    assert all(m["type"] == "ready_message" for m in node.sent_messages)
    print(f"After start(): {len(node.sent_messages)} ready signals sent")

    # Inject one processing_result message
    msg = _make_processing_result_msg(image_id=0, n_fibres=50, rng=rng)
    await node.inject_message("processing_result", msg)

    assert len(received_batches) == 1, \
        f"Expected 1 batch, got {len(received_batches)}"
    assert received_batches[0].batch_id == 0
    assert received_batches[0].source_image_ids == [0]
    print(f"Batch received: id={received_batches[0].batch_id}, "
          f"{received_batches[0].fibre_records.shape[0]} fibres")

    # After processing one message, two MORE ready signals should
    # have been sent (total now 4: 2 from start, 2 from processing)
    assert len(node.sent_messages) == 4
    print(f"After 1 message: {len(node.sent_messages)} total ready signals")

    # Case 2: multi-image-per-batch, callback fires only on completion
    print("\n--- Case 2: 3 images per batch, callback fires once ---")
    node = MockNode()
    aggregator = BatchAggregator(IngestionParams(images_per_batch=3))
    received_batches = []

    handler = MessageHandler(
        node=node,
        aggregator=aggregator,
        on_batch_ready=lambda b: received_batches.append(b),
    )
    handler.register_handlers()
    handler.start()

    for image_id in range(3):
        msg = _make_processing_result_msg(image_id=image_id,
                                           n_fibres=50, rng=rng)
        await node.inject_message("processing_result", msg)

    # Only one batch should have completed (after the third image)
    assert len(received_batches) == 1
    assert received_batches[0].source_image_ids == [0, 1, 2]
    print(f"Batch received after 3 images: "
          f"id={received_batches[0].batch_id}, "
          f"images={received_batches[0].source_image_ids}")

    # Ready signals: 2 (start) + 2*3 (one pair per ingested msg) = 8
    expected_signals = 2 + 2 * 3
    assert len(node.sent_messages) == expected_signals, \
        f"Expected {expected_signals} ready signals, got " \
        f"{len(node.sent_messages)}"
    print(f"Ready signals sent: {len(node.sent_messages)} "
          f"(matches one pair per message regardless of batch completion)")

    # Case 3: default callback when no callback is supplied
    print("\n--- Case 3: default callback discards batches ---")
    node = MockNode()
    aggregator = BatchAggregator()
    handler = MessageHandler(node=node, aggregator=aggregator)
    handler.register_handlers()
    handler.start()

    msg = _make_processing_result_msg(image_id=99, n_fibres=10, rng=rng)
    await node.inject_message("processing_result", msg)

    # No callback was supplied, so nothing to assert on the receiving
    # side. Just confirm that no exception was raised and ready
    # signals were still sent.
    assert len(node.sent_messages) == 4
    print("Message handled with default callback; ready signals sent.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())