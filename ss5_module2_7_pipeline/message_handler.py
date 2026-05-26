"""
Module 7 - Message Handler.

The gateway between Sub-system 5 and the wider system's message
broker. Wraps the team's shared Node infrastructure (common.Node)
and exposes a clean internal API to the rest of SS5: incoming
processing_result messages from Sub-system 4 are validated,
flattened, and aggregated by Module 2 (Data Ingestion); ready
signals are sent back upstream to drive demand-driven flow control.

Module 7 is the only part of SS5 that knows about the broker.
Modules 2 onward operate on plain Python dicts and DataFrames and
are fully testable without any messaging infrastructure.

This module does not handle output to the database: persistence is
Module 8's responsibility, and SS5 does not send analytical
outputs over the broker (per the system interface diagram).
"""

import asyncio
from multiprocessing import Queue
from typing import Callable

from common import Node, cprint

from .ingestion import BatchAggregator, Batch
from .config import IngestionParams


# Type alias for the callback invoked when Module 2 emits a completed batch
BatchReadyCallback = Callable[[Batch], None]


# -----------------------------------------------------------------
# The Message Handler
# -----------------------------------------------------------------

class MessageHandler:
    """
    Wraps the team's Node and routes incoming SS4 messages into
    the SS5 analytical pipeline via Module 2's BatchAggregator.

    Construction is dependency-injection style: the Node and the
    BatchAggregator are passed in rather than constructed
    internally, which makes the class easy to test in isolation
    (with a mock Node and a fresh aggregator).

    Usage
    -----
        node = Node("ss5", inbox, peers)
        aggregator = BatchAggregator(IngestionParams())
        handler = MessageHandler(
            node=node,
            aggregator=aggregator,
            on_batch_ready=lambda batch: print(f"Batch {batch.batch_id} ready"),
        )
        handler.register_handlers()
        handler.start()
        await handler.run_forever()
    """

    def __init__(
        self,
        node: Node,
        aggregator: BatchAggregator,
        on_batch_ready: BatchReadyCallback | None = None,
    ):
        self.node = node
        self.aggregator = aggregator
        # If no callback is supplied, completed batches are logged
        # and discarded. This is the expected default during early
        # development before the downstream analytical pipeline is wired in.
        self.on_batch_ready = on_batch_ready or self._default_batch_handler

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------

    def register_handlers(self):
        """Register message handlers with the underlying Node."""
        self.node.on("processing_result", self._on_processing_result)

    def start(self):
        """
        Send the initial ready signals to upstream subsystems to
        kickstart the pipeline. Should be called once after
        register_handlers().
        """
        self._signal_ready_upstream()
        cprint("ss5", "Pipeline started; waiting for processing_result messages")

    async def run_forever(self, poll_sleep: float = 0.1):
        """
        Run the polling loop indefinitely. Calls node.poll() to
        drain the inbox, then sleeps briefly before checking again.

        poll_sleep is the time in seconds between polling cycles.
        Lower values reduce latency at the cost of CPU usage when
        the inbox is empty.
        """
        while True:
            await self.node.poll()
            await asyncio.sleep(poll_sleep)

    # -------------------------------------------------------------
    # Internal handlers
    # -------------------------------------------------------------

    async def _on_processing_result(self, msg: dict):
        """
        Handle one incoming processing_result message.

        Extracts the SS4 result payload, hands it to the aggregator,
        and invokes the on_batch_ready callback if a batch is
        completed. Always signals ready to upstream after handling
        (regardless of whether a batch was emitted), maintaining
        one-in-one-out flow control.
        """
        result = msg["data"]["result"]
        image_id = result.get("image_id", "?")

        cprint("ss5",
               f"Received processing_result for image {image_id} "
               f"({len(result.get('char', []))} fibres)")

        completed_batch = self.aggregator.ingest(result)

        if completed_batch is not None:
            cprint("ss5",
                   f"Batch {completed_batch.batch_id} ready: "
                   f"{completed_batch.fibre_records.shape[0]} fibres "
                   f"from images {completed_batch.source_image_ids}")
            self.on_batch_ready(completed_batch)

        # Always signal ready; do not gate on batch completion
        self._signal_ready_upstream()

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def _signal_ready_upstream(self):
        """Send ready_message to both SS3 and SS4."""
        self.node.send("ss3", "ready_message", {})
        self.node.send("ss4", "ready_message", {})

    @staticmethod
    def _default_batch_handler(batch: Batch):
        """
        Default callback used when no on_batch_ready is supplied.
        Logs the batch arrival and discards it.
        """
        cprint("ss5",
               f"[default handler] Batch {batch.batch_id} discarded "
               f"({batch.fibre_records.shape[0]} fibres)")


# -----------------------------------------------------------------
# Top-level process entry point
# -----------------------------------------------------------------

def run_ss5(
    inbox: Queue,
    peers: dict[str, Queue],
    ingestion_params: IngestionParams | None = None,
    on_batch_ready: BatchReadyCallback | None = None,
):
    """
    Top-level entry point compatible with the team's process-spawning
    convention.

    Constructs the Node, BatchAggregator, and MessageHandler, then
    runs the async polling loop indefinitely.
    """

    async def main():
        node = Node("ss5", inbox, peers)
        aggregator = BatchAggregator(ingestion_params)
        handler = MessageHandler(
            node=node,
            aggregator=aggregator,
            on_batch_ready=on_batch_ready,
        )
        handler.register_handlers()
        handler.start()
        await handler.run_forever()

    asyncio.run(main())