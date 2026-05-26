"""
ss5.py

Lightweight SS5 modelling-subsystem emulator.

This file is part of the integration emulator used to demonstrate message
flow between SS3, SS4 and SS5. It receives fabricated processing_result
messages from SS4, prints the per-fibre measurements, and issues
ready_message signals back to SS3 and SS4.
"""

import asyncio
from multiprocessing import Queue

from common import Node, cprint


def run_ss5(inbox: Queue, peers: dict[str, Queue]):
    """
    Run the SS5 integration emulator.

    This is not the full analytical SS5 pipeline. The implemented analytical
    modules are exercised by the validate_*.py scripts and module packages.
    This emulator exists only to demonstrate broker-level message flow.
    """

    async def main():
        node = Node("ss5", inbox, peers)

        def signal_ready():
            """
            Request the next upstream image-processing cycle.
            """
            node.send("ss3", "ready_message", {})
            node.send("ss4", "ready_message", {})
            cprint("ss5", "Sent ready signals to SS3 and SS4")

        async def on_processing_result(msg):
            result = msg["data"]["result"]
            image_id = result["image_id"]
            fibres = result["char"]

            cprint(
                "ss5",
                f"Received characterisation for image {image_id} "
                f"({len(fibres)} fibres)",
            )

            for fibre in fibres:
                mesh_id = fibre["mesh_id"]
                length = fibre["dimensions"]["length"]
                width = fibre["dimensions"]["width"]
                cprint(
                    "ss5",
                    f"  mesh {mesh_id}: length={length}, width={width}",
                )

            signal_ready()

        node.on("processing_result", on_processing_result)

        node.send("ss3", "ready_message", {})
        cprint("ss5", "Sent initial ready signal to SS3")

        while True:
            await node.poll()
            await asyncio.sleep(10)

    asyncio.run(main())