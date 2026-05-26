"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

ss3.py
Image capture sub-system emulator.

This standalone version is kept lightweight for the SS5 source-code
submission. It does not require OpenCV at import time.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import asyncio
import os
import pathlib
from multiprocessing import Queue

from common import Node, cprint


IMAGE_PATH = pathlib.Path(__file__).parent / "data"


class ImageCaptureSS3:
    """
    Simulates an image source with a stack of image filenames and metadata.
    """

    def __init__(self):
        self.images = self._get_images()
        self.metadata = [
            {"x_mm": 100, "y_mm": 100, "valid": True},
            {"x_mm": 100, "y_mm": 100, "valid": True},
            {"x_mm": 100, "y_mm": 100, "valid": False},
            {"x_mm": 100, "y_mm": 100, "valid": True},
            {"x_mm": 100, "y_mm": 100, "valid": True},
        ]

    def _get_images(self):
        """
        Return image files from the local data folder if it exists.

        If the source-code package is submitted without image data, the
        emulator remains importable and simply reports an empty image stack.
        """
        if not IMAGE_PATH.exists():
            cprint("ss3", f"No data folder found at {IMAGE_PATH}; image stack is empty.")
            return []

        return [
            file_name
            for file_name in os.listdir(IMAGE_PATH)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

    def pop(self):
        """
        Pop one image path and one metadata record from the stack.
        """
        if self.images and self.metadata:
            filename = self.images.pop()
            meta = self.metadata.pop()
            return IMAGE_PATH / filename, meta

        return None, None

    @property
    def remaining(self):
        """
        Number of images still available in the stack.
        """
        return len(self.images)


def run_ss3(inbox: Queue, peers: dict[str, Queue]):
    """
    Run the SS3 image-capture emulator.

    SS3 waits for a ready_message and then sends either image_data_message
    to SS4 or no_images if the image stack is empty.
    """

    async def main():
        node = Node("ss3", inbox, peers)
        capture = ImageCaptureSS3()

        cprint("ss3", f"Images ready. {capture.remaining} images in stack.")

        async def on_ready(msg):
            image_path, metadata = capture.pop()

            if image_path is not None:
                node.send(
                    "ss4",
                    "image_data_message",
                    {
                        "image_path": str(image_path),
                        "metadata": metadata,
                    },
                )
                cprint("ss3", f"Published {image_path.name} to ss4")
            else:
                node.send("ss4", "no_images", {"reason": "Stack empty"})
                cprint("ss3", "No images left; sent no_images to ss4")

        node.on("ready_message", on_ready)

        while True:
            await node.poll()
            await asyncio.sleep(8)

    asyncio.run(main())