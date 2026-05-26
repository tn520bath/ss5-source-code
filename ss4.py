"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

ss4.py
Image processing sub-system emulator.

This standalone version is optional-dependency safe:
- torch / torchvision are only needed if build_model() is deliberately called.
- cv2 is optional; if unavailable, the emulator still sends fabricated
  processing_result messages to SS5.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import asyncio
from multiprocessing import Queue

from common import Node, cprint


try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None


def build_model(
    variant: str = "maskrcnn_resnet50_fpn_v2",
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
):
    """
    Build a Mask R-CNN model fine-tuned for fibre segmentation.

    This function is retained from the wider team integration emulator, but
    it is not required by the SS5 validation scripts. It imports torchvision
    only when called.

    Args:
        variant:
            Torchvision model name.
        pretrained:
            Whether to load COCO pre-trained weights.
        trainable_backbone_layers:
            Number of FPN backbone layers to unfreeze.

    Returns:
        torch.nn.Module ready for training.

    Raises:
        ImportError:
            If torch or torchvision is unavailable.
    """
    if torch is None:
        raise ImportError(
            "build_model() requires torch. Install torch and torchvision to use the SS4 model builder."
        )

    try:
        import torchvision
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            MaskRCNN_ResNet50_FPN_V2_Weights,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except ImportError as exc:
        raise ImportError(
            "build_model() requires torchvision. Install torch and torchvision to use the SS4 model builder."
        ) from exc

    num_classes = 2  # 0 = background, 1 = fibre

    if variant == "maskrcnn_resnet50_fpn_v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    # Replace box classifier head.
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace mask head.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    return model


class ImageProcessingSS4:
    """
    Simulates an image-processing sub-system.

    The current emulator returns fabricated per-fibre measurements in the
    processing_result schema expected by SS5.
    """

    def __init__(self):
        if torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        print(f"Device : {self.device}")

    def run(self, image, metadata):
        """
        Return fabricated fibre-characterisation results.

        The image and metadata inputs are accepted to match the real SS4
        processing interface, but are not used by this lightweight emulator.
        """
        ret_val = {
            "image_id": 99999,
            "char": [
                {
                    "mesh_id": 0,
                    "dimensions": {
                        "length": 1.3,
                        "width": 0.06,
                    },
                },
                {
                    "mesh_id": 1,
                    "dimensions": {
                        "length": 1.8,
                        "width": 0.03,
                    },
                },
                {
                    "mesh_id": 2,
                    "dimensions": {
                        "length": 0.9,
                        "width": 0.03,
                    },
                },
                {
                    "mesh_id": 3,
                    "dimensions": {
                        "length": 2.0,
                        "width": 0.08,
                    },
                },
            ],
        }
        return ret_val


def run_ss4(inbox: Queue, peers: dict[str, Queue]):
    """
    Run the SS4 image-processing emulator.

    SS4 receives image_data_message from SS3, fabricates a processing_result,
    and sends it to SS5.
    """

    async def main():
        node = Node("ss4", inbox, peers)
        proc = ImageProcessingSS4()

        cprint("ss4", "Image processor ready.")

        async def on_publish_ready(msg):
            cprint("ss4", "Ready signal received; client-facing output route would update here.")

        async def on_image_data(msg):
            image_path = msg["data"]["image_path"]
            metadata = msg["data"]["metadata"]

            image = cv2.imread(image_path) if cv2 is not None else None
            shape = image.shape if image is not None else "not loaded"

            x_mm = metadata.get("x_mm", "unknown")
            y_mm = metadata.get("y_mm", "unknown")

            cprint(
                "ss4",
                f"Loaded {image_path} ({shape}) (x={x_mm}, y={y_mm})",
            )

            result = proc.run(image, metadata)

            cprint("ss4", f"Processing complete: {result}")

            send_analysis(result)

        async def on_no_images(msg):
            reason = msg.get("data", {}).get("reason", "no reason provided")
            cprint("ss4", f"ss3 has no more images. Waiting. Reason: {reason}")

        def send_analysis(result):
            """
            Send fabricated image-analysis results to SS5.
            """
            node.send(
                "ss5",
                "processing_result",
                {
                    "result": result,
                },
            )

        node.on("image_data_message", on_image_data)
        node.on("no_images", on_no_images)
        node.on("ready_message", on_publish_ready)

        while True:
            await node.poll()
            await asyncio.sleep(8)

    asyncio.run(main())