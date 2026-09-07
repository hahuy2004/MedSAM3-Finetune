#!/usr/bin/env python3
"""
SAM3 + LoRA Inference Script

Based on official SAM3 batched inference patterns.
Supports text prompts and visual prompts with LoRA fine-tuned weights.

Usage:
    # Text prompt inference
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" \
        --output output.png

    # Multiple prompts
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" "defect" "damage" \
        --output output.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from torchvision.ops import nms

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights


class SAM3LoRAInference:
    """SAM3 model with LoRA for inference."""

    def __init__(
        self,
        config_path: str,
        weights_path: Optional[str] = None,
        resolution: int = 1008,
        detection_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize SAM3 with LoRA.

        Args:
            config_path: Path to training config YAML
            weights_path: Path to LoRA weights (optional, auto-detected from config)
            resolution: Input image resolution (default: 1008)
            detection_threshold: Confidence threshold for detections (default: 0.5)
            nms_iou_threshold: IoU threshold for NMS (default: 0.5)
            device: Device to run on (default: "cuda")
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Auto-detect weights if not provided
        if weights_path is None:
            output_dir = self.config.get('output', {}).get('output_dir', 'outputs/sam3_lora_full')
            weights_path = os.path.join(output_dir, 'best_lora_weights.pt')
            print(f"ℹ️  Auto-detected weights: {weights_path}")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

        self.weights_path = weights_path
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"🔧 Initializing SAM3 + LoRA...")
        print(f"   Device: {self.device}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Confidence threshold: {detection_threshold}")
        print(f"   NMS IoU threshold: {nms_iou_threshold}")

        # Build base model
        print("\n📦 Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=True
        )

        # Apply LoRA configuration
        print("🔗 Applying LoRA configuration...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,  # No dropout during inference
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # Load LoRA weights
        print(f"💾 Loading LoRA weights from {weights_path}...")
        load_lora_weights(self.model, weights_path)

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms (official SAM3 pattern)
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Setup postprocessor
        # Note: Using simpler manual postprocessing instead of PostProcessImage
        # because PostProcessImage may have additional filtering logic
        self.use_manual_postprocess = True

        print("✅ SAM3 + LoRA ready for inference!\n")

    def create_datapoint(
        self,
        pil_image: PILImage.Image,
        text_prompts: List[str],
        boxes: Optional[List[List[float]]] = None
    ) -> Datapoint:
        """
        Create a SAM3 datapoint from image, text prompts, and optional bbox prompts.

        Args:
            pil_image: PIL Image
            text_prompts: List of text queries
            boxes: Optional list of bounding boxes [[x1, y1, x2, y2], ...]
                   in raw pixel coordinates (XYXY format, original image size).
                   These are passed as visual/geometric prompts to the model.

        Returns:
            Datapoint with image and queries
        """
        w, h = pil_image.size

        # Create SAM Image
        sam_image = SAMImage(
            data=pil_image,
            objects=[],
            size=[h, w]
        )

        # Build bbox tensor if boxes are provided (shape: [N, 4], XYXY pixel coords)
        if boxes is not None and len(boxes) > 0:
            bbox_tensor = torch.as_tensor(boxes, dtype=torch.float32).view(-1, 4)
            # Clamp to image bounds
            bbox_tensor[:, 0::2].clamp_(min=0, max=w)
            bbox_tensor[:, 1::2].clamp_(min=0, max=h)
            bbox_label = torch.ones(len(bbox_tensor), dtype=torch.long)
        else:
            bbox_tensor = None
            bbox_label = None

        # Create queries for each text prompt
        queries = []
        for idx, text_query in enumerate(text_prompts):
            query = FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                input_bbox=bbox_tensor,
                input_bbox_label=bbox_label,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[sam_image]
        )

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        text_prompts: List[str],
        boxes: Optional[List[List[float]]] = None,
        max_points: Optional[int] = None,
        polygon_method: str = "uniform"
    ) -> dict:
        """
        Run inference on an image with text prompts and optional bbox prompts.

        Args:
            image_path: Path to input image
            text_prompts: List of text queries (e.g., ["nucleus", "pronucleus"])
            boxes: Optional list of example bounding boxes [[x1, y1, x2, y2], ...]
                   in pixel coordinates (XYXY). Used as visual reference prompts.
            max_points: Maximum number of polygon points per contour. None = no limit.
                        Uses Douglas-Peucker simplification then downsampling.

        Returns:
            Dictionary mapping prompt index to predictions:
            {
                0: {'boxes': [...], 'scores': [...], 'masks': [...]},
                1: {'boxes': [...], 'scores': [...], 'masks': [...]}
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        pil_image = PILImage.open(image_path).convert("RGB")
        print(f"📷 Loaded image: {image_path}")
        print(f"   Size: {pil_image.size}")
        print(f"   Prompts: {text_prompts}")
        if boxes is not None and len(boxes) > 0:
            print(f"   Bbox prompts: {boxes}")

        print("\n🔮 Running inference...")

        results = {}

        # Process each prompt separately (SAM3 expects one query per forward pass)
        for query_idx, prompt in enumerate(text_prompts):
            # Create datapoint with single prompt (+ shared bbox prompts)
            datapoint = self.create_datapoint(pil_image, [prompt], boxes=boxes)

            # Apply transforms
            datapoint = self.transform(datapoint)

            # Collate into batch
            batch = collate_fn_api([datapoint], dict_key="input")["input"]

            # Move to device
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(batch)

            # Manual post-processing
            last_output = outputs[-1]
            pred_logits = last_output['pred_logits']  # [batch, num_queries, num_classes]
            pred_boxes = last_output['pred_boxes']    # [batch, num_queries, 4]
            pred_masks = last_output.get('pred_masks', None)  # [batch, num_queries, H, W]

            # Get probabilities
            out_probs = pred_logits.sigmoid()  # [batch, num_queries, num_classes]

            # Get scores for this query
            scores = out_probs[0, :, :].max(dim=-1)[0]  # [num_queries]

            # Filter by threshold
            keep = scores > self.detection_threshold
            num_keep = keep.sum().item()

            if num_keep > 0:
                # Get boxes and convert from cxcywh to xyxy
                boxes_cxcywh = pred_boxes[0, keep]  # [num_keep, 4]
                kept_scores = scores[keep]
                cx, cy, w, h = boxes_cxcywh.unbind(-1)

                # Convert to xyxy and scale to original image size
                orig_w, orig_h = pil_image.size
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h

                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                # Apply NMS to remove overlapping boxes
                keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
                boxes_xyxy = boxes_xyxy[keep_nms]
                kept_scores = kept_scores[keep_nms]
                num_keep = len(keep_nms)

                # Get masks and resize to original size
                if pred_masks is not None:
                    # Apply NMS filtering to masks too
                    masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5  # [num_keep_nms, H, W]

                    # Resize masks to original image size
                    import torch.nn.functional as F
                    masks_resized = F.interpolate(
                        masks_small.unsqueeze(0).float(),
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0) > 0.5

                    masks_np = masks_resized.cpu().numpy()
                else:
                    masks_np = None

                # Extract polygon contours for each detected object
                polygons_per_object = []
                if masks_np is not None:
                    for i in range(len(masks_np)):
                        polys = self._extract_polygons(
                            masks_np[i],
                            max_points=max_points,
                            polygon_method=polygon_method
                        )
                        polygons_per_object.append(polys)
                else:
                    polygons_per_object = [[] for _ in range(num_keep)]

                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': boxes_xyxy.cpu().numpy(),
                    'scores': kept_scores.cpu().numpy(),
                    'masks': masks_np,
                    'polygons': polygons_per_object,  # List[List[polygon]] per object
                    'num_detections': num_keep
                }
                print(f"   '{prompt}': {num_keep} detections after NMS (max score: {kept_scores.max().item():.3f})")
            else:
                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': None,
                    'scores': None,
                    'masks': None,
                    'polygons': [],
                    'num_detections': 0
                }
                print(f"   '{prompt}': 0 detections")

        # Store original image for visualization
        results['_image'] = pil_image

        return results

    def _extract_polygons(
        self,
        mask: "np.ndarray",
        min_points: int = 3,
        max_points: Optional[int] = None,
        polygon_method: str = "uniform"
    ) -> List[List[List[int]]]:
        """
        Extract polygon contours from a binary mask.

        Args:
            mask: Boolean numpy array of shape [H, W].
            min_points: Minimum number of points to keep a contour (default: 3).
            max_points: Maximum number of points per contour. None = no limit.
            polygon_method: How to reduce points when max_points is set:
                - 'uniform': Uniform downsampling via np.linspace. Guarantees
                             exactly max_points output, evenly spaced around
                             the contour. Predictable but may skip sharp corners.
                - 'approx':  Douglas-Peucker shape simplification
                             (cv2.approxPolyDP). Keeps visually important corners
                             and removes redundant collinear points. Result may
                             be fewer than max_points but better shape fidelity.

        Returns:
            List of polygons, each polygon is a list of [x, y] integer points.
            Example: [[[10, 20], [30, 40], [50, 60]], ...]
        """
        try:
            import cv2
            mask_uint8 = mask.astype(np.uint8) * 255

            # Choose contour extraction method:
            # - 'approx': uses CHAIN_APPROX_NONE (all border pixels) so DP has
            #   the full set to work with.
            # - 'uniform' + max_points: uses CHAIN_APPROX_NONE for even linspace.
            # - default (no reduction needed): CHAIN_APPROX_SIMPLE is faster.
            if polygon_method == "approx" or (polygon_method == "uniform" and max_points is not None):
                find_method = cv2.CHAIN_APPROX_NONE
            else:
                find_method = cv2.CHAIN_APPROX_SIMPLE

            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, find_method
            )
            polygons = []
            for cnt in contours:
                if len(cnt) < min_points:
                    continue

                if polygon_method == "approx":
                    # --- Douglas-Peucker ---
                    # Always runs regardless of max_points. Removes redundant
                    # collinear/near-collinear points while preserving corners.
                    # Default epsilon = 0.5% of perimeter (good general smoothing).
                    perimeter = cv2.arcLength(cnt, True)
                    epsilon = 0.005 * perimeter
                    cnt = cv2.approxPolyDP(cnt, epsilon, True)

                    # If max_points is also provided, further cap the result:
                    if max_points is not None and len(cnt) > max_points:
                        # Tighten epsilon iteratively until within budget.
                        while len(cnt) > max_points and epsilon < perimeter:
                            epsilon *= 1.5
                            cnt = cv2.approxPolyDP(
                                cv2.approxPolyDP(cnt, 0, True),  # re-expand shape
                                epsilon, True
                            )

                elif polygon_method == "uniform" and max_points is not None:
                    # --- Uniform downsampling ---
                    # Only activates when max_points is explicitly given.
                    # Guarantees exactly max_points, evenly distributed around contour.
                    indices = np.round(
                        np.linspace(0, len(cnt) - 1, max_points)
                    ).astype(int)
                    cnt = cnt[indices]

                # else: default — keep all points from CHAIN_APPROX_SIMPLE as-is.

                poly = cnt.squeeze(axis=1).tolist()  # [[x, y], ...]
                if len(poly) >= min_points:
                    polygons.append(poly)
            return polygons
        except ImportError:
            # Fallback: extract all True pixel coordinates (no OpenCV)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return []
            points = np.stack([xs, ys], axis=-1).tolist()
            # Hard-cap with uniform downsampling if needed
            if max_points is not None and len(points) > max_points:
                step = max(1, len(points) // max_points)
                points = points[::step]
            return [points]

    def visualize(
        self,
        results: dict,
        output_path: str,
        show_boxes: bool = True,
        show_masks: bool = True,
        input_boxes: Optional[List[List[float]]] = None,
        show_input_boxes: bool = False,
        show_polygons: bool = False
    ):
        """
        Visualize predictions on image.

        Args:
            results: Results from predict()
            output_path: Where to save visualization
            show_boxes: Whether to show prediction bounding boxes
            show_masks: Whether to show segmentation masks
            input_boxes: Original bbox prompts [[x1, y1, x2, y2], ...] in pixel coords
            show_input_boxes: Whether to draw the input bbox prompts on the output image
            show_polygons: Whether to draw polygon contour outlines and vertex dots
        """
        pil_image = results['_image']

        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(pil_image)

        # Colors for different prompts
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

        total_detections = 0

        # Draw input bbox prompts (white dashed) if requested
        if show_input_boxes and input_boxes is not None and len(input_boxes) > 0:
            img_w, img_h = pil_image.size
            for b_idx, box in enumerate(input_boxes):
                x1, y1, x2, y2 = box
                x1 = max(0, min(img_w, x1))
                y1 = max(0, min(img_h, y1))
                x2 = max(0, min(img_w, x2))
                y2 = max(0, min(img_h, y2))
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2,
                    edgecolor='white',
                    facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 5,
                    f"bbox prompt #{b_idx + 1}",
                    bbox=dict(facecolor='black', alpha=0.5),
                    fontsize=9,
                    color='white'
                )

        # Draw results for each prompt
        for idx in sorted([k for k in results.keys() if k != '_image']):
            result = results[idx]
            prompt = result['prompt']
            color = colors[idx % len(colors)]

            if result['num_detections'] == 0:
                continue

            total_detections += result['num_detections']

            boxes = result['boxes']
            scores = result['scores']
            masks = result['masks']

            for i in range(result['num_detections']):
                # Draw mask
                if show_masks and masks is not None:
                    mask = masks[i]
                    colored_mask = np.zeros((*mask.shape, 4))
                    # Use different colors for different prompts
                    if color == 'red':
                        colored_mask[mask] = [1, 0, 0, 0.4]
                    elif color == 'blue':
                        colored_mask[mask] = [0, 0, 1, 0.4]
                    elif color == 'green':
                        colored_mask[mask] = [0, 1, 0, 0.4]
                    else:
                        colored_mask[mask] = [1, 1, 0, 0.4]
                    ax.imshow(colored_mask)

                # Draw box
                if show_boxes and boxes is not None:
                    box = boxes[i]  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box

                    # Clamp to image bounds
                    img_w, img_h = pil_image.size
                    x1 = max(0, min(img_w, x1))
                    y1 = max(0, min(img_h, y1))
                    x2 = max(0, min(img_w, x2))
                    y2 = max(0, min(img_h, y2))

                    width = x2 - x1
                    height = y2 - y1

                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add label
                    score = scores[i] if scores is not None else 0
                    label = f"{prompt}: {score:.2f}"
                    ax.text(
                        x1, y1 - 5,
                        label,
                        bbox=dict(facecolor=color, alpha=0.5),
                        fontsize=10,
                        color='white'
                    )

                # Draw polygon contours and vertex dots
                if show_polygons and result.get('polygons') and i < len(result['polygons']):
                    object_polygons = result['polygons'][i]  # List of contours for this object
                    for poly in object_polygons:
                        if len(poly) < 3:
                            continue
                        xy = np.array(poly, dtype=float)  # [[x, y], ...]
                        # Draw filled polygon outline (closed, semi-transparent border)
                        poly_patch = patches.Polygon(
                            xy,
                            closed=True,
                            edgecolor=color,
                            facecolor='none',
                            linewidth=1.5,
                            linestyle='-',
                            alpha=0.9
                        )
                        ax.add_patch(poly_patch)
                        # Draw vertex dots (every few points to avoid clutter)
                        step = max(1, len(xy) // 40)  # at most ~40 dots per contour
                        sampled = xy[::step]
                        ax.scatter(
                            sampled[:, 0], sampled[:, 1],
                            s=6, c=color, marker='o',
                            linewidths=0, alpha=0.85,
                            zorder=5
                        )

        ax.axis('off')

        # Add title with all prompts
        prompts_str = ", ".join([f'"{results[k]["prompt"]}"' for k in sorted([k for k in results.keys() if k != '_image'])])
        plt.suptitle(f'Text Prompts: {prompts_str}', fontsize=12, y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"\n✅ Saved visualization to {output_path}")
        print(f"   Total detections: {total_detections}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 + LoRA Inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights (auto-detected if not provided)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs='+',
        default=["object"],
        help='Text prompt(s) to guide segmentation (e.g., "crack" or "crack" "defect")'
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output visualization path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution (default: 1008)"
    )
    parser.add_argument(
        "--boundingbox",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help="Show bounding boxes: True or False (default: False)"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Don't show segmentation masks"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5, lower = fewer overlapping boxes)"
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        action='append',
        default=None,
        dest='boxes',
        help=(
            "Example bounding box prompt in pixel coords (XYXY): --box x1 y1 x2 y2. "
            "Can be repeated for multiple boxes, e.g.: --box 50 60 150 160 --box 200 220 320 350. "
            "Boxes are shared across all --prompt values and guide the model geometrically."
        )
    )

    parser.add_argument(
        "--show-input-boxes",
        action="store_true",
        default=False,
        help="Draw the input bbox prompts (--box) on the output image as white dashed rectangles"
    )
    parser.add_argument(
        "--show-polygons",
        action="store_true",
        default=False,
        help="Draw polygon contour outlines and vertex dots for each detected object on the output image"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of polygon points per contour. "
            "Example: --max-points 100. Default: no limit."
        )
    )
    parser.add_argument(
        "--polygon-method",
        type=str,
        default="uniform",
        choices=["uniform", "approx"],
        help=(
            "Method to reduce polygon points when --max-points is set. "
            "'uniform': Uniform downsampling — guarantees exactly max_points, evenly spaced (default). "
            "'approx': Douglas-Peucker — preserves shape-critical corners, result may be fewer than max_points."
        )
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Save detection results (bbox, score, polygon coords) to a JSON file. "
            "Example: --save-json results/output.json"
        )
    )

    args = parser.parse_args()

    # Initialize model
    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou
    )

    # Run inference
    results = inferencer.predict(
        args.image, args.prompt,
        boxes=args.boxes,
        max_points=args.max_points,
        polygon_method=args.polygon_method
    )

    # Visualize
    inferencer.visualize(
        results,
        args.output,
        show_boxes=args.boundingbox,
        show_masks=not args.no_masks,
        input_boxes=args.boxes,
        show_input_boxes=args.show_input_boxes,
        show_polygons=args.show_polygons
    )

    # Save JSON output if requested
    if args.save_json:
        json_data = {
            "image": args.image,
            "prompts": args.prompt,
            "results": []
        }
        for idx in sorted([k for k in results.keys() if k != '_image']):
            result = results[idx]
            entry = {
                "prompt": result['prompt'],
                "num_detections": result['num_detections'],
                "objects": []
            }
            if result['num_detections'] > 0 and result['boxes'] is not None:
                for i in range(result['num_detections']):
                    box = result['boxes'][i].tolist()
                    score = float(result['scores'][i])
                    polygons = result['polygons'][i] if result['polygons'] else []
                    entry["objects"].append({
                        "object_id": i,
                        "score": round(score, 4),
                        "bbox_xyxy": [round(v, 1) for v in box],
                        "polygons": polygons  # List of contours, each [[x, y], ...]
                    })
            json_data["results"].append(entry)

        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\n💾 Saved JSON results to: {args.save_json}")

    # Print summary
    print("\n" + "="*60)
    print("📊 Summary:")
    for idx in sorted([k for k in results.keys() if k != '_image']):
        result = results[idx]
        print(f"   Prompt '{result['prompt']}': {result['num_detections']} detections")
        if result['num_detections'] > 0 and result['scores'] is not None:
            print(f"      Max confidence: {result['scores'].max():.3f}")
        if result['num_detections'] > 0 and result.get('polygons'):
            total_polys = sum(len(p) for p in result['polygons'])
            print(f"      Polygon contours extracted: {total_polys}")
    print("="*60)


if __name__ == "__main__":
    main()
