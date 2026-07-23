from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def rgb_float_to_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def save_debug_grid(sample: dict, path: str | Path) -> None:
    """Save target, render, masks, and condition as a labelled 2x3 RGB grid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = sample["target_rgb_np"]
    render = sample["mano_rgb_np"]
    mask = sample["mano_mask_np"]
    edit = sample["edit_mask_np"]
    condition = sample["condition_rgb_np"]
    mask_rgb = np.repeat(mask[..., None], 3, axis=2)
    edit_rgb = np.repeat(edit[..., None], 3, axis=2)
    panels = [("target", target), ("mano render", render), ("mano mask", mask_rgb), ("edit mask", edit_rgb), ("condition", condition)]
    height, width = target.shape[:2]
    metadata_panel = np.zeros((height, width, 3), dtype=np.float32)
    panels.append(("metadata", metadata_panel))
    rendered = []
    for label, panel in panels:
        item = rgb_float_to_u8(panel)
        cv2.putText(item, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 0), 2, cv2.LINE_AA)
        if label == "metadata":
            metadata = sample.get("metadata", {})
            source_size = metadata.get("original_image_size", ["?", "?"])
            lines = [
                f"sequence: {metadata.get('sequence_id', '?')}",
                f"frame: {metadata.get('frame_id', '?')}",
                f"camera: {metadata.get('camera_id', '?')}",
                f"hand: {metadata.get('handedness', '?')}",
                f"source HxW: {source_size[0]}x{source_size[1]}",
                f"output HxW: {height}x{width}",
            ]
            for line_index, line in enumerate(lines):
                cv2.putText(item, line, (12, 72 + line_index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)
        rendered.append(item)
    grid = np.concatenate((np.concatenate(rendered[:3], axis=1), np.concatenate(rendered[3:], axis=1)), axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
