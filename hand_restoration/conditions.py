from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ConditionConfig:
    """Image-space corruption settings; all radii are in output pixels."""

    mode: str = "masked_replace"
    # Together these produce an approximately 8 px outer edit band by default.
    mask_dilation_px: int = 2
    boundary_corruption_px: int = 6
    wrist_extension_px: int = 0
    mano_opacity: float = 1.0
    background_fill: str = "gray"  # gray, black, noise


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(np.float32)


def _background(shape: tuple[int, int, int], strategy: str, rng: np.random.Generator) -> np.ndarray:
    if strategy == "black":
        return np.zeros(shape, dtype=np.float32)
    if strategy == "noise":
        return rng.uniform(0.35, 0.65, size=shape).astype(np.float32)
    if strategy != "gray":
        raise ValueError(f"Unknown background_fill '{strategy}'. Use gray, black, or noise.")
    return np.full(shape, 0.5, dtype=np.float32)


class ConditionBuilder:
    """Build ControlNet's three-channel condition and its localized edit mask.

    Inputs are HWC float32 RGB in [0, 1], and a HW mask in [0, 1]. Outputs
    preserve that convention. The replacement path never retains target pixels
    inside ``edit_mask``.
    """

    def __init__(self, config: ConditionConfig, seed: int = 0) -> None:
        if config.mode not in {"overlay", "masked_replace", "mano_only"}:
            raise ValueError(f"Unsupported condition mode: {config.mode}")
        self.config = config
        self.rng = np.random.default_rng(seed)

    def __call__(self, target_rgb: np.ndarray, mano_rgb: np.ndarray, mano_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if target_rgb.shape != mano_rgb.shape or target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
            raise ValueError("target_rgb and mano_rgb must be matching HWC RGB images.")
        if mano_mask.shape != target_rgb.shape[:2]:
            raise ValueError("mano_mask must have the target image height and width.")

        mask = np.clip(mano_mask.astype(np.float32), 0.0, 1.0)
        replacement = _dilate(mask, self.config.mask_dilation_px)
        edit_mask = _dilate(replacement, self.config.boundary_corruption_px)

        # In upright C1 frames the wrist is normally the lowest hand extent.
        # Extending just this local band teaches a clean hand-to-arm transition.
        if self.config.wrist_extension_px > 0 and np.any(replacement > 0):
            ys, xs = np.where(replacement > 0)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            band_top = max(y0, y1 - max(4, (y1 - y0 + 1) // 5))
            band_bottom = min(target_rgb.shape[0], y1 + self.config.wrist_extension_px + 1)
            pad = self.config.boundary_corruption_px
            edit_mask[max(0, band_top - pad):band_bottom, max(0, x0 - pad):min(target_rgb.shape[1], x1 + pad + 1)] = 1.0

        alpha = np.clip(mask * self.config.mano_opacity, 0.0, 1.0)[..., None]
        if self.config.mode == "overlay":
            condition = target_rgb * (1.0 - alpha) + mano_rgb * alpha
        elif self.config.mode == "mano_only":
            condition = _background(target_rgb.shape, self.config.background_fill, self.rng)
            condition = condition * (1.0 - alpha) + mano_rgb * alpha
        else:
            # Blank all edited pixels first, then place only the coarse mesh.
            condition = target_rgb.copy()
            fill = _background(target_rgb.shape, self.config.background_fill, self.rng)
            condition[edit_mask > 0] = fill[edit_mask > 0]
            condition = condition * (1.0 - alpha) + mano_rgb * alpha

        return np.clip(condition, 0.0, 1.0).astype(np.float32), np.clip(edit_mask, 0.0, 1.0).astype(np.float32)
