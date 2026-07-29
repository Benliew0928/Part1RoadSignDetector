from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import cv2

from roadsign_assist.baseline.member_modules.ben_red_sign_segmentation import segment_red_signs
from roadsign_assist.baseline.member_modules.jy_yellow_sign_segmentation import (
    segment_yellow_signs,
)
from roadsign_assist.baseline.member_modules.mj_blue_sign_segmentation import segment_blue_signs
from roadsign_assist.baseline.models import UInt8Image

ColorSegmenter = Callable[[UInt8Image, Mapping[str, Any]], UInt8Image]

COLOR_SEGMENTERS: tuple[tuple[str, ColorSegmenter], ...] = (
    ("red", segment_red_signs),
    ("blue", segment_blue_signs),
    ("yellow", segment_yellow_signs),
)


def preprocess_bgr(image: UInt8Image, config: dict[str, Any]) -> UInt8Image:
    """Apply only an explicitly enabled small blur before HSV conversion."""
    blur_size = int(config["preprocessing"].get("gaussian_blur_size", 0))
    if blur_size < 3:
        return image.copy()
    if blur_size % 2 == 0:
        blur_size += 1
    return cast(UInt8Image, cv2.GaussianBlur(image, (blur_size, blur_size), 0))


def _morphology(mask: UInt8Image, *, color: str, config: dict[str, Any]) -> UInt8Image:
    settings = config["morphology"]
    iterations = int(settings["iterations"])
    open_size = int(settings["open_kernel"])
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel, iterations=iterations)

    # The supplied guide recommends a closing operation for red only, after
    # opening, because red sign borders are frequently interrupted by white.
    if color == "red":
        close_size = int(settings["red_close_kernel"])
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, closing_kernel, iterations=iterations)
    return cast(UInt8Image, result)


def segment_color_stages(
    image: UInt8Image,
    config: dict[str, Any],
) -> tuple[dict[str, UInt8Image], dict[str, UInt8Image]]:
    """Return raw HSV masks and their cleaned equivalents for the dashboard."""
    processed = preprocess_bgr(image, config)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    raw_masks: dict[str, UInt8Image] = {}
    clean_masks: dict[str, UInt8Image] = {}
    color_configs = cast(dict[str, Mapping[str, Any]], config["colors"])
    for color, segmenter in COLOR_SEGMENTERS:
        if color not in color_configs:
            continue
        raw_mask = segmenter(cast(UInt8Image, hsv), color_configs[color])
        raw_masks[color] = cast(UInt8Image, raw_mask)
        clean_masks[color] = _morphology(raw_mask, color=color, config=config)
    return raw_masks, clean_masks


def segment_colors(image: UInt8Image, config: dict[str, Any]) -> dict[str, UInt8Image]:
    """Compatibility helper returning only the cleaned HSV masks."""
    _, clean_masks = segment_color_stages(image, config)
    return clean_masks
