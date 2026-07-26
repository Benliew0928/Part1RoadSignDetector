from __future__ import annotations

from typing import Any, Mapping, cast

import cv2
import numpy as np

from roadsign_assist.baseline.models import UInt8Image

MODULE_OWNER = "JY"
MODULE_TITLE = "Yellow sign segmentation using color information"
COLOR_NAME = "yellow"


def segment_yellow_signs(hsv_image: UInt8Image, color_settings: Mapping[str, Any]) -> UInt8Image:
    """Create the yellow traffic-sign mask from configured HSV yellow ranges."""
    combined = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for color_range in cast(list[Mapping[str, Any]], color_settings["ranges"]):
        lower = np.asarray(color_range["lower"], dtype=np.uint8)
        upper = np.asarray(color_range["upper"], dtype=np.uint8)
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv_image, lower, upper))
    return cast(UInt8Image, combined)
