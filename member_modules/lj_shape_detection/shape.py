from __future__ import annotations

import math

import numpy as np

MODULE_OWNER = "LJ"
MODULE_TITLE = "Shape detection of signs to support the segmentation"


def classify_shape(vertices: int, circularity: float, aspect_ratio: float) -> str:
    """Classify supported traffic-sign shapes from contour geometry."""
    if circularity >= 0.76 and 0.72 <= aspect_ratio <= 1.38:
        return "circle"
    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "square_or_rectangle"
    if 7 <= vertices <= 10:
        return "octagon"
    return "other"


def score_candidate(
    *,
    solidity: float,
    extent: float,
    circularity: float,
    aspect_ratio: float,
    shape_label: str,
    area_ratio: float,
) -> float:
    """Score a color candidate using shape and geometry evidence."""
    shape_bonus = {
        "circle": 0.15,
        "triangle": 0.15,
        "square_or_rectangle": 0.10,
        "octagon": 0.18,
        "other": 0.0,
    }[shape_label]
    aspect_score = max(0.0, 1.0 - abs(math.log(max(aspect_ratio, 1e-6))) / 2.0)
    area_score = min(1.0, area_ratio / 0.02)
    return float(
        np.clip(
            0.26 * solidity
            + 0.20 * extent
            + 0.18 * circularity
            + 0.12 * aspect_score
            + 0.09 * area_score
            + shape_bonus,
            0.0,
            1.0,
        )
    )
