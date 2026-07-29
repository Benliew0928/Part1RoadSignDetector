from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

MODULE_OWNER = "LJ"
MODULE_TITLE = "Shape detection of signs to support the segmentation"

SUPPORTED_SIGN_SHAPES = frozenset(
    {"circle", "triangle", "square_or_rectangle", "octagon"}
)


def classify_shape(
    vertices: int | Sequence[int],
    circularity: float,
    aspect_ratio: float,
    *,
    triangle_fit: float = 0.0,
    circle_min_circularity: float = 0.75,
    near_square_min: float = 0.75,
    near_square_max: float = 1.33,
    triangle_min_fit: float = 0.84,
    triangle_max_circularity: float = 0.88,
    triangle_min_vertex_votes: int = 2,
    rectangle_min_vertex_votes: int = 2,
    rectangle_max_triangle_fit: float = 0.70,
    octagon_min_vertex_votes: int = 3,
) -> str:
    """Classify one filled colour silhouette with conventional geometry.

    Several nearby ``approxPolyDP`` epsilons are used instead of relying on a
    single polygon.  This keeps small rasterisation changes from turning a
    triangle into a six-sided contour.  ``triangle_fit`` is the ratio between
    the convex-hull area and its minimum enclosing triangle area; it is a
    contour measurement, not a template or learned classifier.
    """
    if isinstance(vertices, int):
        vertex_counts = (vertices,)
    else:
        vertex_counts = tuple(int(value) for value in vertices)
    if not vertex_counts:
        return "other"

    triangle_votes = sum(value == 3 for value in vertex_counts)
    rectangle_votes = sum(value == 4 for value in vertex_counts)
    octagon_votes = sum(value in {7, 8, 9} for value in vertex_counts)

    # Triangle is checked before circle. A warning sign often has a rounded
    # base or a small clipped tip, so its best polygon is not always exactly
    # three-sided; the enclosing-triangle fit protects this case.
    if triangle_votes >= triangle_min_vertex_votes or (
        triangle_fit >= triangle_min_fit
        and circularity <= triangle_max_circularity
        and near_square_min <= aspect_ratio <= near_square_max
    ):
        return "triangle"

    # A rectangle has a stable four-corner polygon and fits its enclosing
    # triangle much less tightly than a true triangular sign.
    if (
        rectangle_votes >= rectangle_min_vertex_votes
        and triangle_fit <= rectangle_max_triangle_fit
    ):
        return "square_or_rectangle"

    # A regular octagon keeps exactly eight hull corners across all sampled
    # epsilons. Circular contours normally lose corners as epsilon increases,
    # so they do not meet this stability requirement.
    if octagon_votes >= octagon_min_vertex_votes and all(
        value == 8 for value in vertex_counts
    ):
        return "octagon"

    if (
        circularity >= circle_min_circularity
        and near_square_min <= aspect_ratio <= near_square_max
    ):
        return "circle"
    return "other"


def representative_vertex_count(vertex_counts: Sequence[int]) -> int:
    """Return the median vote for concise dashboard and CSV display."""
    if not vertex_counts:
        return 0
    ordered = sorted(int(value) for value in vertex_counts)
    return ordered[len(ordered) // 2]


def is_supported_sign_shape(shape_label: str) -> bool:
    return shape_label in SUPPORTED_SIGN_SHAPES


def score_candidate(
    *,
    solidity: float,
    extent: float,
    circularity: float,
    aspect_ratio: float,
    shape_label: str,
    area_ratio: float,
    shape_confidence: float = 0.0,
) -> float:
    """Provide a display confidence from basic geometry only.

    This score is not a trained probability and is not used to select the
    largest primary contour.  It exists only so the dashboard can display the
    strength of the measured geometry consistently across candidates.
    """
    del area_ratio
    shape_support = 1.0 if shape_label in SUPPORTED_SIGN_SHAPES else 0.0
    aspect_score = max(0.0, 1.0 - abs(math.log(max(aspect_ratio, 1e-6))) / 2.0)
    return float(
        np.clip(
            0.28 * solidity
            + 0.22 * extent
            + 0.15 * min(1.0, circularity)
            + 0.10 * aspect_score
            + 0.10 * shape_support
            + 0.15 * shape_confidence,
            0.0,
            1.0,
        )
    )
