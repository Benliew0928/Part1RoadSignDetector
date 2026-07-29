from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

MODULE_OWNER = "LJ"
MODULE_TITLE = "Shape detection of signs to support the segmentation"

SUPPORTED_SIGN_SHAPES = frozenset(
    {"circle", "triangle", "square_or_rectangle", "octagon"}
)

_SHAPE_PRIORITY = {
    "octagon": 4,
    "triangle": 3,
    "square_or_rectangle": 2,
    "circle": 1,
}


def shape_fit_scores(
    vertices: int | Sequence[int],
    circularity: float,
    *,
    circle_fit: float,
    ellipse_fit: float,
    ellipse_axis_ratio: float,
    triangle_fit: float,
    rectangle_fit: float,
    octagon_fit: float,
    circle_min_circularity: float = 0.75,
    triangle_max_circularity: float = 0.88,
    rectangle_max_triangle_fit: float = 0.70,
    perspective_ellipse_min_axis_ratio: float = 0.55,
) -> dict[str, float]:
    """Return global, explainable evidence scores for the four sign shapes.

    The fit values are contour-to-model overlaps measured in the shared
    candidate pipeline.  This function only combines those measurements with
    conventional polygon and circularity evidence; it has no image-specific
    rule, stored template image, or learned model.
    """
    if isinstance(vertices, int):
        vertex_counts = (vertices,)
    else:
        vertex_counts = tuple(int(value) for value in vertices)
    if not vertex_counts:
        return {shape: 0.0 for shape in SUPPORTED_SIGN_SHAPES}

    total = len(vertex_counts)
    triangle_votes = sum(value == 3 for value in vertex_counts) / total
    rectangle_votes = sum(value == 4 for value in vertex_counts) / total
    octagon_votes = sum(value == 8 for value in vertex_counts) / total
    # Smooth circles often receive an eight-sided approximation at one small
    # epsilon. A real octagon must retain eight corners at every sampled
    # epsilon before it can compete with the circle/ellipse fit.
    octagon_stable = float(all(value == 8 for value in vertex_counts))

    # A circle viewed obliquely becomes an ellipse. The axis-ratio guard keeps
    # extremely narrow ellipses from turning arbitrary long regions into a
    # circular sign candidate.
    perspective_circle_fit = (
        ellipse_fit
        if ellipse_axis_ratio >= perspective_ellipse_min_axis_ratio
        else 0.0
    )
    circle_geometry = max(circle_fit, perspective_circle_fit)
    circularity_score = min(1.0, circularity / max(1e-6, circle_min_circularity))

    # A circle and an octagon can both fit a circle closely. The octagon score
    # retains its independent stable-corner evidence so it wins when that
    # evidence is stronger.
    triangle_geometry = triangle_fit
    if circularity > triangle_max_circularity:
        triangle_geometry *= 0.50

    rectangle_geometry = rectangle_fit
    if triangle_fit > rectangle_max_triangle_fit:
        rectangle_geometry *= 0.50

    return {
        "circle": float(np.clip(0.60 * circle_geometry + 0.40 * circularity_score, 0.0, 1.0)),
        "triangle": float(np.clip(0.85 * triangle_geometry + 0.15 * triangle_votes, 0.0, 1.0)),
        "square_or_rectangle": float(
            np.clip(0.85 * rectangle_geometry + 0.15 * rectangle_votes, 0.0, 1.0)
        ),
        "octagon": float(
            np.clip(
                octagon_stable * (0.70 * octagon_fit + 0.20 * octagon_votes + 0.10),
                0.0,
                1.0,
            )
        ),
    }


def choose_shape(scores: Mapping[str, float], *, minimum_score: float) -> str:
    """Return the strongest globally supported shape, otherwise ``other``."""
    if not scores:
        return "other"
    best_shape = max(
        SUPPORTED_SIGN_SHAPES,
        key=lambda shape: (float(scores.get(shape, 0.0)), _SHAPE_PRIORITY[shape]),
    )
    return best_shape if float(scores.get(best_shape, 0.0)) >= minimum_score else "other"


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
    """Provide a display confidence from global geometric evidence only."""
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
