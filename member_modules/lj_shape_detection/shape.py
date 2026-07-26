from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

MODULE_OWNER = "LJ"
MODULE_TITLE = "Shape detection of signs to support the segmentation"

# The coursework only treats these geometries as traffic-sign shapes.  Keeping
# this rule with the shape module stops arbitrary colour blobs becoming final
# candidates in the shared pipeline.
SUPPORTED_SIGN_SHAPES = frozenset(
    {"circle", "triangle", "square_or_rectangle", "octagon"}
)

Contour = NDArray[np.int32]


@dataclass(frozen=True)
class ShapeEvidence:
    """Shape measurements calculated from a colour-segmented contour."""

    label: str
    confidence: float
    triangle_similarity: float
    circle_similarity: float
    radial_variation: float


def _regular_polygon(sides: int, radius: float = 100.0) -> Contour:
    """Build a template contour for OpenCV's scale/rotation invariant matcher."""
    points = [
        (
            120.0 + radius * math.cos(-math.pi / 2.0 + 2.0 * math.pi * index / sides),
            120.0 + radius * math.sin(-math.pi / 2.0 + 2.0 * math.pi * index / sides),
        )
        for index in range(sides)
    ]
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


_CIRCLE_TEMPLATE = _regular_polygon(96)
_TRIANGLE_TEMPLATE = _regular_polygon(3)


def _polygon_residual(contour: Contour, polygon: Contour, radius: float) -> float:
    """Return the contour-to-polygon distance relative to its enclosing radius."""
    if radius <= 0.0:
        return 1.0
    distances = [
        abs(cv2.pointPolygonTest(polygon, (float(point[0][0]), float(point[0][1])), True))
        for point in contour
    ]
    return float(np.mean(distances) / radius) if distances else 1.0


def analyze_contour_shape(
    contour: Contour,
    *,
    circularity: float,
    aspect_ratio: float,
) -> ShapeEvidence:
    """Classify a sign contour while tolerating incomplete colour masks.

    Colour masks from real road scenes can be interrupted by glare, symbols, a
    pole, or perspective. Looking at the convex hull at several approximation
    scales is more reliable than making a decision from one contour vertex
    count. ``matchShapes`` is used only as a tie-breaker for triangular signs.
    """
    hull = cv2.convexHull(contour)
    hull_perimeter = float(cv2.arcLength(hull, True))
    if hull_perimeter <= 0.0:
        return ShapeEvidence("other", 0.0, 1.0, 1.0, 1.0)

    fine = cv2.approxPolyDP(hull, 0.012 * hull_perimeter, True)
    medium = cv2.approxPolyDP(hull, 0.025 * hull_perimeter, True)
    coarse = cv2.approxPolyDP(hull, 0.070 * hull_perimeter, True)
    fine_vertices = len(fine)
    medium_vertices = len(medium)
    coarse_vertices = len(coarse)

    (center_x, center_y), radius = cv2.minEnclosingCircle(hull)
    hull_points = hull.reshape((-1, 2)).astype(np.float32)
    if radius > 0.0 and hull_points.size:
        normalized_radii = np.hypot(
            hull_points[:, 0] - center_x,
            hull_points[:, 1] - center_y,
        ) / radius
        radial_variation = float(np.std(normalized_radii))
    else:
        radial_variation = 1.0

    triangle_similarity = float(
        cv2.matchShapes(contour, _TRIANGLE_TEMPLATE, cv2.CONTOURS_MATCH_I1, 0.0)
    )
    circle_similarity = float(
        cv2.matchShapes(contour, _CIRCLE_TEMPLATE, cv2.CONTOURS_MATCH_I1, 0.0)
    )

    # A true octagon has straight sides that closely fit its eight-corner
    # polygon. A circle also often approximates to eight points, but has a
    # substantially larger residual because its boundary remains curved.
    octagon_residual = _polygon_residual(contour, medium, radius)
    if medium_vertices == 8 and octagon_residual <= 0.020:
        return ShapeEvidence(
            "octagon",
            0.95,
            triangle_similarity,
            circle_similarity,
            radial_variation,
        )

    # A rectangle stays four-sided at a fine approximation. The slightly
    # looser second condition keeps perspective-skewed rectangular signs while
    # avoiding yellow triangular masks whose corners were softened by blur.
    rectangle_like = (
        fine_vertices == 4 and radial_variation <= 0.110
    ) or (
        medium_vertices == 4
        and radial_variation <= 0.095
        and triangle_similarity >= circle_similarity * 0.95
    )
    if rectangle_like:
        return ShapeEvidence(
            "square_or_rectangle",
            0.88,
            triangle_similarity,
            circle_similarity,
            radial_variation,
        )

    triangle_by_polygon = (
        coarse_vertices == 3 and triangle_similarity < circle_similarity * 0.80
    )
    if triangle_by_polygon:
        return ShapeEvidence(
            "triangle",
            0.90,
            triangle_similarity,
            circle_similarity,
            radial_variation,
        )

    # Circular signs can be skewed or partly occluded, so a weak circularity
    # score is not a reason to reject them. The absence of a stable polygon is
    # useful circle evidence after the checks above.
    circle_confidence = float(
        np.clip(
            0.50
            + 0.30 * min(1.0, circularity / 0.80)
            + 0.12 * max(0.0, 1.0 - radial_variation / 0.18)
            + 0.08 * max(0.0, 1.0 - abs(math.log(max(aspect_ratio, 1e-6))) / 1.5),
            0.35,
            0.95,
        )
    )
    return ShapeEvidence(
        "circle",
        circle_confidence,
        triangle_similarity,
        circle_similarity,
        radial_variation,
    )


def classify_shape(vertices: int, circularity: float, aspect_ratio: float) -> str:
    """Classify a simple polygon when a contour is not available.

    ``analyze_contour_shape`` is the preferred runtime method. This function
    remains as a small public helper for coursework members who only have basic
    polygon measurements.
    """
    if circularity >= 0.76 and 0.72 <= aspect_ratio <= 1.38:
        return "circle"
    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "square_or_rectangle"
    if 7 <= vertices <= 10:
        return "octagon"
    return "other"


def is_supported_sign_shape(shape_label: str) -> bool:
    """Return whether a contour has one of the coursework sign shapes."""
    return shape_label in SUPPORTED_SIGN_SHAPES


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
    # Keep scale as a soft preference. It helps an outer sign candidate win
    # over a tiny coloured icon inside it without excluding distant signs.
    area_score = min(1.0, math.sqrt(area_ratio / 0.10))
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
