from __future__ import annotations

import math
from typing import Any, cast

import cv2
import numpy as np

from roadsign_assist.baseline.member_modules.lj_shape_detection import (
    analyze_contour_shape,
    score_candidate,
)
from roadsign_assist.baseline.models import BoundingBox, Candidate, Contour, UInt8Image


def _candidate_score(
    *,
    solidity: float,
    extent: float,
    circularity: float,
    aspect_ratio: float,
    shape_label: str,
    area_ratio: float,
) -> float:
    return score_candidate(
        solidity=solidity,
        extent=extent,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        shape_label=shape_label,
        area_ratio=area_ratio,
    )


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _visual_evidence(
    gray_image: UInt8Image,
    edges: UInt8Image,
    contour: Contour,
    *,
    edge_band_width: int,
) -> tuple[float, float, float]:
    """Measure simple visual cues that distinguish signs from colour blobs.

    Road signs normally have a distinct outer boundary plus symbols, lettering,
    or a contrasting inner region.  These inexpensive OpenCV checks are used
    alongside colour and shape; they do not require a trained model.
    """
    contour_mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    kernel_size = max(3, edge_band_width * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    boundary_band = cv2.subtract(
        cv2.dilate(contour_mask, kernel), cv2.erode(contour_mask, kernel)
    )
    boundary_pixels = max(1, cv2.countNonZero(boundary_band))
    boundary_edge_ratio = cv2.countNonZero(cv2.bitwise_and(edges, boundary_band)) / boundary_pixels

    inner_mask = cv2.erode(contour_mask, kernel)
    if cv2.countNonZero(inner_mask) == 0:
        inner_mask = contour_mask
    inner_pixels = max(1, cv2.countNonZero(inner_mask))
    inner_edge_density = cv2.countNonZero(cv2.bitwise_and(edges, inner_mask)) / inner_pixels

    values = gray_image[contour_mask > 0]
    inner_contrast = float(np.std(values)) if values.size else 0.0
    return float(boundary_edge_ratio), float(inner_edge_density), inner_contrast


def _visual_score(
    *,
    boundary_edge_ratio: float,
    inner_edge_density: float,
    inner_contrast: float,
    settings: dict[str, Any],
) -> float:
    return (
        0.50
        * _clip01(boundary_edge_ratio / float(settings["target_boundary_edge_ratio"]))
        + 0.25 * _clip01(inner_edge_density / float(settings["target_inner_edge_density"]))
        + 0.25 * _clip01(inner_contrast / float(settings["target_inner_contrast"]))
    )


def _size_score(area_ratio: float) -> float:
    """Prefer an outer sign region over small coloured details, softly."""
    return _clip01(math.sqrt(area_ratio / 0.12))


def _contained_by_larger(candidate: Candidate, other: Candidate) -> bool:
    """Return whether ``candidate`` is a small detail inside another sign.

    One colour mask can contain an outer sign region plus smaller disconnected
    details. Such a detail should not become the first dashboard result when a
    larger, plausible region of the same colour surrounds it.
    """
    if other.bbox.area < candidate.bbox.area * 1.7:
        return False
    overlap_width = max(
        0,
        min(candidate.bbox.x2, other.bbox.x2) - max(candidate.bbox.x, other.bbox.x),
    )
    overlap_height = max(
        0,
        min(candidate.bbox.y2, other.bbox.y2) - max(candidate.bbox.y, other.bbox.y),
    )
    overlap_of_candidate = overlap_width * overlap_height / candidate.bbox.area
    return (
        candidate.color == other.color
        and overlap_of_candidate >= 0.82
        and other.shape_label != "other"
    )


def extract_candidates(
    masks: dict[str, UInt8Image],
    image: UInt8Image,
    config: dict[str, Any],
) -> list[Candidate]:
    settings = config["candidates"]
    verification = config["verification"]
    image_height, image_width = image.shape[:2]
    image_area = image_width * image_height
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray_image,
        int(verification["canny_lower_threshold"]),
        int(verification["canny_upper_threshold"]),
    )
    candidates: list[Candidate] = []

    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for raw_contour in contours:
            contour = cast(Contour, raw_contour)
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            area_ratio = area / image_area
            x, y, width, height = cv2.boundingRect(contour)
            if (
                area_ratio < float(settings["min_area_ratio"])
                or area_ratio > float(settings["max_area_ratio"])
                or width < int(settings["min_width"])
                or height < int(settings["min_height"])
            ):
                continue

            bbox_area = max(1, width * height)
            extent = area / bbox_area
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area else 0.0
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
            approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            vertices = len(approximation)
            aspect_ratio = width / max(1, height)
            if (
                extent < float(settings["min_extent"])
                or solidity < float(settings["min_solidity"])
                or aspect_ratio > float(settings["max_aspect_ratio"])
                or aspect_ratio < 1.0 / float(settings["max_aspect_ratio"])
            ):
                continue

            shape_evidence = analyze_contour_shape(
                contour,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
            )
            shape_label = shape_evidence.label
            shape_confidence = shape_evidence.confidence
            # Yellow warning signs are commonly triangular. When glare or a
            # low-resolution mask has softened all three corners, use the
            # colour-and-geometry combination to recover that shape. A stable
            # rectangle remains a rectangle above.
            if (
                color == "yellow"
                and shape_label == "circle"
                and circularity < 0.75
                and area_ratio >= 0.0003
            ):
                shape_label = "triangle"
                shape_confidence = max(shape_confidence, 0.68)

            boundary_edge_ratio, inner_edge_density, inner_contrast = _visual_evidence(
                gray_image,
                edges,
                contour,
                edge_band_width=int(verification["edge_band_width"]),
            )
            geometry_score = _candidate_score(
                solidity=solidity,
                extent=extent,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                shape_label=shape_label,
                area_ratio=area_ratio,
            )
            # Visual checks are deliberately evidence, not hard gates. A valid
            # sign can be blurred, shadowed, distant, or partly hidden.
            score = (
                0.55 * geometry_score
                + 0.12
                * _visual_score(
                    boundary_edge_ratio=boundary_edge_ratio,
                    inner_edge_density=inner_edge_density,
                    inner_contrast=inner_contrast,
                    settings=verification,
                )
                + 0.25 * _size_score(area_ratio)
                + 0.08 * shape_confidence
            )
            if score < float(settings["min_candidate_score"]):
                continue
            candidates.append(
                Candidate(
                    color=color,
                    bbox=BoundingBox(x=x, y=y, width=width, height=height),
                    contour=contour,
                    area=area,
                    area_ratio=area_ratio,
                    aspect_ratio=aspect_ratio,
                    extent=extent,
                    solidity=solidity,
                    circularity=circularity,
                    polygon_vertices=vertices,
                    shape_label=shape_label,
                    shape_confidence=shape_confidence,
                    boundary_edge_ratio=boundary_edge_ratio,
                    inner_edge_density=inner_edge_density,
                    inner_contrast=inner_contrast,
                    score=score,
                )
            )

    return non_maximum_suppression(
        candidates,
        dominant_area_ratio=float(settings["dominant_candidate_min_area_ratio"]),
        secondary_area_ratio=float(settings["secondary_min_relative_area"]),
        dominant_score=float(settings["dominant_candidate_min_score"]),
    )


def non_maximum_suppression(
    candidates: list[Candidate],
    iou_threshold: float = 0.45,
    *,
    dominant_area_ratio: float = 0.08,
    secondary_area_ratio: float = 0.15,
    dominant_score: float = 0.75,
) -> list[Candidate]:
    selected: list[Candidate] = []
    ranked_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    for candidate in ranked_candidates:
        # A small coloured detail is retained if it is the only strong result,
        # but is removed when a considerably larger sign contains it.
        contained = any(
            _contained_by_larger(candidate, other)
            and other.score >= candidate.score - 0.08
            for other in ranked_candidates
            if other is not candidate
        )
        if contained:
            continue
        if all(
            candidate.bbox.intersection_over_union(existing.bbox) < iou_threshold
            for existing in selected
        ):
            selected.append(candidate)

    if not selected:
        return selected

    primary = selected[0]
    if primary.score < dominant_score or primary.area_ratio < dominant_area_ratio:
        return selected

    # A clear foreground sign lets us discard much smaller, unrelated colour
    # fragments elsewhere in the image (for example text, logos, or arrows).
    # The rule is not applied when the leading sign itself is small, preserving
    # recall for distant-road scenes.
    relative_area_floor = primary.area_ratio * secondary_area_ratio
    return [
        primary,
        *[
            candidate
            for candidate in selected[1:]
            if candidate.area_ratio >= relative_area_floor
        ],
    ]
