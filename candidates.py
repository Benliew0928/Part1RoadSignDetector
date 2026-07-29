from __future__ import annotations

import math
from typing import Any, cast

import cv2
import numpy as np

from roadsign_assist.baseline.member_modules.lj_shape_detection import (
    classify_shape,
    is_supported_sign_shape,
    representative_vertex_count,
    score_candidate,
)
from roadsign_assist.baseline.models import BoundingBox, Candidate, Contour, UInt8Image


def extract_candidates(
    masks: dict[str, UInt8Image],
    image: UInt8Image,
    config: dict[str, Any],
) -> list[Candidate]:
    """Extract colour regions and describe their elementary geometry.

    This is intentionally a Part 1 contour baseline.  It makes no decision
    from filenames, an image database, a trained model, Canny edges, or a
    colour-specific shape assumption. Candidates are ordered by simple
    geometry evidence (solidity, extent, circularity, aspect ratio, and a
    recognised coursework shape), rather than only by area. This reduces the
    chance of a large coloured background panel becoming the first item.
    """
    candidate_settings = config["candidates"]
    shape_settings = config["shape"]
    image_height, image_width = image.shape[:2]
    image_area = image_width * image_height
    candidates: list[Candidate] = []

    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for raw_contour in contours:
            contour = cast(Contour, raw_contour)
            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue

            area_ratio = area / image_area
            x, y, width, height = cv2.boundingRect(contour)
            if (
                area_ratio < float(candidate_settings["min_area_ratio"])
                or area_ratio > float(candidate_settings["max_area_ratio"])
                or width < int(candidate_settings["min_width"])
                or height < int(candidate_settings["min_height"])
            ):
                continue

            border_touches = sum(
                (
                    x <= 0,
                    y <= 0,
                    x + width >= image_width,
                    y + height >= image_height,
                )
            )
            if border_touches > int(candidate_settings.get("max_border_touches", 4)):
                continue

            # Build a local silhouette from this one colour component only.
            # Filling the exterior contour restores the white/black interior
            # of a traffic sign, while the local crop prevents a different
            # colour mask from changing the contour used for shape analysis.
            shape_contour = _filled_silhouette_contour(
                contour,
                padding=int(shape_settings.get("silhouette_padding", 2)),
                close_kernel=int(shape_settings.get("silhouette_close_kernel", 3)),
            )
            x, y, width, height = cv2.boundingRect(shape_contour)
            bbox_area = max(1, width * height)
            silhouette_area = float(cv2.contourArea(shape_contour))
            final_area_ratio = silhouette_area / image_area
            extent = silhouette_area / bbox_area
            hull = cv2.convexHull(shape_contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = silhouette_area / hull_area if hull_area else 0.0
            hull_perimeter = float(cv2.arcLength(hull, True))
            if hull_perimeter <= 0.0:
                continue
            # The hull gives a conventional contour-based outline when a
            # coloured sign border is interrupted by a symbol, glare, or a
            # small occlusion.
            circularity = (
                4.0 * math.pi * hull_area / (hull_perimeter * hull_perimeter)
                if hull_area
                else 0.0
            )
            epsilon_fractions = tuple(
                float(value)
                for value in shape_settings.get(
                    "polygon_epsilon_fractions",
                    (shape_settings["polygon_epsilon_fraction"],),
                )
            )
            vertex_counts = tuple(
                len(cv2.approxPolyDP(hull, epsilon * hull_perimeter, True))
                for epsilon in epsilon_fractions
            )
            vertices = representative_vertex_count(vertex_counts)
            aspect_ratio = width / max(1, height)
            rotated_width, rotated_height = cv2.minAreaRect(hull)[1]
            rotated_aspect_ratio = (
                min(rotated_width, rotated_height) / max(rotated_width, rotated_height)
                if rotated_width > 0.0 and rotated_height > 0.0
                else 0.0
            )
            enclosing_triangle_area, _ = cv2.minEnclosingTriangle(hull)
            triangle_fit = (
                hull_area / float(enclosing_triangle_area)
                if enclosing_triangle_area > 0.0
                else 0.0
            )
            if not (
                1.0 / float(candidate_settings["max_aspect_ratio"])
                <= aspect_ratio
                <= float(candidate_settings["max_aspect_ratio"])
            ):
                continue
            if (
                extent < float(candidate_settings["min_extent"])
                or solidity < float(candidate_settings["min_solidity"])
            ):
                continue

            color_coverage = _color_coverage(mask, shape_contour)
            color_support = min(
                1.0,
                color_coverage
                / max(1e-6, float(candidate_settings.get("minimum_color_coverage", 0.25))),
            )
            scale_evidence = _scale_evidence(final_area_ratio, candidate_settings)

            shape_label = classify_shape(
                vertex_counts,
                circularity,
                rotated_aspect_ratio,
                triangle_fit=triangle_fit,
                circle_min_circularity=float(shape_settings["circle_min_circularity"]),
                near_square_min=float(shape_settings["near_square_aspect_min"]),
                near_square_max=float(shape_settings["near_square_aspect_max"]),
                triangle_min_fit=float(shape_settings.get("triangle_min_fit", 0.84)),
                triangle_max_circularity=float(
                    shape_settings.get("triangle_max_circularity", 0.88)
                ),
                triangle_min_vertex_votes=int(
                    shape_settings.get("triangle_min_vertex_votes", 2)
                ),
                rectangle_min_vertex_votes=int(
                    shape_settings.get("rectangle_min_vertex_votes", 2)
                ),
                rectangle_max_triangle_fit=float(
                    shape_settings.get("rectangle_max_triangle_fit", 0.70)
                ),
                octagon_min_vertex_votes=int(
                    shape_settings.get("octagon_min_vertex_votes", 3)
                ),
            )
            shape_confidence = _shape_confidence(
                shape_label,
                circularity,
                rotated_aspect_ratio,
                vertex_counts,
                triangle_fit,
                shape_settings,
            )
            candidates.append(
                Candidate(
                    color=color,
                    bbox=BoundingBox(x=x, y=y, width=width, height=height),
                    contour=shape_contour,
                    area=silhouette_area,
                    area_ratio=final_area_ratio,
                    aspect_ratio=aspect_ratio,
                    extent=extent,
                    solidity=solidity,
                    circularity=circularity,
                    polygon_vertices=vertices,
                    polygon_vertex_counts=vertex_counts,
                    triangle_fit=triangle_fit,
                    rotated_aspect_ratio=rotated_aspect_ratio,
                    color_coverage=color_coverage,
                    scale_evidence=scale_evidence,
                    border_touches=border_touches,
                    shape_label=shape_label,
                    shape_confidence=shape_confidence,
                    # Retained in the API schema for compatibility.  The fair
                    # baseline deliberately does not use edge-derived scoring.
                    boundary_edge_ratio=0.0,
                    inner_edge_density=0.0,
                    inner_contrast=0.0,
                    score=_combined_candidate_score(
                        geometry_score=score_candidate(
                            solidity=solidity,
                            extent=extent,
                            circularity=circularity,
                            aspect_ratio=aspect_ratio,
                            shape_label=shape_label,
                            area_ratio=final_area_ratio,
                            shape_confidence=shape_confidence,
                        ),
                        scale_evidence=scale_evidence,
                        color_support=color_support,
                        settings=candidate_settings,
                    ),
                )
            )

    return non_maximum_suppression(
        candidates,
        iou_threshold=float(candidate_settings["nms_iou_threshold"]),
    )


def _shape_confidence(
    shape_label: str,
    circularity: float,
    aspect_ratio: float,
    vertex_counts: tuple[int, ...],
    triangle_fit: float,
    settings: dict[str, Any],
) -> float:
    if shape_label == "triangle":
        votes = sum(value == 3 for value in vertex_counts) / len(vertex_counts)
        return max(votes, min(1.0, triangle_fit))
    if shape_label == "square_or_rectangle":
        votes = sum(value == 4 for value in vertex_counts) / len(vertex_counts)
        return max(votes, min(1.0, 1.0 - triangle_fit))
    if shape_label == "octagon":
        votes = sum(value in {7, 8, 9} for value in vertex_counts) / len(vertex_counts)
        stability = 1.0 if max(vertex_counts) - min(vertex_counts) <= 1 else 0.0
        return 0.5 * votes + 0.5 * stability
    if shape_label == "circle":
        circularity_score = min(1.0, circularity / float(settings["circle_min_circularity"]))
        aspect_distance = abs(math.log(max(aspect_ratio, 1e-6)))
        return max(0.0, min(1.0, 0.75 * circularity_score + 0.25 * (1.0 - aspect_distance)))
    return 0.0


def _color_coverage(mask: UInt8Image, contour: Contour) -> float:
    """Measure how much of a filled candidate is supported by its own colour mask."""
    x, y, width, height = cv2.boundingRect(contour)
    silhouette = np.zeros((height, width), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(silhouette, [shifted], -1, 255, thickness=cv2.FILLED)
    silhouette_pixels = max(1, cv2.countNonZero(silhouette))
    matching_pixels = cv2.countNonZero(
        cv2.bitwise_and(mask[y : y + height, x : x + width], silhouette)
    )
    return matching_pixels / silhouette_pixels


def _scale_evidence(area_ratio: float, settings: dict[str, Any]) -> float:
    """Give compact sign-sized regions more ranking evidence without rejecting crops.

    This is deliberately a soft score. A distant sign may be small and a
    close-up sign may be large, so neither case is removed from the candidate
    list; the score only stops a tiny coloured fragment from automatically
    outranking a plausible sign contour.
    """
    preferred = max(1e-6, float(settings.get("preferred_area_ratio", 0.06)))
    large_start = float(settings.get("large_region_area_ratio", 0.55))
    large_minimum = float(settings.get("large_region_min_evidence", 0.25))
    small_region_evidence = min(1.0, math.sqrt(max(0.0, area_ratio) / preferred))
    if area_ratio <= large_start:
        return small_region_evidence
    progress = min(1.0, (area_ratio - large_start) / max(1e-6, 1.0 - large_start))
    large_region_evidence = 1.0 - (1.0 - large_minimum) * progress
    return small_region_evidence * large_region_evidence


def _combined_candidate_score(
    *,
    geometry_score: float,
    scale_evidence: float,
    color_support: float,
    settings: dict[str, Any],
) -> float:
    """Rank candidates from global geometry, scale, and mask support only."""
    geometry_weight = float(settings.get("geometry_score_weight", 0.70))
    scale_weight = float(settings.get("scale_score_weight", 0.25))
    color_weight = float(settings.get("color_support_weight", 0.05))
    total_weight = geometry_weight + scale_weight + color_weight
    if total_weight <= 0.0:
        return geometry_score
    return (
        geometry_weight * geometry_score
        + scale_weight * scale_evidence
        + color_weight * color_support
    ) / total_weight


def _filled_silhouette_contour(
    contour: Contour,
    *,
    padding: int,
    close_kernel: int,
) -> Contour:
    """Fill one contour in a padded local image and return its outer boundary."""
    x, y, width, height = cv2.boundingRect(contour)
    safe_padding = max(0, padding)
    local = cast(
        UInt8Image,
        np.zeros((height + 2 * safe_padding, width + 2 * safe_padding), dtype=np.uint8),
    )
    shifted = contour.copy()
    shifted[:, 0, 0] -= x - safe_padding
    shifted[:, 0, 1] -= y - safe_padding
    cv2.drawContours(local, [shifted], -1, 255, thickness=cv2.FILLED)
    if close_kernel >= 3:
        if close_kernel % 2 == 0:
            close_kernel += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        local = cast(UInt8Image, cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel))
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return contour
    largest = max(contours, key=cv2.contourArea).copy()
    largest[:, 0, 0] += x - safe_padding
    largest[:, 0, 1] += y - safe_padding
    return cast(Contour, largest)


def non_maximum_suppression(
    candidates: list[Candidate],
    *,
    iou_threshold: float,
) -> list[Candidate]:
    """Keep overlapping duplicates once, then order by geometry evidence."""
    selected: list[Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.score, is_supported_sign_shape(item.shape_label), item.area),
        reverse=True,
    ):
        if all(
            candidate.bbox.intersection_over_union(existing.bbox) < iou_threshold
            for existing in selected
        ):
            selected.append(candidate)
    return selected
