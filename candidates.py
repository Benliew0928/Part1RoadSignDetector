from __future__ import annotations

import math
from typing import Any, cast

import cv2
import numpy as np

from roadsign_assist.baseline.member_modules.lj_shape_detection import (
    choose_shape,
    is_supported_sign_shape,
    representative_vertex_count,
    score_candidate,
    shape_fit_scores,
)
from roadsign_assist.baseline.models import BoundingBox, Candidate, Contour, UInt8Image


def extract_candidates(
    masks: dict[str, UInt8Image],
    image: UInt8Image,
    config: dict[str, Any],
) -> list[Candidate]:
    """Extract colour regions and classify their visible geometric shape.

    Each candidate uses only the current image's colour mask.  The shape
    decision is a global combination of contour-to-model fits and conventional
    geometry; no reference images, filename rules, or learned model are used.
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

            raw_area_ratio = area / image_area
            raw_x, raw_y, raw_width, raw_height = cv2.boundingRect(contour)
            if (
                raw_area_ratio < float(candidate_settings["min_area_ratio"])
                or raw_area_ratio > float(candidate_settings["max_area_ratio"])
                or raw_width < int(candidate_settings["min_width"])
                or raw_height < int(candidate_settings["min_height"])
            ):
                continue

            raw_border_touches = sum(
                (
                    raw_x <= 0,
                    raw_y <= 0,
                    raw_x + raw_width >= image_width,
                    raw_y + raw_height >= image_height,
                )
            )
            if raw_border_touches > int(candidate_settings.get("max_border_touches", 4)):
                continue

            # Refine only the local colour evidence around this contour, then
            # fill its outer boundary. This reconnects small local breaks but
            # rejects growth that would absorb a nearby unrelated object.
            shape_contour, refinement_ratio = _refined_silhouette_contour(
                mask,
                contour,
                padding=int(shape_settings.get("silhouette_padding", 2)),
                close_kernel=int(shape_settings.get("silhouette_close_kernel", 3)),
                context_padding=int(shape_settings.get("silhouette_refine_context_padding", 4)),
                max_growth=float(shape_settings.get("silhouette_max_refinement_ratio", 1.35)),
                min_seed_overlap=float(shape_settings.get("silhouette_min_seed_overlap", 0.55)),
            )
            x, y, width, height = cv2.boundingRect(shape_contour)
            silhouette_area = float(cv2.contourArea(shape_contour))
            if silhouette_area <= 0.0:
                continue
            final_area_ratio = silhouette_area / image_area
            bbox_area = max(1, width * height)
            extent = silhouette_area / bbox_area
            hull = cv2.convexHull(shape_contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = silhouette_area / hull_area if hull_area else 0.0
            hull_perimeter = float(cv2.arcLength(hull, True))
            if hull_perimeter <= 0.0:
                continue

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
            approximations = tuple(
                cast(Contour, cv2.approxPolyDP(hull, epsilon * hull_perimeter, True))
                for epsilon in epsilon_fractions
            )
            vertex_counts = tuple(len(approximation) for approximation in approximations)
            vertices = representative_vertex_count(vertex_counts)
            aspect_ratio = width / max(1, height)
            rotated_width, rotated_height = cv2.minAreaRect(hull)[1]
            rotated_aspect_ratio = (
                min(rotated_width, rotated_height) / max(rotated_width, rotated_height)
                if rotated_width > 0.0 and rotated_height > 0.0
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

            fits = _shape_fit_measurements(shape_contour, hull, approximations)
            scores = shape_fit_scores(
                vertex_counts,
                circularity,
                circle_fit=fits["circle_fit"],
                ellipse_fit=fits["ellipse_fit"],
                ellipse_axis_ratio=fits["ellipse_axis_ratio"],
                triangle_fit=fits["triangle_fit"],
                rectangle_fit=fits["rectangle_fit"],
                octagon_fit=fits["octagon_fit"],
                circle_min_circularity=float(shape_settings["circle_min_circularity"]),
                triangle_max_circularity=float(
                    shape_settings.get("triangle_max_circularity", 0.88)
                ),
                rectangle_max_triangle_fit=float(
                    shape_settings.get("rectangle_max_triangle_fit", 0.70)
                ),
                perspective_ellipse_min_axis_ratio=float(
                    shape_settings.get("perspective_ellipse_min_axis_ratio", 0.55)
                ),
            )
            shape_label = choose_shape(
                scores,
                minimum_score=float(shape_settings.get("minimum_shape_fit_score", 0.74)),
            )
            shape_confidence = float(scores.get(shape_label, 0.0))
            ranking_shape_confidence = _ranking_shape_confidence(
                shape_label,
                circularity,
                rotated_aspect_ratio,
                vertex_counts,
                fits["triangle_fit"],
                shape_settings,
            )

            color_coverage = _color_coverage(mask, shape_contour)
            color_support = min(
                1.0,
                color_coverage
                / max(1e-6, float(candidate_settings.get("minimum_color_coverage", 0.25))),
            )
            scale_evidence = _scale_evidence(final_area_ratio, candidate_settings)
            border_touches = sum(
                (
                    x <= 0,
                    y <= 0,
                    x + width >= image_width,
                    y + height >= image_height,
                )
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
                    circle_fit=fits["circle_fit"],
                    ellipse_fit=fits["ellipse_fit"],
                    ellipse_axis_ratio=fits["ellipse_axis_ratio"],
                    triangle_fit=fits["triangle_fit"],
                    rectangle_fit=fits["rectangle_fit"],
                    octagon_fit=fits["octagon_fit"],
                    rotated_aspect_ratio=rotated_aspect_ratio,
                    silhouette_refinement_ratio=refinement_ratio,
                    color_coverage=color_coverage,
                    scale_evidence=scale_evidence,
                    border_touches=border_touches,
                    shape_label=shape_label,
                    shape_confidence=shape_confidence,
                    # Kept only for the existing API schema. This baseline
                    # deliberately does not use edge-derived scoring.
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
                            shape_confidence=ranking_shape_confidence,
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


def _ranking_shape_confidence(
    shape_label: str,
    circularity: float,
    aspect_ratio: float,
    vertex_counts: tuple[int, ...],
    triangle_fit: float,
    settings: dict[str, Any],
) -> float:
    """Keep candidate ordering tied to the original broad geometry evidence.

    Fit scores choose the shape label. This separate value prevents a new
    label from unnecessarily changing which overlapping colour candidate is
    ranked first.
    """
    if shape_label == "triangle":
        votes = sum(value == 3 for value in vertex_counts) / len(vertex_counts)
        return max(votes, min(1.0, triangle_fit))
    if shape_label == "square_or_rectangle":
        votes = sum(value == 4 for value in vertex_counts) / len(vertex_counts)
        return max(votes, min(1.0, 1.0 - triangle_fit))
    if shape_label == "octagon":
        votes = sum(value == 8 for value in vertex_counts) / len(vertex_counts)
        stability = 1.0 if all(value == 8 for value in vertex_counts) else 0.0
        return 0.5 * votes + 0.5 * stability
    if shape_label == "circle":
        circularity_score = min(1.0, circularity / float(settings["circle_min_circularity"]))
        aspect_distance = abs(math.log(max(aspect_ratio, 1e-6)))
        return max(
            0.0,
            min(1.0, 0.75 * circularity_score + 0.25 * (1.0 - aspect_distance)),
        )
    return 0.0


def _shape_fit_measurements(
    silhouette: Contour,
    hull: Contour,
    approximations: tuple[Contour, ...],
) -> dict[str, float]:
    """Measure overlap between a silhouette and conventional fitted shapes."""
    circle_center, circle_radius = cv2.minEnclosingCircle(hull)
    circle = _circle_contour(circle_center, circle_radius)
    circle_fit = _contour_iou(silhouette, circle)

    ellipse_fit = 0.0
    ellipse_axis_ratio = 0.0
    if len(hull) >= 5:
        ellipse_center, ellipse_axes, ellipse_angle = cv2.fitEllipse(hull)
        major_axis = max(float(ellipse_axes[0]), float(ellipse_axes[1]))
        minor_axis = min(float(ellipse_axes[0]), float(ellipse_axes[1]))
        ellipse_axis_ratio = minor_axis / major_axis if major_axis > 0.0 else 0.0
        ellipse = _ellipse_contour(ellipse_center, ellipse_axes, ellipse_angle)
        ellipse_fit = _contour_iou(silhouette, ellipse)

    enclosing_triangle_area, _ = cv2.minEnclosingTriangle(hull)
    # The minimum enclosing triangle contains the convex hull, so this is an
    # explainable coverage fit: a true triangle fills most of its fitted
    # triangle even when its coloured border is slightly irregular.
    hull_area = float(cv2.contourArea(hull))
    triangle_fit = (
        hull_area / float(enclosing_triangle_area)
        if enclosing_triangle_area > 0.0
        else 0.0
    )

    rotated_rectangle = cv2.boxPoints(cv2.minAreaRect(hull)).reshape(-1, 1, 2)
    rectangle_fit = _contour_iou(silhouette, rotated_rectangle)
    for approximation in approximations:
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            rectangle_fit = max(rectangle_fit, _contour_iou(silhouette, approximation))

    octagon_fit = 0.0
    for approximation in approximations:
        if len(approximation) == 8 and cv2.isContourConvex(approximation):
            octagon_fit = max(octagon_fit, _contour_iou(silhouette, approximation))

    return {
        "circle_fit": circle_fit,
        "ellipse_fit": ellipse_fit,
        "ellipse_axis_ratio": ellipse_axis_ratio,
        "triangle_fit": triangle_fit,
        "rectangle_fit": rectangle_fit,
        "octagon_fit": octagon_fit,
    }


def _circle_contour(center: tuple[float, float], radius: float) -> Contour:
    radius_int = max(1, int(round(radius)))
    center_int = tuple(int(round(value)) for value in center)
    points = cv2.ellipse2Poly(center_int, (radius_int, radius_int), 0, 0, 360, 4)
    return cast(Contour, points.reshape(-1, 1, 2))


def _ellipse_contour(
    center: tuple[float, float],
    axes: tuple[float, float],
    angle: float,
) -> Contour:
    center_int = tuple(int(round(value)) for value in center)
    semi_axes = tuple(max(1, int(round(value / 2.0))) for value in axes)
    points = cv2.ellipse2Poly(center_int, semi_axes, int(round(angle)), 0, 360, 4)
    return cast(Contour, points.reshape(-1, 1, 2))


def _contour_iou(first: Contour, second: np.ndarray[Any, Any] | None) -> float:
    """Return filled-contour IoU in a small local canvas."""
    if second is None:
        return 0.0
    first_int = np.rint(first).astype(np.int32).reshape(-1, 1, 2)
    second_int = np.rint(second).astype(np.int32).reshape(-1, 1, 2)
    if len(first_int) < 3 or len(second_int) < 3:
        return 0.0

    all_points = np.concatenate((first_int.reshape(-1, 2), second_int.reshape(-1, 2)))
    x0, y0 = np.min(all_points, axis=0) - 1
    x1, y1 = np.max(all_points, axis=0) + 1
    width = int(x1 - x0 + 1)
    height = int(y1 - y0 + 1)
    if width <= 0 or height <= 0:
        return 0.0

    first_local = first_int.copy()
    second_local = second_int.copy()
    first_local[:, 0, 0] -= x0
    first_local[:, 0, 1] -= y0
    second_local[:, 0, 0] -= x0
    second_local[:, 0, 1] -= y0
    first_mask = np.zeros((height, width), dtype=np.uint8)
    second_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(first_mask, [first_local], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(second_mask, [second_local], -1, 255, thickness=cv2.FILLED)
    union = cv2.countNonZero(cv2.bitwise_or(first_mask, second_mask))
    if union == 0:
        return 0.0
    intersection = cv2.countNonZero(cv2.bitwise_and(first_mask, second_mask))
    return intersection / union


def _refined_silhouette_contour(
    mask: UInt8Image,
    contour: Contour,
    *,
    padding: int,
    close_kernel: int,
    context_padding: int,
    max_growth: float,
    min_seed_overlap: float,
) -> tuple[Contour, float]:
    """Repair a local colour component before making its filled silhouette.

    The local context allows a small closing operation to reconnect broken
    parts of the same sign border. Candidate growth is bounded and must cover
    most of the original seed, preventing a nearby coloured object from being
    silently absorbed.
    """
    x, y, width, height = cv2.boundingRect(contour)
    margin = max(0, padding) + max(0, context_padding)
    y0 = max(0, y - margin)
    x0 = max(0, x - margin)
    y1 = min(mask.shape[0], y + height + margin)
    x1 = min(mask.shape[1], x + width + margin)
    local_colour = mask[y0:y1, x0:x1].copy()

    seed = np.zeros(local_colour.shape, dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x0
    shifted[:, 0, 1] -= y0
    cv2.drawContours(seed, [shifted], -1, 255, thickness=cv2.FILLED)
    seed_pixels = cv2.countNonZero(seed)
    if seed_pixels == 0:
        return contour, 1.0

    if context_padding > 0:
        context_size = 2 * context_padding + 1
        context_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (context_size, context_size)
        )
        search_region = cv2.dilate(seed, context_kernel)
        local_colour = cv2.bitwise_and(local_colour, search_region)

    adjusted_kernel = max(1, close_kernel)
    if adjusted_kernel % 2 == 0:
        adjusted_kernel += 1
    if adjusted_kernel >= 3:
        close_shape = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (adjusted_kernel, adjusted_kernel)
        )
        local_colour = cv2.morphologyEx(local_colour, cv2.MORPH_CLOSE, close_shape)

    refined_contours, _ = cv2.findContours(
        local_colour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    choices: list[tuple[float, float, float, Contour]] = []
    for local_contour in refined_contours:
        local_filled = np.zeros(local_colour.shape, dtype=np.uint8)
        cv2.drawContours(local_filled, [local_contour], -1, 255, thickness=cv2.FILLED)
        refined_area = float(cv2.contourArea(local_contour))
        seed_area = float(cv2.contourArea(shifted))
        if seed_area <= 0.0 or refined_area <= 0.0:
            continue
        growth = refined_area / seed_area
        seed_overlap = cv2.countNonZero(cv2.bitwise_and(local_filled, seed)) / seed_pixels
        if growth > max_growth or seed_overlap < min_seed_overlap:
            continue
        choices.append((seed_overlap, -abs(growth - 1.0), growth, local_contour))

    if not choices:
        return _filled_silhouette_contour(contour, padding=padding, close_kernel=close_kernel), 1.0

    _, _, growth, best_local_contour = max(choices, key=lambda item: (item[0], item[1]))
    best = best_local_contour.copy()
    best[:, 0, 0] += x0
    best[:, 0, 1] += y0
    return _filled_silhouette_contour(
        cast(Contour, best), padding=padding, close_kernel=close_kernel
    ), growth


def _filled_silhouette_contour(
    contour: Contour,
    *,
    padding: int,
    close_kernel: int,
) -> Contour:
    """Fill one local contour and return its stable outer boundary."""
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
    adjusted_kernel = max(1, close_kernel)
    if adjusted_kernel % 2 == 0:
        adjusted_kernel += 1
    if adjusted_kernel >= 3:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (adjusted_kernel, adjusted_kernel)
        )
        local = cast(UInt8Image, cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel))
    local_contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not local_contours:
        return contour
    largest = max(local_contours, key=cv2.contourArea).copy()
    largest[:, 0, 0] += x - safe_padding
    largest[:, 0, 1] += y - safe_padding
    return cast(Contour, largest)


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
    """Give compact sign-sized regions ranking evidence without rejecting crops."""
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
    """Rank candidates from global geometry, scale and mask support only."""
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


def non_maximum_suppression(
    candidates: list[Candidate],
    *,
    iou_threshold: float,
) -> list[Candidate]:
    """Keep overlapping duplicates once, then order by global evidence."""
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
