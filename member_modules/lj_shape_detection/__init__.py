"""LJ: shape detection of signs to support segmentation."""

from roadsign_assist.baseline.member_modules.lj_shape_detection.shape import (
    MODULE_OWNER,
    MODULE_TITLE,
    classify_shape,
    is_supported_sign_shape,
    representative_vertex_count,
    score_candidate,
)

__all__ = [
    "MODULE_OWNER",
    "MODULE_TITLE",
    "classify_shape",
    "is_supported_sign_shape",
    "representative_vertex_count",
    "score_candidate",
]
