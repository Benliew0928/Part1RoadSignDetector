from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import cv2

from roadsign_assist.baseline.candidates import extract_candidates
from roadsign_assist.baseline.models import BaselineResult, UInt8Image
from roadsign_assist.baseline.segmentation import segment_color_stages


def read_bgr(path: str | Path) -> UInt8Image:
    resolved = Path(path)
    data = cast(UInt8Image | None, cv2.imread(str(resolved), cv2.IMREAD_COLOR))
    if data is None:
        raise ValueError(f"Unable to decode image: {resolved}")
    return data


def process_image(
    image: UInt8Image,
    *,
    image_id: str,
    image_path: str,
    config: dict[str, Any],
) -> BaselineResult:
    started = time.perf_counter()
    raw_masks, masks = segment_color_stages(image, config)
    candidates = extract_candidates(masks, image, config)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return BaselineResult(
        image_id=image_id,
        image_path=image_path,
        width=image.shape[1],
        height=image.shape[0],
        runtime_ms=elapsed_ms,
        candidates=tuple(candidates),
        raw_masks=raw_masks,
        masks=masks,
    )
