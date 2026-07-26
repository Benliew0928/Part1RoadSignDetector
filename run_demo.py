from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


DEFAULT_CONFIG: dict[str, Any] = {
    "preprocessing": {
        "clahe_enabled": True,
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": 8,
        "gaussian_blur_size": 3,
    },
    "colors": {
        "red": {
            "ranges": [
                {"lower": [0, 70, 45], "upper": [12, 255, 255]},
                {"lower": [165, 70, 45], "upper": [179, 255, 255]},
            ]
        },
        "blue": {"ranges": [{"lower": [90, 65, 40], "upper": [135, 255, 255]}]},
        "yellow": {"ranges": [{"lower": [15, 70, 70], "upper": [40, 255, 255]}]},
    },
    "morphology": {
        "open_kernel": 3,
        "close_kernel": 7,
        "iterations": 1,
        "background_refinement": {
            "enabled": True,
            "giant_contour_area_ratio": 0.70,
            "saturation_floor": 130,
        },
    },
    "candidates": {
        "min_area_ratio": 0.0005,
        "max_area_ratio": 0.70,
        "min_width": 8,
        "min_height": 8,
        "min_extent": 0.20,
        "min_solidity": 0.45,
        "max_aspect_ratio": 3.5,
    },
}


def _package(name: str, path: Path | None = None) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)] if path else []
    sys.modules[name] = module


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_part1_modules() -> Any:
    _package("roadsign_assist")
    _package("roadsign_assist.baseline")
    _package("roadsign_assist.baseline.member_modules", ROOT / "member_modules")

    _load("roadsign_assist.baseline.models", ROOT / "models.py")

    for owner_folder in (
        "ben_red_sign_segmentation",
        "mj_blue_sign_segmentation",
        "jy_yellow_sign_segmentation",
        "lj_shape_detection",
    ):
        package_name = f"roadsign_assist.baseline.member_modules.{owner_folder}"
        package_path = ROOT / "member_modules" / owner_folder
        _package(package_name, package_path)
        _load(package_name, package_path / "__init__.py")

    _load("roadsign_assist.baseline.segmentation", ROOT / "segmentation.py")
    _load("roadsign_assist.baseline.candidates", ROOT / "candidates.py")
    return _load("roadsign_assist.baseline.pipeline", ROOT / "pipeline.py")


def iter_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    images = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No image files found in {input_dir}")
    return images


def annotate(image: np.ndarray[Any, np.dtype[np.uint8]], candidates: tuple[Any, ...]) -> Any:
    annotated = image.copy()
    colors = {"red": (30, 50, 235), "blue": (235, 130, 30), "yellow": (30, 220, 230)}
    for index, candidate in enumerate(candidates, start=1):
        bbox = candidate.bbox
        color = colors.get(candidate.color, (50, 210, 80))
        cv2.rectangle(annotated, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 2)
        label = f"{index} {candidate.color} {candidate.shape_label} {candidate.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (bbox.x, max(18, bbox.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def run(input_dir: Path, output_dir: Path) -> None:
    pipeline = load_part1_modules()
    images = iter_images(input_dir)

    for name in ("masks", "annotated", "crops"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for image_path in images:
        image = pipeline.read_bgr(image_path)
        result = pipeline.process_image(
            image,
            image_id=image_path.stem,
            image_path=str(image_path),
            config=DEFAULT_CONFIG,
        )

        for color, mask in result.masks.items():
            cv2.imwrite(str(output_dir / "masks" / f"{image_path.stem}__{color}.png"), mask)

        cv2.imwrite(str(output_dir / "annotated" / f"{image_path.stem}.jpg"), annotate(image, result.candidates))

        for index, candidate in enumerate(result.candidates, start=1):
            bbox = candidate.bbox
            crop = image[bbox.y : bbox.y2, bbox.x : bbox.x2]
            crop_path = output_dir / "crops" / f"{image_path.stem}__{index:02d}.png"
            if crop.size:
                cv2.imwrite(str(crop_path), crop)
            candidate_rows.append(
                {
                    "image_id": image_path.stem,
                    "candidate_index": index,
                    **candidate.serializable(),
                    "crop_path": crop_path.relative_to(output_dir).as_posix(),
                }
            )

        image_rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "width": result.width,
                "height": result.height,
                "candidate_count": len(result.candidates),
                "runtime_ms": round(result.runtime_ms, 3),
            }
        )

    with (output_dir / "images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)

    with (output_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Processed {len(image_rows)} images into {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run assignment Part 1 traffic-sign segmentation.")
    parser.add_argument("input_dir", help="Folder containing traffic sign images.")
    parser.add_argument("--output", default="outputs/part1_demo", help="Output folder.")
    args = parser.parse_args()

    run(Path(args.input_dir), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
