from __future__ import annotations

import argparse
import base64
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from run_demo import DEFAULT_CONFIG, annotate, load_part1_modules


ROOT = Path(__file__).resolve().parent
WEB_DIST = ROOT / "apps" / "web" / "dist"
PIPELINE = load_part1_modules()

app = FastAPI(title="Part 1 Colour and Shape Dashboard")

if (WEB_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


def _base64_image(image: np.ndarray[Any, Any], extension: str = ".jpg") -> str:
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise ValueError("Unable to encode dashboard image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _contour_overlay(image: np.ndarray[Any, Any], candidates: tuple[Any, ...]) -> np.ndarray[Any, Any]:
    """Draw the retained contours and their measured classical features."""
    overlay = image.copy()
    colours = {"red": (30, 50, 235), "blue": (235, 130, 30), "yellow": (30, 220, 230)}
    for candidate in candidates:
        colour = colours.get(candidate.color, (80, 210, 80))
        cv2.drawContours(overlay, [candidate.contour], -1, colour, 2)
        bbox = candidate.bbox
        label = (
            f"{candidate.color}: v={candidate.polygon_vertices} "
            f"C={candidate.circularity:.2f} T={candidate.triangle_fit:.2f} "
            f"M={candidate.color_coverage:.2f}"
        )
        cv2.putText(
            overlay,
            label,
            (bbox.x, max(16, bbox.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colour,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _processing_trace(
    image: np.ndarray[Any, Any], result: Any, annotated: np.ndarray[Any, Any]
) -> dict[str, Any]:
    """Return the visible evidence for each Part 1 processing stage."""
    return {
        "original_jpeg_base64": _base64_image(image),
        "raw_masks_png_base64": {
            colour: _base64_image(mask, ".png") for colour, mask in result.raw_masks.items()
        },
        "clean_masks_png_base64": {
            colour: _base64_image(mask, ".png") for colour, mask in result.masks.items()
        },
        "contours_jpeg_base64": _base64_image(_contour_overlay(image, result.candidates)),
        "final_jpeg_base64": _base64_image(annotated),
        "parameters": {
            "hsv_ranges": DEFAULT_CONFIG["colors"],
            "morphology": DEFAULT_CONFIG["morphology"],
            "minimum_contour_area_percent": DEFAULT_CONFIG["candidates"]["min_area_ratio"]
            * 100,
            "minimum_extent": DEFAULT_CONFIG["candidates"]["min_extent"],
            "minimum_solidity": DEFAULT_CONFIG["candidates"]["min_solidity"],
            "maximum_aspect_ratio": DEFAULT_CONFIG["candidates"]["max_aspect_ratio"],
            "minimum_color_coverage": DEFAULT_CONFIG["candidates"]["minimum_color_coverage"],
            "preferred_area_percent": DEFAULT_CONFIG["candidates"]["preferred_area_ratio"]
            * 100,
            "ranking_weights": {
                "geometry": DEFAULT_CONFIG["candidates"]["geometry_score_weight"],
                "scale": DEFAULT_CONFIG["candidates"]["scale_score_weight"],
                "color_support": DEFAULT_CONFIG["candidates"]["color_support_weight"],
            },
            "polygon_epsilon_fractions": DEFAULT_CONFIG["shape"]["polygon_epsilon_fractions"],
            "circle_min_circularity": DEFAULT_CONFIG["shape"]["circle_min_circularity"],
            "triangle_min_fit": DEFAULT_CONFIG["shape"]["triangle_min_fit"],
            "near_square_aspect_ratio": [
                DEFAULT_CONFIG["shape"]["near_square_aspect_min"],
                DEFAULT_CONFIG["shape"]["near_square_aspect_max"],
            ],
        },
    }


def _read_upload_image(data: bytes) -> np.ndarray[Any, Any]:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")
    return image


def _event_from_candidate(candidate: Any, index: int, frame_id: int, latency_ms: float) -> dict[str, Any]:
    """Create a compact colour-and-shape result for the browser dashboard."""
    bbox = candidate.bbox
    label = f"{candidate.color} {candidate.shape_label.replace('_', ' ')}"
    return {
        "frame_id": frame_id,
        "track_id": index,
        "label": label.title(),
        "confidence": float(candidate.score),
        "bbox": {"x1": bbox.x, "y1": bbox.y, "x2": bbox.x2, "y2": bbox.y2},
        "severity": "information",
        "latency_ms": round(latency_ms, 2),
        "evidence": [
            f"area_ratio={candidate.area_ratio:.5f}",
            f"circularity={candidate.circularity:.3f}",
            f"vertices={candidate.polygon_vertices}",
            f"vertex_votes={','.join(str(value) for value in candidate.polygon_vertex_counts)}",
            f"triangle_fit={candidate.triangle_fit:.3f}",
            f"color_coverage={candidate.color_coverage:.3f}",
            f"scale_evidence={candidate.scale_evidence:.3f}",
        ],
    }


def analyze_bgr(
    image: np.ndarray[Any, Any], frame_id: int = 1
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    started = time.perf_counter()
    result = PIPELINE.process_image(
        image,
        image_id=f"frame_{frame_id}",
        image_path="dashboard_input",
        config=DEFAULT_CONFIG,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    frame = {
        "frame_id": frame_id,
        "width": int(result.width),
        "height": int(result.height),
        "mode": "baseline",
        "latency_ms": round(latency_ms, 2),
        "events": [
            _event_from_candidate(candidate, index, frame_id, latency_ms)
            for index, candidate in enumerate(result.candidates, start=1)
        ],
        "warnings": ["Part 1 output is limited to colour and shape."],
    }
    annotated = annotate(image, result.candidates)
    return frame, _base64_image(annotated), _processing_trace(image, result, annotated)


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "part1-hsv-contour",
        "diagnostics": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "cuda_available": False,
            "healthy": True,
        },
        "models": {
            "mode": "baseline",
            "detector": "HSV masks, morphology and contour geometry",
            "detector_device": "cpu",
            "warnings": ["Part 1 output is colour and shape only."],
        },
    }


@app.post("/api/v1/infer/image")
async def infer_image(file: UploadFile = File(...)) -> dict[str, Any]:
    frame, annotated, processing = analyze_bgr(_read_upload_image(await file.read()))
    return {"result": frame, "annotated_jpeg_base64": annotated, "processing": processing}


@app.post("/api/v1/infer/batch")
async def infer_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    results = []
    for file in files:
        try:
            frame, _, _ = analyze_bgr(_read_upload_image(await file.read()))
            results.append({"filename": file.filename, "result": frame, "error": None})
        except Exception as exc:
            results.append({"filename": file.filename, "result": None, "error": str(exc)})
    return {"count": len(results), "results": results}


@app.get("/")
@app.get("/{path:path}")
def serve_dashboard(path: str = "") -> Any:
    index = WEB_DIST / "index.html"
    requested = WEB_DIST / path
    if path and requested.is_file():
        return FileResponse(requested)
    if index.is_file():
        return FileResponse(index)
    return PlainTextResponse(
        "Dashboard has not been built. Run: cd apps\\web; npm run build", status_code=503
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Part 1 colour and shape dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
