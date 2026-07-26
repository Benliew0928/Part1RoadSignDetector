from __future__ import annotations

import argparse
import base64
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from run_demo import DEFAULT_CONFIG, annotate, load_part1_modules


ROOT = Path(__file__).resolve().parent
WEB_DIST = ROOT / "apps" / "web" / "dist"
PIPELINE = load_part1_modules()

app = FastAPI(title="Part 1 Color/Shape Dashboard")

if (WEB_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


def _jpeg_base64(image: np.ndarray[Any, Any]) -> str:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Unable to encode annotated image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _read_upload_image(data: bytes) -> np.ndarray[Any, Any]:
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")
    return image


def _event_from_candidate(candidate: Any, index: int, frame_id: int, latency_ms: float) -> dict[str, Any]:
    bbox = candidate.bbox
    label = f"{candidate.color} {candidate.shape_label.replace('_', ' ')}"
    evidence = [
        f"color={candidate.color}",
        f"shape={candidate.shape_label}",
        f"score={candidate.score:.3f}",
        f"area_ratio={candidate.area_ratio:.5f}",
        f"circularity={candidate.circularity:.3f}",
        f"vertices={candidate.polygon_vertices}",
    ]
    return {
        "schema_version": "part1-color-shape-v1",
        "frame_id": frame_id,
        "track_id": index,
        "coursework_id": None,
        "semantic_sign_id": f"{candidate.color}_{candidate.shape_label}",
        "meaning": {
            "en": label.title(),
            "ms": label.title(),
            "zh": label.title(),
        },
        "ocr": {
            "text": "",
            "confidence": 0.0,
            "script": "none",
            "language": "none",
            "numeric_value": None,
            "unit": None,
            "semantic_sign_id": None,
        },
        "confidence": float(candidate.score),
        "bbox": {
            "x1": int(bbox.x),
            "y1": int(bbox.y),
            "x2": int(bbox.x2),
            "y2": int(bbox.y2),
        },
        "mask": {
            "encoding": "polygon",
            "points": [
                [int(bbox.x), int(bbox.y)],
                [int(bbox.x2), int(bbox.y)],
                [int(bbox.x2), int(bbox.y2)],
                [int(bbox.x), int(bbox.y2)],
            ],
        },
        "action": {
            "code": "COLOR_SHAPE_SEGMENTATION",
            "target_speed_kmh": None,
            "restriction_value": None,
            "restriction_unit": None,
            "direction": None,
            "advisory_only": True,
        },
        "advisory": {
            "headline": {
                "en": label.title(),
                "ms": label.title(),
                "zh": label.title(),
            },
            "instruction": {
                "en": "Detected by Part 1 color and shape segmentation.",
                "ms": "Detected by Part 1 color and shape segmentation.",
                "zh": "Detected by Part 1 color and shape segmentation.",
            },
            "safe_to_announce": False,
        },
        "severity": "information",
        "latency_ms": round(latency_ms, 2),
        "device": "part1-classical-baseline",
        "stable": True,
        "should_announce": False,
        "evidence": evidence,
    }


def analyze_bgr(image: np.ndarray[Any, Any], frame_id: int = 1) -> tuple[dict[str, Any], str]:
    started = time.perf_counter()
    result = PIPELINE.process_image(
        image,
        image_id=f"frame_{frame_id}",
        image_path="dashboard_input",
        config=DEFAULT_CONFIG,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    events = [
        _event_from_candidate(candidate, index, frame_id, latency_ms)
        for index, candidate in enumerate(result.candidates, start=1)
    ]
    frame = {
        "frame_id": frame_id,
        "width": int(result.width),
        "height": int(result.height),
        "mode": "baseline",
        "latency_ms": round(latency_ms, 2),
        "events": events,
        "warnings": [
            "Part 1 mode: output is color and shape only, not traffic-sign meaning recognition."
        ],
    }
    annotated = annotate(image, result.candidates)
    return frame, _jpeg_base64(annotated)


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "part1-color-shape-dashboard",
        "diagnostics": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "cuda_available": False,
            "official_image_count": 84,
            "healthy": True,
        },
        "models": {
            "mode": "baseline",
            "detector": "Part 1 HSV color segmentation + contour shape detection",
            "detector_available": True,
            "detector_loaded": True,
            "detector_device": "cpu",
            "detector_profile": {
                "output": "color_and_shape",
                "colors": "red, blue, yellow",
                "shapes": "circle, triangle, square_or_rectangle, octagon, other",
            },
            "classifier": "disabled for Part 1",
            "classifier_available": False,
            "classifier_loaded": False,
            "classifier_providers": [],
            "classifier_profile": {"reason": "assignment preliminary work only"},
            "tracker": "single-frame candidate index",
            "ocr_available": False,
            "ocr_loaded": False,
            "ocr_load_error": None,
            "warnings": [
                "This dashboard reports traffic sign color and shape only."
            ],
        },
    }


@app.post("/api/v1/infer/image")
async def infer_image(file: UploadFile = File(...)) -> dict[str, Any]:
    frame, annotated = analyze_bgr(_read_upload_image(await file.read()))
    return {"result": frame, "annotated_jpeg_base64": annotated}


@app.post("/api/v1/infer/batch")
async def infer_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    results = []
    for file in files:
        try:
            frame, _ = analyze_bgr(_read_upload_image(await file.read()))
            results.append({"filename": file.filename, "result": frame, "error": None})
        except Exception as exc:
            results.append({"filename": file.filename, "result": None, "error": str(exc)})
    return {"count": len(results), "results": results}


@app.post("/api/v1/infer/video")
async def infer_video(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "video.mp4").suffix) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    try:
        capture = cv2.VideoCapture(str(temp_path))
        frames_read = 0
        sampled = 0
        frame_results = []
        event_samples = []
        representative = None
        while sampled < 12:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1
            if frames_read == 1 or frames_read % 15 == 0:
                sampled += 1
                result, _ = analyze_bgr(frame, sampled)
                frame_results.append({"source_frame": frames_read, "result": result})
                event_samples.extend(result["events"][:2])
                if representative is None and result["events"]:
                    representative = result
        fps = capture.get(cv2.CAP_PROP_FPS) or None
        capture.release()
        return {
            "frames_read": frames_read,
            "sampled_frames": sampled,
            "events": len(event_samples),
            "fps": fps,
            "frame_results": frame_results,
            "event_samples": event_samples[:10],
            "representative_result": representative,
        }
    finally:
        temp_path.unlink(missing_ok=True)


@app.websocket("/api/v1/ws/camera/{session_id}")
async def camera_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    frame_id = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            frame_id += 1
            frame, _ = analyze_bgr(_read_upload_image(data), frame_id)
            await websocket.send_json(frame)
    except WebSocketDisconnect:
        return


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
        "Original React dashboard has not been built yet. Run: cd apps\\web; npm.cmd ci; npm.cmd run build",
        status_code=503,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the original dashboard with Part 1 color/shape backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
