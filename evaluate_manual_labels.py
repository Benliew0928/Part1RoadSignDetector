from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from run_demo import DEFAULT_CONFIG, load_part1_modules


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate primary Part 1 colour/shape results against a CSV label file."
    )
    parser.add_argument("input_dir", type=Path, help="Root folder containing the labelled images.")
    parser.add_argument("labels", type=Path, help="CSV with source_folder, filename, expected_color, expected_shape.")
    args = parser.parse_args()

    with args.labels.open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))

    required_columns = {"source_folder", "filename", "expected_color", "expected_shape"}
    if not labels or not required_columns.issubset(labels[0]):
        raise ValueError(f"Label CSV must contain: {', '.join(sorted(required_columns))}")

    pipeline = load_part1_modules()
    metrics: Counter[str] = Counter()
    failures: list[str] = []

    for label in labels:
        image_path = args.input_dir / label["source_folder"] / label["filename"]
        result = pipeline.process_image(
            pipeline.read_bgr(image_path),
            image_id=image_path.stem,
            image_path=str(image_path),
            config=DEFAULT_CONFIG,
        )
        primary = result.candidates[0] if result.candidates else None
        metrics["images"] += 1

        if primary is not None:
            metrics["detected"] += 1
            metrics["colour_correct"] += primary.color == label["expected_color"]
            metrics["shape_correct"] += primary.shape_label == label["expected_shape"]
            exact = (
                primary.color == label["expected_color"]
                and primary.shape_label == label["expected_shape"]
            )
            metrics["exact"] += exact
        else:
            exact = False

        if not exact:
            received = "no candidate" if primary is None else f"{primary.color} {primary.shape_label}"
            failures.append(
                f"{image_path.name}: expected {label['expected_color']} "
                f"{label['expected_shape']}; received {received}"
            )

    total = metrics["images"]
    print(f"Images: {total}")
    for metric in ("detected", "colour_correct", "shape_correct", "exact"):
        count = metrics[metric]
        print(f"{metric.replace('_', ' ').title()}: {count}/{total} ({count / total:.1%})")
    if failures:
        print("\nFailures:")
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
