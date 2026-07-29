# Road Sign Detector Part 1 — Defence and Q&A Guide

## One-minute introduction

This project is a Python and OpenCV classical computer-vision system for the Part 1 coursework component.

It identifies red, blue and yellow sign-like regions, then classifies the visible shape as a circle, triangle, square/rectangle, octagon or `other`.

It does **not** use YOLO, deep learning, OCR, a trained model, image filenames or a stored answer table.

Every uploaded image follows the same HSV segmentation, morphology, contour and geometry pipeline.

The scope is colour and shape only. A red circle is not claimed to be a specific speed limit because this project does not read the number inside it.

## Technology

| Technology | Purpose |
|---|---|
| Python | Detector, pipeline and dashboard backend |
| OpenCV (`cv2`) | HSV conversion, masks, morphology, contours, hulls and geometry |
| NumPy | Image-array operations |
| FastAPI | Image and batch upload API |
| React + TypeScript | Browser dashboard |

## Architecture and member ownership

| Component | Main file | Responsibility |
|---|---|---|
| Red segmentation | `member_modules/ben_red_sign_segmentation/red.py` | Creates the red HSV mask |
| Blue segmentation | `member_modules/mj_blue_sign_segmentation/blue.py` | Creates the blue HSV mask |
| Yellow segmentation | `member_modules/jy_yellow_sign_segmentation/yellow.py` | Creates the yellow HSV mask |
| Shape detection | `member_modules/lj_shape_detection/shape.py` | Classifies geometry and scores shape evidence |
| Shared segmentation | `segmentation.py` | Converts BGR to HSV and cleans masks |
| Candidate selection | `candidates.py` | Finds, measures, filters and ranks candidates |
| Pipeline | `pipeline.py` | Runs the complete process for one image |
| Dashboard | `dashboard.py` and `apps/web` | Shows every processing stage |

The colour modules are deliberately short. Each has one job: receive the HSV image and create a mask for its assigned colour with `cv2.inRange()`.

The HSV values are kept centrally in `default.yaml`, so they are recorded once rather than hidden in three duplicated functions.

Common work, such as morphology and contours, belongs in the shared pipeline because all colours use it.

## Processing flow

```text
Input BGR image
  -> HSV conversion
  -> red / blue / yellow masks
  -> mask cleaning
  -> external contours
  -> filled candidate silhouette
  -> convex hull and geometry measurements
  -> candidate ranking
  -> colour + shape result
```

## Step 1 — HSV colour segmentation

OpenCV reads ordinary images in BGR format. The project converts BGR to HSV because hue describes the basic colour, while saturation and value help reject grey, dark and weakly coloured pixels.

| Colour | Recorded HSV range |
|---|---|
| Red | H 0–12 and 165–179, S 70–255, V 45–255 |
| Blue | H 90–135, S 65–255, V 40–255 |
| Yellow | H 15–40, S 70–255, V 70–255 |

Red uses two ranges because OpenCV hue wraps from 179 back to 0, and red lies at both ends of that hue scale.

If asked why the values differ slightly from suggested starting values, say they are still one documented global set, adjusted only to cover normal brightness and colour variation. No value changes by filename or by image.

## Step 2 — Mask cleaning

All colour masks use a 3×3 elliptical opening to remove isolated noise.

Red also uses a 3×3 closing operation to reconnect small breaks in red borders caused by white digits, glare or compression artifacts.

## Step 3 — Contours and basic filtering

The project uses external contours. The outside contour represents the sign boundary, while inner digits, arrows and graphics should not be separate sign shapes.

Candidates must pass global filters for minimum area, size, extent, solidity, aspect ratio and excessive frame-border contact.

| Measurement | Meaning |
|---|---|
| Extent | Filled candidate area / bounding-box area |
| Solidity | Filled candidate area / convex-hull area |
| Circularity | `4π × area / perimeter²`; nearer 1 is more circle-like |
| Aspect ratio | Width compared with height |
| Polygon vertices | Corners from simplified hulls |
| Triangle fit | How tightly the hull fits its smallest enclosing triangle |

## Step 4 — Filled silhouette and convex hull

Early contours could be thin coloured borders. A sign can contain white centres, black arrows, digits, shadows and small gaps.

The current pipeline fills each outer contour in a local image before shape analysis, producing one solid sign-like silhouette.

It then uses a convex hull, like stretching a rubber band around the outside of the candidate. This reduces the effect of small dents, broken pixels and internal artwork.

## Step 5 — Shape detection

The hull is simplified at three nearby epsilon values: `0.012P`, `0.018P` and `0.024P`, where `P` is hull perimeter.

| Shape | Main evidence |
|---|---|
| Triangle | Two three-vertex votes, or strong triangle fit with suitable geometry |
| Square/rectangle | Two four-vertex votes and low triangle fit |
| Octagon | Eight vertices stay stable across all three approximations |
| Circle | High circularity and near-square rotated aspect ratio |
| Other | Weak or conflicting geometry |

Triangle is checked before circle. This stops a triangle with a rounded base or clipped point from being incorrectly labelled as a circle first.

Triangle fit comes from OpenCV's `minEnclosingTriangle()`. It measures how tightly the hull occupies its smallest enclosing triangle. It is ordinary contour geometry, not a learned template or classifier.

## Step 6 — Candidate ranking

One image may contain several red, blue or yellow objects. A tiny coloured letter, reflection or light can have neat geometry while the real sign is larger but has a broken border.

| Evidence | Weight | Reason |
|---|---:|---|
| Geometry score | 70% | Solidity, extent, circularity, aspect ratio and shape confidence |
| Soft scale evidence | 25% | Prevents tiny fragments automatically outranking plausible signs |
| Colour-mask support | 5% | Checks that the silhouette is supported by its own colour mask |

The score is an explainable heuristic, not a machine-learning probability.

Small or large candidates are not automatically removed; scale affects only their ranking, so a distant small sign can still be returned.

## How the result improved

| Stage | Problem | Global improvement |
|---|---|---|
| First baseline | Raw coloured borders produced unstable geometry | Fill the local silhouette and use a convex hull |
| Shape refinement | One polygon setting was sensitive to blur | Use three epsilon values and vertex voting |
| Triangle refinement | Some triangles had extra noisy corners | Add enclosing-triangle fit |
| Candidate refinement | Tiny coloured fragments outranked real signs | Add soft scale and colour support to ranking |
| Background refinement | Large colour regions at image edges appeared as candidates | Limit excessive border contact |

All changes are global. The code contains no filename checks, image hashes, per-image parameters or manual correction after detection.

## Final coursework audit record

| Metric | Result |
|---|---:|
| Detection | 81/84 (96.4%) |
| Correct colour | 79/84 (94.0%) |
| Correct shape | 70/84 (83.3%) |
| Correct colour and shape | 70/84 (83.3%) |

This is a coursework audit figure, not a promise of the same result on every real road scene.

Expected limitations include shadows, blur, low saturation, occlusion, perspective and coloured background objects.

## Dashboard demonstration

The dashboard exposes only **Image** and **Batch** inputs.

A single image shows the original image, raw HSV masks, cleaned masks, retained contours and the final result.

It also exposes area ratio, circularity, vertices, triangle fit, colour coverage and scale evidence. Use these views to explain how a result was reached.

## Common Q&A

### Why HSV instead of RGB/BGR?

HSV separates hue from brightness, so red, blue and yellow are easier to segment when illumination changes than raw BGR values.

### Why does red need two masks?

OpenCV hue wraps around. Red occurs near both 0 and 179, so two intervals are combined.

### Why use opening and closing?

Opening removes small isolated noise. Red closing reconnects small breaks in a red sign border.

### Why external contours?

The outside boundary represents the sign shape. Digits and arrows inside the sign should not become separate shape candidates.

### Why fill the contour and use a hull?

Filling restores the outer silhouette when a sign has a white or black interior. The hull reduces small contour defects from text, glare and missing pixels.

### Why use three epsilon values?

One approximation can be unstable. A shape that remains similar across nearby values gives more reliable vertex evidence.

### Is this deep learning or a trained model?

No. It is Python/OpenCV rule-based computer vision. The ranking score is made from visible geometry and mask evidence, not learned weights.

### Why is the result not 100%?

HSV cannot understand meaning. Coloured text, reflections, vegetation, vehicles and building panels can resemble a sign. Blur, shadows and occlusion also damage masks and contours.

### Can it read a speed-limit number?

No. It reports colour and shape only. OCR or a trained semantic classifier is outside this Part 1 component.

### Why are the member colour modules short?

The assignment component is colour segmentation. Each member owns one clear OpenCV operation—its colour mask—and one recorded HSV configuration.

The shared code avoids copying the same preprocessing and contour code three times. In the report, show the module function, its HSV range, one successful mask screenshot and one failure/limitation screenshot.

## Suggested report placement

- **Red member:** two red hue intervals and red closing.
- **Blue member:** blue HSV interval and opening.
- **Yellow member:** yellow HSV interval and opening.
- **Shape member:** filled silhouettes, hulls, vertices, circularity, aspect ratio and triangle fit.
- **Integration section:** shared candidate filtering, ranking and dashboard evidence.

For each member, include one success and one failure example and record the global parameter values from `default.yaml`.

## Run commands

```powershell
Set-Location C:\Part1MiniProject\Road-Sign-Detector-Part-1
& 'C:\MiniProject\.venv\Scripts\python.exe' .\dashboard.py
```

Open `http://127.0.0.1:8010`.

To rebuild the browser dashboard after source changes:

```powershell
Set-Location C:\Part1MiniProject\Road-Sign-Detector-Part-1\apps\web
npm run build
```
