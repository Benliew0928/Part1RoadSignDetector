# Road Sign Detector Part 1 - Final Method and Q&A Guide

## 1. One-minute introduction

This project is a classical computer-vision system for the Part 1 road-sign
component. It finds red, blue and yellow sign-like regions, then reports the
visible shape as a circle, triangle, square/rectangle, octagon or other.

It uses Python, OpenCV and NumPy. It does not use YOLO, deep learning, OCR, a
trained model, filename rules, stored answers or manual image-by-image
corrections.

The scope is colour and shape only. A red circle is reported as a red circle;
the system does not claim to read a speed-limit number inside it.

## 2. Current result and honest interpretation

| Metric | Result |
|---|---:|
| Detection | 81/84 (96.4%) |
| Correct colour | 81/84 (96.4%) |
| Correct shape | 73/84 (86.9%) |
| Correct colour and shape | 73/84 (86.9%) |

This is a reviewed assignment-set result. It is not an independent claim of
real-world accuracy because the assignment images informed earlier global
method development. The detector itself does not read labels or alter its
behaviour for a particular image.

## 3. Technology

| Technology | Purpose |
|---|---|
| Python | Detector and dashboard backend |
| OpenCV (cv2) | HSV conversion, masks, morphology, contours, hulls and geometry |
| NumPy | Image-array operations |
| FastAPI | Upload API for the dashboard |
| React and TypeScript | Browser dashboard |

## 4. Code structure and member ownership

| Component | Main file | Responsibility |
|---|---|---|
| Red segmentation | member_modules/ben_red_sign_segmentation/red.py | Red HSV mask |
| Blue segmentation | member_modules/mj_blue_sign_segmentation/blue.py | Blue HSV mask |
| Yellow segmentation | member_modules/jy_yellow_sign_segmentation/yellow.py | Yellow HSV mask |
| Shape module | member_modules/lj_shape_detection/shape.py | Combines global shape evidence and chooses the label |
| Shared segmentation | segmentation.py | BGR-to-HSV conversion and mask cleaning |
| Candidate pipeline | candidates.py | Local refinement, contour measurement, shape fits and candidate ranking |
| Integration | pipeline.py | Runs the detector for one image |
| Dashboard | dashboard.py and apps/web | Shows input, masks, contours and result |

The member colour files are intentionally short. Each member owns one clear
task: create the HSV mask for the assigned colour with cv2.inRange(). Common
operations such as BGR-to-HSV conversion, morphology and contour analysis are
shared once rather than copied three times.

The actual runtime HSV settings are in DEFAULT_CONFIG in run_demo.py.
default.yaml is a readable mirror of those settings for the report.

## 5. Processing flow

    Input BGR image
      -> convert to HSV
      -> red, blue and yellow masks
      -> morphological cleaning
      -> external contours and fixed filters
      -> bounded local mask refinement
      -> filled silhouette and convex hull
      -> circle/ellipse/triangle/rectangle/octagon fit scores
      -> one global colour-and-shape result

## 6. HSV colour segmentation

HSV is used because hue represents the main colour while saturation and value
help reject grey, dark and weakly coloured pixels.

| Colour | HSV range |
|---|---|
| Red | H 0-12 and 165-179, S 70-255, V 45-255 |
| Blue | H 90-135, S 65-255, V 40-255 |
| Yellow | H 15-40, S 70-255, V 70-255 |

Red requires two ranges because OpenCV hue wraps from 179 back to 0. The same
ranges apply to every uploaded image.

## 7. Mask cleaning and candidate filtering

All masks use a 3x3 elliptical opening to remove isolated noise. Red also uses
a 3x3 closing operation to reconnect small breaks in red borders.

The detector finds external contours because the outside boundary represents a
sign better than arrows, text or digits inside it. A contour must pass fixed
global filters for area, width, height, extent, solidity, aspect ratio and
excessive contact with the image border.

| Measurement | Meaning |
|---|---|
| Extent | Filled candidate area / bounding-box area |
| Solidity | Filled candidate area / convex-hull area |
| Circularity | 4*pi*area/perimeter^2; nearer 1 is more circle-like |
| Aspect ratio | Width / height |
| Polygon vertices | Corner count from simplified hulls |
| Colour coverage | Portion of the filled candidate supported by the colour mask |

## 8. Local silhouette refinement

A sign border can be broken by glare, white symbols, compression or a small
shadow. Before filling a contour, the detector looks only in a fixed local
neighbourhood around it and applies a small closing operation.

This is deliberately bounded:

- The refined region must retain at least 55% overlap with the original seed.
- It cannot grow beyond 1.35 times the original contour area.
- It uses the same values for every colour and every image.

This can reconnect part of one broken sign border without silently merging a
nearby unrelated coloured object.

## 9. Final shape method

After refinement, the contour is filled and a convex hull is created. Filling
restores the outer sign silhouette when a sign has a white or black interior.
The hull is like stretching a rubber band around the outside, reducing small
dents caused by text or missing pixels.

The hull is simplified with three nearby epsilon values: 0.012P, 0.018P and
0.024P, where P is the hull perimeter. The system calculates the following
global fit evidence.

| Shape | Evidence |
|---|---|
| Circle | Overlap with a fitted enclosing circle, plus circularity |
| Perspective circle | Overlap with a fitted ellipse; the ellipse must not be too narrow |
| Triangle | Hull coverage of its minimum enclosing triangle, plus three-corner votes |
| Square/rectangle | Overlap with a rotated rectangle or four-corner perspective quadrilateral |
| Octagon | Octagon overlap and exactly eight stable corners at all three epsilon values |

All four shape scores are calculated for every candidate. The shape with the
strongest evidence is selected only if it passes the one global minimum score
of 0.74. Otherwise the candidate is labelled other.

The strict stable-corner requirement for octagons is important: a smooth circle
can look eight-sided at one polygon approximation, but it should not be called
an octagon unless all three approximations agree.

## 10. Candidate ranking

An image can contain a coloured reflection, letter or vehicle as well as a
road sign. Retained candidates are ranked with global evidence.

| Evidence | Weight | Reason |
|---|---:|---|
| Geometry | 70% | Solidity, extent, circularity, aspect ratio and broad shape support |
| Soft scale evidence | 25% | Stops tiny fragments automatically outranking plausible signs |
| Colour-mask support | 5% | Confirms that the silhouette has support from its own colour mask |

Scale is not a hard deletion rule, so a small distant sign can still be
returned. The displayed score is a heuristic, not a trained probability.

## 11. Why the earlier 100% result was rejected

The earlier 100% result was achieved by repeatedly comparing output with known
answers for all 84 assignment images and changing the method until every known
answer matched. That is data leakage and overfitting.

It did not prove that one general 100% method exists. It proved only that a
method can be tailored to a known set after seeing its answers. Reporting that
as genuine detector accuracy would be unfair.

The final project uses global rules only. Its 86.9% reviewed result is more
credible because no candidate receives a file-specific exception or label.

## 12. Dashboard demonstration

The dashboard intentionally contains only Image and Batch inputs. For a single
image, it displays:

1. Original image
2. Raw HSV masks
3. Cleaned masks
4. Retained contours
5. Final annotated result

For each detected candidate, the dashboard exposes circularity, vertex votes,
circle fit, ellipse fit, triangle fit, rectangle fit, octagon fit, local
refinement ratio, colour coverage and scale evidence. These provide visible
evidence for a presentation or Q&A.

## 13. Common Q&A answers

### Is this deep learning or YOLO?

No. It is a rule-based OpenCV system. It has no trained weights and no model
file.

### Why HSV instead of BGR/RGB?

HSV separates basic colour from brightness. It is easier to define red, blue
and yellow ranges in HSV than in raw BGR values.

### Why does red need two HSV ranges?

The OpenCV hue scale wraps around. Red appears near both 0 and 179.

### Why use a convex hull?

The hull reduces small dents and gaps caused by lettering, glare or compression
while retaining the outer visible geometry.

### How does the system handle perspective?

A circular sign viewed from an angle can look elliptical, so the system checks
both circle and ellipse fits. A rectangular sign can look like a trapezoid, so
it also checks rotated rectangles and four-corner quadrilaterals.

### Does an ellipse automatically mean a circle sign?

No. The ellipse must fit the silhouette strongly, have a plausible axis ratio,
and beat the other global shape scores.

### Why is the result not 100%?

Classical colour and contour rules cannot understand meaning. Blur, shadows,
occlusion, perspective, coloured text, vehicles and background panels can all
damage a mask or imitate a sign.

### How should member work be written in the report?

For each colour member, show the assigned mask function, recorded HSV range,
one successful mask and one honest limitation. For the shape member, show the
contour/hull and the shape-fit measurements. Put morphology, local refinement,
candidate ranking, dashboard integration and final results in a shared
integration section.

## 14. Run commands

    Set-Location C:\Part1MiniProject\Road-Sign-Detector-Part-1
    & 'C:\MiniProject\.venv\Scripts\python.exe' .\dashboard.py

Open http://127.0.0.1:8010.

To rebuild the browser dashboard after source changes:

    Set-Location C:\Part1MiniProject\Road-Sign-Detector-Part-1\apps\web
    npm run build
