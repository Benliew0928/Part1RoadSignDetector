# Final Part 1 Result Record

## Scope

This project is a rule-based Python/OpenCV detector for traffic-sign colour
and visible shape. It reports red, blue or yellow together with circle,
triangle, square/rectangle, octagon or other.

It does not use YOLO, deep learning, OCR, a trained model, filename rules,
stored answers, or image-specific parameter changes.

## Current reviewed assignment-set result

| Metric | Result |
|---|---:|
| Detection | 81/84 (96.4%) |
| Correct colour | 81/84 (96.4%) |
| Correct shape | 73/84 (86.9%) |
| Correct colour and shape | 73/84 (86.9%) |

This is a reviewed result on the assignment image set. It is not an
independent claim of real-world accuracy because the images informed earlier
global method development.

## Final fixed method

1. Convert the BGR input image to HSV.
2. Produce red, blue and yellow masks using one global HSV configuration.
3. Use 3x3 opening for all masks and 3x3 closing for red.
4. Find external colour contours and apply fixed area, extent, solidity,
   aspect-ratio and border-contact filters.
5. Refine each contour only in a small local mask region. Refinement must keep
   most of the original contour and cannot grow beyond the fixed limit.
6. Fill the accepted contour and calculate its convex hull.
7. Measure circle, fitted-ellipse, enclosing-triangle, perspective
   quadrilateral/rotated-rectangle and octagon fits.
8. Combine those fits with circularity and three polygon-approximation votes
   to choose one global shape label.
9. Rank overlapping candidates using geometry (70%), soft scale evidence
   (25%) and colour-mask support (5%).

## Important limitation

Coloured text, reflections, vehicles, buildings, shadows, blur, occlusion and
low saturation can still produce an incorrect mask or shape. The result should
be reported as a classical colour-and-shape baseline, not as semantic traffic
sign recognition.

See [PROJECT_QNA_GUIDE.md](PROJECT_QNA_GUIDE.md) for the full explanation.
