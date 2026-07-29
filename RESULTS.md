# Final Part 1 Result Record

## Scope

This is a rule-based Python/OpenCV system that reports sign **colour** and
**shape** only. It uses HSV masks, morphology, contours, filled silhouettes,
convex-hull geometry and candidate ranking. It does not use YOLO, deep
learning, OCR, training images or filename lookup.

## Reviewed coursework audit

| Metric | Result |
|---|---:|
| Detection | 81/84 (96.4%) |
| Correct colour | 79/84 (94.0%) |
| Correct shape | 70/84 (83.3%) |
| Correct colour and shape | 70/84 (83.3%) |

## Final method

1. BGR-to-HSV conversion and red/blue/yellow masks.
2. 3×3 opening for all masks; 3×3 red closing.
3. External contours and global geometric filters.
4. Filled local silhouette and convex hull.
5. Three polygon approximations, circularity, aspect ratio and triangle fit.
6. Candidate ranking: geometry 70%, soft scale 25%, colour-mask support 5%.

## Known limitations

The system can still confuse a traffic sign with coloured text, reflections,
vegetation, vehicles or building panels. Blur, shadows, occlusion and low
saturation can also prevent a reliable mask or contour.

See [PROJECT_QNA_GUIDE.md](PROJECT_QNA_GUIDE.md) for the full method and
defence explanation.
