# Road Sign Detector - Part 1

An assignment-focused, rule-based road-sign colour and shape detector.

The detector uses Python, OpenCV and NumPy only. It converts an uploaded image
from BGR to HSV, makes red/blue/yellow masks, cleans them with morphology, and
then measures contours to identify circle, triangle, square/rectangle and
octagon candidates. It does not use YOLO, deep learning, OCR, training data or
filename-based rules.

## Project layout

| Path | Purpose |
|---|---|
| `member_modules/` | Red, blue, yellow and shape member components |
| `segmentation.py` | Shared HSV conversion and mask cleaning |
| `candidates.py` | Contour filtering and explainable candidate ranking |
| `pipeline.py` | Complete detector pipeline |
| `run_demo.py` | Batch command-line runner |
| `dashboard.py` | Image/Batch API and dashboard server |
| `apps/web/` | React dashboard showing each processing stage |
| `RESULTS.md` | Final concise result record |
| `PROJECT_QNA_GUIDE.md` | Method and presentation Q&A guide |

## Run the dashboard

The existing Python environment in the main project can be used:

```powershell
Set-Location C:\Part1MiniProject\Road-Sign-Detector-Part-1
& 'C:\MiniProject\.venv\Scripts\python.exe' .\dashboard.py
```

Open http://127.0.0.1:8010 in a browser.

If the dashboard browser files need rebuilding:

```powershell
Set-Location C:\Part1MiniProject\Road-Sign-Detector-Part-1\apps\web
npm ci
npm run build
```

## Run a batch from PowerShell

```powershell
Set-Location C:\Part1MiniProject\Road-Sign-Detector-Part-1
& 'C:\MiniProject\.venv\Scripts\python.exe' .\run_demo.py 'C:\path\to\your\images' --output .\results
```

The command creates annotated images, the three cleaned colour masks, crops,
and concise CSV/JSON result records in `results`.
