# Road Sign Detector Part 1

Offline Malaysian road-sign color and shape segmentation dashboard for Part 1 coursework.

## Run the Dashboard

Install Python dependencies in a Python 3.11 environment, then build the web dashboard:

```powershell
pip install -e .
cd apps\web
npm ci
npm run build
cd ..\..
python dashboard.py
```

Open <http://127.0.0.1:8010/> after the server starts.

For frontend development, run the backend on port 8000 and Vite separately:

```powershell
python dashboard.py --port 8000
cd apps\web
npm run dev
```

Then open <http://127.0.0.1:5173/>.

## Recheck the reviewed assignment inputs

The evaluator is optional: it keeps labels outside the project and accepts
their locations as arguments. With the folders used for this coursework, run:

```powershell
& 'C:\MiniProject\.venv\Scripts\python.exe' .\evaluate_manual_labels.py `
  'C:\MiniProject_OfficialBackup\Color Inputs' `
  '..\manual_review\manual_shape_colour_ground_truth.csv'
```
