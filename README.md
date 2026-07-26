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
