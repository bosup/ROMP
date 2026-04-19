# ROMP onset-metrics frontend

Zero-build Plotly UI over a small FastAPI backend. The app loads the demo
AIFS / NGCM / IMD 2015 rainfall, detects onset with ROMP's production
`momp.stats.detect.detect_onset`, and renders the five Milestone 1/2
metrics interactively.

This is deliberately minimal — no npm, no build step. If you want a
richer React experience, the `ROMP_frontend_POC` sibling repo has a
FastAPI + React/Vite stack with broader coverage of the original ROMP
pipeline; this frontend focuses narrowly on the new progression and
probabilistic metrics.

## Install

```bash
uv pip install -e '.[dev]' fastapi uvicorn
```

(`uv pip install -e .` plus `fastapi uvicorn` also works.)

## Run

```bash
./frontend/run.sh
```

Then open <http://127.0.0.1:8000>.

First request triggers onset detection over the demo data; this takes
~10 seconds on a laptop and is cached for subsequent requests.

## Endpoints

| Endpoint | Returns |
| --- | --- |
| `GET /api/health` | liveness ping |
| `GET /api/fields` | onset-DOY fields (obs + AIFS + ensemble mean) + selected init indices |
| `GET /api/metrics/crps` | per-cell censored CRPS |
| `GET /api/metrics/fss?thresholds=...&neighborhoods=...` | FSS sweep |
| `GET /api/metrics/displacement` | centroid shift + area bias sweep |
| `GET /api/metrics/progression?step=3` | IOE, extent, misplacement, SPS through the season |
| `GET /api/metrics/isochrones` | forecast + observed isochrones as line segments + Hausdorff/Fréchet |
| `GET /api/metrics/corp?tau=...` | MCB / DSC / UNC decomposition + calibration curve |

## Layout

```
frontend/
├── api/
│   └── app.py            FastAPI app — endpoints above
├── static/
│   ├── index.html        single page
│   ├── app.css           styling
│   └── app.js            Plotly rendering
├── run.sh
└── README.md
```
