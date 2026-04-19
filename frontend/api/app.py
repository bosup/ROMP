"""ROMP metrics frontend — FastAPI backend.

Exposes the Milestone-1 and Milestone-2 metrics over a small JSON API so a
browser can render them interactively. The demo AIFS / NGCM / IMD rainfall
fields at ``demo/data`` are processed through ROMP's production
``momp.stats.detect.detect_onset`` on first request and cached in-process;
subsequent requests hit the cache.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from momp.stats.detect import detect_onset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO = REPO_ROOT / "demo" / "data"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

ONSET_KW = dict(wet_init=1.0, wet_spell=3, dry_spell=7, dry_threshold=1.0, dry_extent=0)
WET_THRESH = 20.0

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# onset-DOY field construction (wraps the real detector)
# ---------------------------------------------------------------------------
def _first_onset_doy(rain_series: np.ndarray, start_doy: int) -> float:
    n = rain_series.size
    for offset in range(n - ONSET_KW["wet_spell"] + 1):
        if detect_onset(
            day=offset + 1,
            forecast_series=rain_series,
            thresh=WET_THRESH,
            **ONSET_KW,
        ):
            return float(start_doy + offset)
    return float("nan")


def _obs_onset_field() -> xr.DataArray:
    rain = xr.open_dataset(DEMO / "obs" / "2015.nc")["RAINFALL"]
    times = pd.to_datetime(rain["TIME"].values)
    sub = rain.isel(TIME=np.where((times.month >= 5) & (times.month <= 9))[0])
    start_doy = int(pd.Timestamp(sub["TIME"].values[0]).dayofyear)
    vals = sub.values
    out = np.full(vals.shape[1:], np.nan)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = _first_onset_doy(vals[:, i, j], start_doy)
    return xr.DataArray(
        out, coords={"lat": sub["lat"].values, "lon": sub["lon"].values},
        dims=("lat", "lon"), name="onset_doy",
    )


def _fcst_onset_field(da: xr.DataArray, init_select: int) -> xr.DataArray:
    has_member = "number" in da.dims
    sel = da.isel(time=init_select)
    start_doy = pd.Timestamp(da["time"].values[init_select]).dayofyear
    if has_member:
        arr = sel.transpose("number", "day", "lat", "lon").values
        M, _, Ny, Nx = arr.shape
        out = np.full((M, Ny, Nx), np.nan)
        for m in range(M):
            for i in range(Ny):
                for j in range(Nx):
                    out[m, i, j] = _first_onset_doy(arr[m, :, i, j], start_doy)
        return xr.DataArray(
            out, dims=("member", "lat", "lon"),
            coords={"member": sel["number"].values,
                    "lat": sel["lat"].values, "lon": sel["lon"].values},
        )
    arr = sel.transpose("day", "lat", "lon").values
    out = np.full(arr.shape[1:], np.nan)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = _first_onset_doy(arr[:, i, j], start_doy)
    return xr.DataArray(
        out, dims=("lat", "lon"),
        coords={"lat": sel["lat"].values, "lon": sel["lon"].values},
    )


def _pick_overlapping_init(da: xr.DataArray, obs_lo: float, obs_hi: float):
    best_idx, best_overlap, best_range = 0, -1.0, (np.nan, np.nan)
    for k in range(da.sizes["time"]):
        f = _fcst_onset_field(da, init_select=k)
        v = f.values[np.isfinite(f.values)]
        if v.size == 0:
            continue
        lo, hi = float(v.min()), float(v.max())
        ov = max(0.0, min(hi, obs_hi) - max(lo, obs_lo))
        if ov > best_overlap:
            best_overlap, best_idx, best_range = ov, k, (lo, hi)
    return best_idx, best_range


def _build_cache() -> dict[str, Any]:
    obs = _obs_onset_field()
    obs_v = obs.values[np.isfinite(obs.values)]
    if obs_v.size == 0:
        raise RuntimeError("detector found no observed onset in demo data")
    obs_lo, obs_hi = float(obs_v.min()), float(obs_v.max())

    aifs_da = xr.open_dataset(DEMO / "aifs" / "2015.nc")["tp"]
    ngcm_da = xr.open_dataset(DEMO / "ngcm" / "2015.nc")["tp"]
    aifs_init, aifs_rng = _pick_overlapping_init(aifs_da, obs_lo, obs_hi)
    ngcm_init, ngcm_rng = _pick_overlapping_init(ngcm_da, obs_lo, obs_hi)

    aifs = _fcst_onset_field(aifs_da, init_select=aifs_init)
    ens = _fcst_onset_field(ngcm_da, init_select=ngcm_init)

    lo = max(obs_lo, aifs_rng[0], ngcm_rng[0]) + 2
    hi = min(obs_hi, aifs_rng[1], ngcm_rng[1]) - 2
    iso_days = sorted({int(round(x)) for x in np.linspace(lo, hi, 4)})

    return {
        "obs": obs, "aifs": aifs, "ens": ens,
        "iso_days": iso_days,
        "aifs_init_idx": aifs_init, "ngcm_init_idx": ngcm_init,
        "obs_range": (obs_lo, obs_hi),
        "aifs_range": aifs_rng, "ngcm_range": ngcm_rng,
    }


def get_cache() -> dict[str, Any]:
    with _cache_lock:
        if not _cache:
            _cache.update(_build_cache())
    return _cache


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the cache up-front so the first request is fast.
    get_cache()
    yield


app = FastAPI(title="ROMP metrics API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _as_list(arr) -> list:
    """NumPy/xr values -> JSON-safe list, with NaN -> None."""
    a = np.asarray(arr, dtype=float)
    return [None if not np.isfinite(x) else float(x) for x in a.ravel()]


def _field_payload(da: xr.DataArray) -> dict:
    return {
        "lat": da["lat"].values.tolist(),
        "lon": da["lon"].values.tolist(),
        "values": [[None if not np.isfinite(v) else float(v) for v in row]
                   for row in np.asarray(da.values, dtype=float)],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/api/fields")
def fields():
    c = get_cache()
    return {
        "obs": _field_payload(c["obs"]),
        "aifs": _field_payload(c["aifs"]),
        "ens_mean": _field_payload(c["ens"].mean("member", skipna=True)),
        "iso_days": c["iso_days"],
        "aifs_init_idx": c["aifs_init_idx"],
        "ngcm_init_idx": c["ngcm_init_idx"],
        "obs_range": c["obs_range"],
        "aifs_range": c["aifs_range"],
        "ngcm_range": c["ngcm_range"],
        "ens_members": int(c["ens"].sizes["member"]),
    }


@app.get("/api/metrics/crps")
def crps_field():
    from momp.metrics.crps import censored_crps_field
    c = get_cache()
    crps = censored_crps_field(c["ens"], c["obs"], season_end=220)
    return {
        "field": _field_payload(crps),
        "mean": float(np.nanmean(crps.values)),
        "max": float(np.nanmax(crps.values)),
        "n_finite": int(np.isfinite(crps.values).sum()),
    }


@app.get("/api/metrics/fss")
def fss_sweep(
    thresholds: str | None = None,
    neighborhoods: str | None = None,
):
    from momp.metrics.neighborhood import fss
    c = get_cache()
    thr = ([int(t) for t in thresholds.split(",")] if thresholds else c["iso_days"])
    nbr = ([int(n) for n in neighborhoods.split(",")] if neighborhoods else [1, 3, 5])
    out = fss(c["aifs"], c["obs"], thresholds=thr, neighborhoods=nbr)
    vals = np.asarray(out.values, dtype=float)
    return {
        "thresholds": thr, "neighborhoods": nbr,
        "fss": [[None if not np.isfinite(v) else float(v) for v in row] for row in vals],
    }


@app.get("/api/metrics/displacement")
def displacement():
    from momp.metrics.displacement import displacement_bias_sweep
    c = get_cache()
    ds = displacement_bias_sweep(c["aifs"], c["obs"], thresholds=c["iso_days"])
    return {
        "thresholds": c["iso_days"],
        "delta_lat_deg": _as_list(ds["delta_lat_deg"].values),
        "delta_lon_deg": _as_list(ds["delta_lon_deg"].values),
        "great_circle_km": _as_list(ds["great_circle_km"].values),
        "area_bias_fraction": _as_list(ds["area_bias_fraction"].values),
    }


@app.get("/api/metrics/progression")
def progression(step: int = 3):
    from momp.metrics.progression import (
        integrated_onset_error, spatial_probability_score,
    )
    c = get_cache()
    obs = c["obs"]
    lo = int(min(c["aifs_range"][0], c["ngcm_range"][0]))
    hi = int(max(c["aifs_range"][1], c["ngcm_range"][1])) + 1
    days = list(range(lo, hi + 1, max(1, step)))
    ioe = integrated_onset_error(c["aifs"], obs, days=days)
    sps = spatial_probability_score(c["ens"], obs, days=days)
    return {
        "days": list(days),
        "ioe_km2": _as_list(ioe["ioe_km2"].values),
        "extent_km2": _as_list(ioe["extent_km2"].values),
        "misplacement_km2": _as_list(ioe["misplacement_km2"].values),
        "sps_km2": _as_list(sps["sps_km2"].values),
        "season": {
            "ioe_km2_day": float(ioe["ioe_season_km2_day"]),
            "extent_km2_day": float(ioe["extent_season_km2_day"]),
            "misplacement_km2_day": float(ioe["misplacement_season_km2_day"]),
            "sps_km2_day": float(sps["sps_season_km2_day"]),
        },
    }


@app.get("/api/metrics/isochrones")
def isochrones():
    """Extract forecast and observed isochrones at the shared-range days.

    Returns line segments as (lon, lat) arrays plus Hausdorff/Fréchet
    distances per day.
    """
    from momp.graphics.isochrone import (
        extract_isochrone, isochrone_distance_sweep,
    )
    c = get_cache()
    out = []
    for d in c["iso_days"]:
        f_segs = extract_isochrone(c["aifs"], float(d))
        o_segs = extract_isochrone(c["obs"], float(d))
        out.append({
            "day": int(d),
            "forecast": [seg.tolist() for seg in f_segs],
            "observed": [seg.tolist() for seg in o_segs],
        })
    sweep = isochrone_distance_sweep(c["aifs"], c["obs"], days=c["iso_days"])
    return {
        "isochrones": out,
        "days": c["iso_days"],
        "hausdorff_km": _as_list(sweep["hausdorff_km"].values),
        "frechet_km": _as_list(sweep["frechet_km"].values),
        "n_segments_fcst": [int(v) for v in sweep["n_segments_fcst"].values],
        "n_segments_obs": [int(v) for v in sweep["n_segments_obs"].values],
    }


@app.get("/api/metrics/corp")
def corp(tau: int | None = None):
    from momp.graphics.corp_reliability import (
        corp_decompose_brier, _consolidate_curve,
    )
    c = get_cache()
    if tau is None:
        tau = int(c["iso_days"][len(c["iso_days"]) // 2])
    m = c["ens"].values
    p = np.where(np.isnan(m) | (m > 250), 0.0, (m <= tau).astype(float)).mean(axis=0)
    y = np.where(
        np.isfinite(c["obs"].values) & (c["obs"].values <= tau), 1.0, 0.0
    )
    decomp = corp_decompose_brier(p.ravel(), y.ravel())
    f_rep, c_rep = _consolidate_curve(decomp.forecast_prob, decomp.calibrated_y)
    return {
        "tau": tau,
        "mean_score": float(decomp.mean_score),
        "mcb": float(decomp.mcb),
        "dsc": float(decomp.dsc),
        "unc": float(decomp.unc),
        "identity_residual": float(
            (decomp.mcb - decomp.dsc + decomp.unc) - decomp.mean_score
        ),
        "n": int(decomp.n),
        "curve": {
            "forecast_prob": [float(x) for x in f_rep],
            "calibrated_prob": [float(x) for x in c_rep],
        },
        "forecast_prob_histogram": _as_list(decomp.forecast_prob),
    }


# Serve the single-page frontend from /.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")
