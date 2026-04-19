"""ROMP metrics frontend — FastAPI app.

Thin routing layer. Data discovery lives in ``catalog.py``, onset-DOY
construction in ``onset.py``, metric serialization in ``metrics.py``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import metrics as M
from .catalog import load_catalog, model_by_key, obs_source, shared_years
from .onset import (
    OnsetParams, Region,
    available_inits, best_init_for, get_forecast_onset, get_obs_onset,
    onset_range,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_catalog()  # warm the catalog cache
    yield


app = FastAPI(title="ROMP metrics API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _parse_params(
    wet_init: float, wet_spell: int, wet_threshold: float,
    dry_spell: int, dry_threshold: float, dry_extent: int,
) -> OnsetParams:
    return OnsetParams(
        wet_init=wet_init, wet_spell=wet_spell, wet_threshold=wet_threshold,
        dry_spell=dry_spell, dry_threshold=dry_threshold, dry_extent=dry_extent,
    )


def _parse_region(
    lat_min: float | None, lat_max: float | None,
    lon_min: float | None, lon_max: float | None,
) -> Region:
    return Region(lat_min=lat_min, lat_max=lat_max,
                  lon_min=lon_min, lon_max=lon_max)


OnsetQuery = dict(
    wet_init=(1.0, "Wet-day rainfall threshold (mm)"),
    wet_spell=(3, "Min consecutive wet days"),
    wet_threshold=(20.0, "Cumulative wet-spell rainfall (mm)"),
    dry_spell=(7, "Dry-spell length (days)"),
    dry_threshold=(1.0, "Dry-day rainfall threshold (mm)"),
    dry_extent=(0, "Search extent after wet spell"),
)


def onset_deps(
    wet_init: float = Query(OnsetQuery["wet_init"][0]),
    wet_spell: int = Query(OnsetQuery["wet_spell"][0]),
    wet_threshold: float = Query(OnsetQuery["wet_threshold"][0]),
    dry_spell: int = Query(OnsetQuery["dry_spell"][0]),
    dry_threshold: float = Query(OnsetQuery["dry_threshold"][0]),
    dry_extent: int = Query(OnsetQuery["dry_extent"][0]),
) -> OnsetParams:
    return _parse_params(wet_init, wet_spell, wet_threshold,
                         dry_spell, dry_threshold, dry_extent)


def region_deps(
    lat_min: float | None = Query(None),
    lat_max: float | None = Query(None),
    lon_min: float | None = Query(None),
    lon_max: float | None = Query(None),
) -> Region:
    return _parse_region(lat_min, lat_max, lon_min, lon_max)


# ---------------------------------------------------------------------------
# catalog + config
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/api/catalog")
def catalog():
    cat = load_catalog()
    return {
        "root": cat["root"],
        "models": [m.to_json() for m in cat["models"]],
        "obs": cat["obs"].to_json() if cat["obs"] else None,
        "shared_years": list(shared_years()),
        "onset_defaults": {k: v[0] for k, v in OnsetQuery.items()},
        "onset_docs": {k: v[1] for k, v in OnsetQuery.items()},
    }


@app.get("/api/inits")
def inits(model: str, year: int):
    m = model_by_key(model)
    times = available_inits(m, year)
    return {
        "model": m.key, "year": year, "n": len(times),
        "inits": [t.isoformat() for t in times],
    }


def _resolve_init(model_key: str, year: int, init: int | Literal["auto"] | None,
                  params: OnsetParams, obs_lo: float, obs_hi: float) -> int:
    """Resolve ``init`` param (int index, 'auto', or None→auto)."""
    if init is None or init == "auto":
        return best_init_for(model_by_key(model_key), obs_lo, obs_hi, params, year)
    try:
        return int(init)
    except (TypeError, ValueError):
        raise HTTPException(400, f"invalid init '{init}'")


def _fields_for(model_key: str, year: int, init: int | str | None,
                params: OnsetParams, region: Region):
    obs = get_obs_onset(year, params)
    obs_lo, obs_hi = onset_range(obs)
    idx = _resolve_init(model_key, year, init, params, obs_lo, obs_hi)
    fcst = get_forecast_onset(model_key, year, idx, params)
    if not region.is_empty():
        obs = region.crop(obs)
        fcst = region.crop(fcst)
    ens = fcst if "member" in fcst.dims else None
    det = ens.mean("member", skipna=True) if ens is not None else fcst
    return {"obs": obs, "fcst_det": det, "ens": ens, "init_idx": idx}


# ---------------------------------------------------------------------------
# state + metrics
# ---------------------------------------------------------------------------
@app.get("/api/state")
def state(
    model: str, year: int,
    init: str | None = None,
    params: OnsetParams = Depends(onset_deps),
    region: Region = Depends(region_deps),
):
    """Baseline onset fields + suggested isochrone days for a config."""
    bundle = _fields_for(model, year, init, params, region)
    obs_rng = onset_range(bundle["obs"])
    fcst_rng = onset_range(bundle["fcst_det"])
    ens_rng = onset_range(bundle["ens"]) if bundle["ens"] is not None else (None, None)

    lo = max(obs_rng[0], fcst_rng[0]) + 2
    hi = min(obs_rng[1], fcst_rng[1]) - 2
    import numpy as np
    iso_days = [int(round(x)) for x in np.linspace(lo, hi, 4)] if hi > lo else []
    iso_days = sorted(set(iso_days))

    return {
        "model": model, "year": year, "init_idx": bundle["init_idx"],
        "params": params.to_json(),
        "region": {"lat_min": region.lat_min, "lat_max": region.lat_max,
                   "lon_min": region.lon_min, "lon_max": region.lon_max},
        "is_ensemble": bundle["ens"] is not None,
        "n_members": int(bundle["ens"].sizes["member"]) if bundle["ens"] is not None else 1,
        "obs_onset": M.field_payload(bundle["obs"]),
        "fcst_onset": M.field_payload(bundle["fcst_det"]),
        "ens_mean_onset": (M.field_payload(bundle["ens"].mean("member", skipna=True))
                           if bundle["ens"] is not None else None),
        "obs_range": obs_rng,
        "fcst_range": fcst_rng,
        "ens_range": ens_rng,
        "iso_days": iso_days,
    }


@app.get("/api/metrics/crps")
def crps(
    model: str, year: int, init: str | None = None, season_end: int = 220,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    b = _fields_for(model, year, init, params, region)
    if b["ens"] is None:
        ens_like = b["fcst_det"].expand_dims({"member": [0]}).transpose("member", "lat", "lon")
        return M.compute_crps(ens_like, b["obs"], season_end=season_end)
    return M.compute_crps(b["ens"], b["obs"], season_end=season_end)


@app.get("/api/metrics/fss")
def fss_route(
    model: str, year: int, init: str | None = None,
    thresholds: str | None = None,
    neighborhoods: str = "1,3,5",
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    b = _fields_for(model, year, init, params, region)
    thr = ([int(t) for t in thresholds.split(",")] if thresholds
           else _suggest_thresholds(b))
    nbr = [int(n) for n in neighborhoods.split(",")]
    return M.compute_fss(b["fcst_det"], b["obs"], thresholds=thr, neighborhoods=nbr)


@app.get("/api/metrics/displacement")
def displacement(
    model: str, year: int, init: str | None = None,
    thresholds: str | None = None,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    b = _fields_for(model, year, init, params, region)
    thr = ([int(t) for t in thresholds.split(",")] if thresholds
           else _suggest_thresholds(b))
    return M.compute_displacement(b["fcst_det"], b["obs"], thresholds=thr)


@app.get("/api/metrics/progression")
def progression(
    model: str, year: int, init: str | None = None,
    step: int = 3,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    b = _fields_for(model, year, init, params, region)
    lo, hi = _progression_window(b)
    days = list(range(int(lo), int(hi) + 1, max(1, step)))
    return M.compute_progression(b["fcst_det"], b["ens"], b["obs"], days=days)


@app.get("/api/metrics/isochrones")
def isochrones(
    model: str, year: int, init: str | None = None,
    days: str | None = None,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    b = _fields_for(model, year, init, params, region)
    iso = ([int(d) for d in days.split(",")] if days else _suggest_thresholds(b))
    return M.compute_isochrones(b["fcst_det"], b["obs"], days=iso)


@app.get("/api/metrics/corp")
def corp(
    model: str, year: int, init: str | None = None,
    tau: int | None = None, season_end: int = 220,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    b = _fields_for(model, year, init, params, region)
    if tau is None:
        thr = _suggest_thresholds(b)
        tau = thr[len(thr) // 2] if thr else 170
    return M.compute_corp(b["ens"], b["fcst_det"], b["obs"],
                          tau=tau, season_end=season_end)


@app.get("/api/compare")
def compare(
    models: str,
    year: int,
    init: str | None = None,
    params: OnsetParams = Depends(onset_deps),
    region: Region = Depends(region_deps),
):
    """Cross-model summary: season IOE/SPS, CRPS mean, CORP MCB/DSC/UNC
    per selected model against the same obs/year."""
    keys = [k.strip() for k in models.split(",") if k.strip()]
    rows = []
    for mk in keys:
        try:
            b = _fields_for(mk, year, init, params, region)
        except (KeyError, ValueError) as e:
            rows.append({"model": mk, "error": str(e)})
            continue
        lo, hi = _progression_window(b)
        days = list(range(int(lo), int(hi) + 1, 3))
        prog = M.compute_progression(b["fcst_det"], b["ens"], b["obs"], days=days)
        ens_like = (b["ens"] if b["ens"] is not None
                    else b["fcst_det"].expand_dims({"member": [0]}).transpose("member", "lat", "lon"))
        crps_out = M.compute_crps(ens_like, b["obs"], season_end=220)
        thr = _suggest_thresholds(b)
        tau = thr[len(thr) // 2] if thr else 170
        corp_out = M.compute_corp(b["ens"], b["fcst_det"], b["obs"],
                                  tau=tau, season_end=220)
        model_info = model_by_key(mk)
        rows.append({
            "model": mk,
            "label": model_info.label,
            "is_ensemble": b["ens"] is not None,
            "n_members": (int(b["ens"].sizes["member"])
                          if b["ens"] is not None else 1),
            "init_idx": b["init_idx"],
            "progression": {
                "days": prog["days"],
                "ioe_km2": prog["ioe_km2"],
                "sps_km2": prog["sps_km2"],
                "extent_km2": prog["extent_km2"],
                "misplacement_km2": prog["misplacement_km2"],
                "season": prog["season"],
            },
            "crps": {
                "mean": crps_out["mean"],
                "max": crps_out["max"],
                "n_finite": crps_out["n_finite"],
            },
            "corp": {
                "tau": corp_out["tau"],
                "bs": corp_out["mean_score"],
                "mcb": corp_out["mcb"],
                "dsc": corp_out["dsc"],
                "unc": corp_out["unc"],
            },
        })
    return {"year": year, "params": params.to_json(), "rows": rows}


def _suggest_thresholds(bundle) -> list[int]:
    import numpy as np
    obs_rng = onset_range(bundle["obs"])
    fcst_rng = onset_range(bundle["fcst_det"])
    lo = max(obs_rng[0], fcst_rng[0]) + 2
    hi = min(obs_rng[1], fcst_rng[1]) - 2
    if hi <= lo:
        return []
    return sorted({int(round(x)) for x in np.linspace(lo, hi, 4)})


def _progression_window(bundle) -> tuple[int, int]:
    obs_rng = onset_range(bundle["obs"])
    fcst_rng = onset_range(bundle["fcst_det"])
    lo = min(obs_rng[0], fcst_rng[0]) - 2
    hi = max(obs_rng[1], fcst_rng[1]) + 2
    return int(lo), int(hi)


# ---------------------------------------------------------------------------
# static frontend
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")
