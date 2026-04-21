"""ROMP metrics frontend — FastAPI app.

Thin routing layer. Data discovery lives in ``catalog.py``, onset-DOY
construction in ``onset.py``, metric serialization in ``metrics.py``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import aggregate as AGG
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


@app.exception_handler(FileNotFoundError)
async def _missing_file(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def _bad_key(request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


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
    import os
    cat = load_catalog()
    return {
        "root": cat["root"],
        "models": [m.to_json() for m in cat["models"]],
        "obs": cat["obs"].to_json() if cat["obs"] else None,
        "shared_years": list(shared_years()),
        "onset_defaults": {k: v[0] for k, v in OnsetQuery.items()},
        "onset_docs": {k: v[1] for k, v in OnsetQuery.items()},
        "land_mask": os.environ.get("ROMP_LAND_MASK", "") or None,
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


def _align_fcst_to(fcst, obs):
    """Conform the forecast to the obs grid when they differ. Obs is the
    authoritative measurement; downsampling the model to the obs grid
    avoids inflating area-weighted metrics by N where N is the upsample
    ratio. For ensembles, we keep the member dim and interp per-member."""
    import numpy as np
    if (fcst.sizes.get("lat") == obs.sizes.get("lat") and
            fcst.sizes.get("lon") == obs.sizes.get("lon") and
            np.array_equal(fcst["lat"].values, obs["lat"].values) and
            np.array_equal(fcst["lon"].values, obs["lon"].values)):
        return fcst
    # interp_like preserves extra dims (including 'member'), using nearest.
    return fcst.interp_like(obs, method="nearest")


_LAND_MASK_CACHE: dict = {}


def _land_mask_for(da):
    """Return a (lat, lon) boolean mask of land cells for the region set
    via the ``ROMP_LAND_MASK`` env var (e.g. ``India``). False elsewhere
    (ocean) so masked fields become NaN there. No-op if unset."""
    import os
    region = os.environ.get("ROMP_LAND_MASK", "").strip()
    if not region:
        return None
    lat_key = tuple(da["lat"].values.round(4).tolist())
    lon_key = tuple(da["lon"].values.round(4).tolist())
    key = (region, lat_key, lon_key)
    if key in _LAND_MASK_CACHE:
        return _LAND_MASK_CACHE[key]
    import numpy as np
    try:
        import regionmask
        # Use natural_earth's countries; region kwarg names the country.
        # Exact case-insensitive match first so inputs like "Niger" don't
        # silently pick up "Nigeria" (idx 99 vs 148) or "Korea" -> "South
        # Korea" etc. Only fall back to unambiguous substring match.
        countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
        names_lower = [nm.lower() for nm in countries.names]
        target = region.lower()
        idx = None
        if target in names_lower:
            idx = names_lower.index(target)
        else:
            hits = [i for i, nm in enumerate(names_lower) if target in nm]
            if len(hits) == 1:
                idx = hits[0]
            elif len(hits) > 1:
                matches = ", ".join(repr(countries.names[i]) for i in hits[:5])
                raise ValueError(
                    f"region {region!r} is ambiguous — matches {matches}"
                    + (" and more" if len(hits) > 5 else "")
                    + "; use an exact country name"
                )
        if idx is None:
            raise ValueError(f"region {region!r} not found in natural_earth countries")
        mask2d = countries.mask(da["lon"], da["lat"]) == idx
        arr = np.asarray(mask2d.values, dtype=bool)
    except Exception as exc:  # fall back to no mask on failure
        import sys
        print(f"[warn] land mask for {region!r} failed: {exc}", file=sys.stderr)
        arr = None
    _LAND_MASK_CACHE[key] = arr
    return arr


def _apply_land_mask(da):
    import numpy as np
    mask = _land_mask_for(da)
    if mask is None:
        return da
    # Broadcast to extra dims and set non-land to NaN.
    if "member" in da.dims:
        mfull = mask[np.newaxis, :, :] | np.zeros(
            (da.sizes["member"], 1, 1), dtype=bool
        )
    else:
        mfull = mask
    return da.where(mfull)


def _fields_for(model_key: str, year: int, init: int | str | None,
                params: OnsetParams, region: Region):
    obs = get_obs_onset(year, params)
    obs_lo, obs_hi = onset_range(obs)
    idx = _resolve_init(model_key, year, init, params, obs_lo, obs_hi)
    fcst = get_forecast_onset(model_key, year, idx, params)
    # Conform forecast -> obs (obs is the authoritative grid).
    fcst = _align_fcst_to(fcst, obs)
    # Optional land-only masking: cells outside the named country go NaN.
    obs = _apply_land_mask(obs)
    fcst = _apply_land_mask(fcst)
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


def _multi_year_bundles(model_key, years, init, params, region):
    """Detect onset for each requested year. Skip years with no data;
    raise FileNotFoundError if no year succeeds."""
    out = []
    errors = []
    for y in years:
        try:
            out.append((int(y), _fields_for(model_key, int(y), init, params, region)))
        except FileNotFoundError as e:
            errors.append((int(y), str(e)))
            continue
    if not out:
        msg = f"no data for model={model_key!r} across years={list(years)!r}"
        if errors:
            msg += f" (first miss: {errors[0][1]})"
        raise FileNotFoundError(msg)
    return out


def _global_range(bundles, key):
    import numpy as np
    los, his = [], []
    for _, b in bundles:
        lo, hi = onset_range(b[key])
        if np.isfinite(lo) and np.isfinite(hi):
            los.append(lo); his.append(hi)
    if not los:
        return (float("nan"), float("nan"))
    return (min(los), max(his))


def _global_thresholds(bundles, n=4) -> list[int]:
    import numpy as np
    obs = _global_range(bundles, "obs")
    fcst = _global_range(bundles, "fcst_det")
    lo = max(obs[0], fcst[0]) + 2
    hi = min(obs[1], fcst[1]) - 2
    if hi <= lo:
        return []
    return sorted({int(round(x)) for x in np.linspace(lo, hi, n)})


def _global_progression_window(bundles) -> tuple[int, int]:
    obs = _global_range(bundles, "obs")
    fcst = _global_range(bundles, "fcst_det")
    lo = min(obs[0], fcst[0]) - 2
    hi = max(obs[1], fcst[1]) + 2
    return int(lo), int(hi)


def _resolve_years(years_arg: str | None, year: int | None) -> list[int]:
    try:
        return AGG.expand_years(years_arg, year)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def _years_meta(bundles, requested) -> dict:
    return {"requested": list(requested),
            "computed": [int(y) for y, _ in bundles]}


# ---------------------------------------------------------------------------
# metric routes (single-year passthrough, multi-year aggregation)
# ---------------------------------------------------------------------------

@app.get("/api/metrics/crps")
def crps(
    model: str,
    year: int | None = None, years: str | None = None,
    init: str | None = None, season_end: int = 220,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    per_year = []
    for _, b in bundles:
        ens = (b["ens"] if b["ens"] is not None
               else b["fcst_det"].expand_dims({"member": [0]}).transpose("member", "lat", "lon"))
        per_year.append(M.compute_crps(ens, b["obs"], season_end=season_end))
    if len(per_year) == 1:
        out = per_year[0]
        out["n_years"] = 1
        out["years"] = _years_meta(bundles, yrs)
        return out
    out = AGG.aggregate_crps(per_year)
    out["years"] = _years_meta(bundles, yrs)
    return out


@app.get("/api/metrics/fss")
def fss_route(
    model: str,
    year: int | None = None, years: str | None = None, init: str | None = None,
    thresholds: str | None = None,
    neighborhoods: str = "1,3,5,7,9",
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    thr = ([int(t) for t in thresholds.split(",")] if thresholds
           else _global_thresholds(bundles, n=5))
    if not thr:
        raise HTTPException(
            422,
            "no valid FSS thresholds across the selected years — "
            "the forecast / obs onset windows do not overlap. "
            "Narrow the region, pick different years, or pass thresholds= explicitly."
        )
    nbr = [int(n) for n in neighborhoods.split(",")]
    per_year = [M.compute_fss(b["fcst_det"], b["obs"], thresholds=thr, neighborhoods=nbr)
                for _, b in bundles]

    # base rate per threshold across years (fraction of obs cells with onset <= tau)
    import numpy as np
    base_rates = []
    for t in thr:
        rates = []
        for _, b in bundles:
            v = b["obs"].values
            f = np.isfinite(v)
            if not f.any(): continue
            rates.append(float((v[f] <= t).mean()))
        base_rates.append(float(np.mean(rates)) if rates else None)

    if len(per_year) == 1:
        out = per_year[0]
        out["n_years"] = 1
    else:
        out = AGG.aggregate_fss(per_year)
    out["years"] = _years_meta(bundles, yrs)
    out["base_rate"] = base_rates
    # Roberts & Lean 2008 conventions:
    #   FSS_random  = base_rate (FSS of a forecast that always predicts f0)
    #   FSS_useful  = 0.5 + 0.5 * base_rate (Roberts & Lean's "useful" line)
    out["fss_no_skill"] = [br if br is not None else None for br in base_rates]
    out["fss_useful"]   = [0.5 + 0.5 * br if br is not None else None
                           for br in base_rates]
    return out


@app.get("/api/metrics/displacement")
def displacement(
    model: str,
    year: int | None = None, years: str | None = None, init: str | None = None,
    thresholds: str | None = None,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    thr = ([int(t) for t in thresholds.split(",")] if thresholds
           else _global_thresholds(bundles))
    if not thr:
        raise HTTPException(
            422,
            "no valid displacement thresholds across the selected years — "
            "the forecast / obs onset windows do not overlap."
        )
    per_year = [M.compute_displacement(b["fcst_det"], b["obs"], thresholds=thr)
                for _, b in bundles]
    if len(per_year) == 1:
        out = per_year[0]
        out["n_years"] = 1
    else:
        out = AGG.aggregate_displacement(per_year)
    out["years"] = _years_meta(bundles, yrs)
    return out


@app.get("/api/metrics/progression")
def progression(
    model: str,
    year: int | None = None, years: str | None = None, init: str | None = None,
    step: int = 3,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    lo, hi = _global_progression_window(bundles)
    days = list(range(int(lo), int(hi) + 1, max(1, step)))
    per_year = [M.compute_progression(b["fcst_det"], b["ens"], b["obs"], days=days)
                for _, b in bundles]
    if len(per_year) == 1:
        out = per_year[0]
        out["n_years"] = 1
    else:
        out = AGG.aggregate_progression(per_year)
    out["years"] = _years_meta(bundles, yrs)
    return out


@app.get("/api/metrics/isochrones")
def isochrones(
    model: str, year: int, init: str | None = None,
    days: str | None = None,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    """Always single-year. The hero panel needs concrete contours; multi-year
    averaging would smear them. Use the iso_year sidebar control to pick which
    year's contours to render."""
    b = _fields_for(model, int(year), init, params, region)
    iso = ([int(d) for d in days.split(",")]
           if days else _global_thresholds([(year, b)]))
    out = M.compute_isochrones(b["fcst_det"], b["obs"], days=iso)
    out["year"] = int(year)
    return out


@app.get("/api/metrics/corp")
def corp(
    model: str,
    year: int | None = None, years: str | None = None, init: str | None = None,
    tau: int | None = None, season_end: int = 220,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    """Multi-year CORP pools all (forecast probability, obs binary) pairs
    across years before the PAV decomposition — that's the statistically
    proper way to combine, not averaging the per-year decompositions."""
    import numpy as np
    yrs = _resolve_years(years, year)
    bundles = _multi_year_bundles(model, yrs, init, params, region)
    if tau is None:
        thr = _global_thresholds(bundles)
        tau = thr[len(thr) // 2] if thr else 170
    p_parts, y_parts = [], []
    for _, b in bundles:
        p, y_ = M.corp_inputs(b["ens"], b["fcst_det"], b["obs"],
                              tau=tau, season_end=season_end)
        p_parts.append(p); y_parts.append(y_)
    p_pool = np.concatenate(p_parts)
    y_pool = np.concatenate(y_parts)
    out = M.compute_corp_pooled(p_pool, y_pool, tau=tau)
    out["n_years"] = len(bundles)
    out["years"] = _years_meta(bundles, yrs)
    return out


@app.get("/api/compare")
def compare(
    models: str,
    year: int | None = None, years: str | None = None, init: str | None = None,
    season_end: int = 220,
    params: OnsetParams = Depends(onset_deps), region: Region = Depends(region_deps),
):
    """Cross-model summary table. Per-row season scalars are median-across-years
    (with q25/q75) when years > 1, raw when single year."""
    yrs = _resolve_years(years, year)
    keys = [k.strip() for k in models.split(",") if k.strip()]
    rows = []
    for mk in keys:
        try:
            bundles = _multi_year_bundles(mk, yrs, init, params, region)
        except FileNotFoundError as e:
            rows.append({"model": mk, "error": str(e)})
            continue
        lo, hi = _global_progression_window(bundles)
        days = list(range(int(lo), int(hi) + 1, 3))
        prog_per = [M.compute_progression(b["fcst_det"], b["ens"], b["obs"], days=days)
                    for _, b in bundles]
        prog = AGG.aggregate_progression(prog_per) if len(prog_per) > 1 else prog_per[0]
        # CRPS mean across years
        crps_per = []
        for _, b in bundles:
            ens_like = (b["ens"] if b["ens"] is not None
                        else b["fcst_det"].expand_dims({"member": [0]}).transpose("member", "lat", "lon"))
            crps_per.append(M.compute_crps(ens_like, b["obs"], season_end=season_end))
        crps_means = [c["mean"] for c in crps_per if c.get("mean") is not None]
        # CORP pooled across years
        thr = _global_thresholds(bundles)
        tau = thr[len(thr) // 2] if thr else 170
        import numpy as np
        p_parts, y_parts = [], []
        for _, b in bundles:
            p, y_ = M.corp_inputs(b["ens"], b["fcst_det"], b["obs"],
                                  tau=tau, season_end=season_end)
            p_parts.append(p); y_parts.append(y_)
        corp_out = M.compute_corp_pooled(np.concatenate(p_parts),
                                         np.concatenate(y_parts), tau=tau)
        model_info = model_by_key(mk)
        b0 = bundles[0][1]
        rows.append({
            "model": mk,
            "label": model_info.label,
            "is_ensemble": b0["ens"] is not None,
            "n_members": (int(b0["ens"].sizes["member"])
                          if b0["ens"] is not None else 1),
            "init_idx": b0["init_idx"],
            "n_years": len(bundles),
            "years": [int(y) for y, _ in bundles],
            "progression": {
                "days": prog["days"],
                "ioe_km2": prog["ioe_km2"],
                "ioe_km2_q25": prog.get("ioe_km2_q25"),
                "ioe_km2_q75": prog.get("ioe_km2_q75"),
                "sps_km2": prog.get("sps_km2"),
                "sps_km2_q25": prog.get("sps_km2_q25"),
                "sps_km2_q75": prog.get("sps_km2_q75"),
                "extent_km2": prog.get("extent_km2"),
                "misplacement_km2": prog.get("misplacement_km2"),
                "season": prog.get("season", {}),
            },
            "crps": {
                "mean": float(np.mean(crps_means)) if crps_means else None,
                "median": float(np.median(crps_means)) if crps_means else None,
                "n_years": len(crps_means),
            },
            "corp": {
                "tau": corp_out["tau"],
                "bs": corp_out["mean_score"],
                "mcb": corp_out["mcb"],
                "dsc": corp_out["dsc"],
                "unc": corp_out["unc"],
            },
        })
    return {"years": yrs, "n_years": len(yrs), "params": params.to_json(), "rows": rows}


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
