"""Multi-year aggregation for the frontend metric endpoints.

Each helper takes a list of per-year metric dicts (the same shapes that
metrics.py emits) and returns a single aggregated dict. Aggregation rule
is per-metric:

- CRPS field          : per-cell mean across years (nanmean)
- FSS matrix          : per-(threshold, neighborhood) mean
- Displacement sweep  : per-threshold median + IQR
- Progression curves  : per-day median + IQR for ioe / extent / misp / sps
- Isochrones          : NOT aggregated — one representative year only
- CORP                : pool raw (p, y) pairs across years, then decompose
                         once. Implemented in metrics.py via a separate
                         entry point (compute_corp_pooled).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _stack(lists, fill=np.nan) -> np.ndarray:
    """Stack a list of equal-length iterables into a 2-D array (years × N).
    None entries become ``fill``."""
    rows = []
    width = 0
    for r in lists:
        rows.append([fill if v is None else float(v) for v in (r or [])])
        width = max(width, len(rows[-1]))
    if not rows:
        return np.zeros((0, 0))
    out = np.full((len(rows), width), fill, dtype=float)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def _quantiles(arr: np.ndarray, q: float) -> list:
    if arr.size == 0:
        return []
    return [float(v) if np.isfinite(v) else None
            for v in np.nanquantile(arr, q, axis=0)]


def _median(arr: np.ndarray) -> list:
    if arr.size == 0:
        return []
    return [float(v) if np.isfinite(v) else None
            for v in np.nanmedian(arr, axis=0)]


def aggregate_crps(per_year: Sequence[dict]) -> dict:
    if not per_year:
        return {"field": None, "mean": None, "max": None, "n_finite": 0,
                "n_years": 0, "median": None, "q25": None, "q75": None}
    fields = [p.get("field") for p in per_year if p.get("field")]
    if not fields:
        return {"field": None, "mean": None, "max": None, "n_finite": 0,
                "n_years": len(per_year)}
    lat, lon = fields[0]["lat"], fields[0]["lon"]
    Ny, Nx = len(lat), len(lon)
    stack = np.full((len(fields), Ny, Nx), np.nan, dtype=float)
    for i, f in enumerate(fields):
        for r in range(min(Ny, len(f["values"]))):
            row = f["values"][r] or []
            for c in range(min(Nx, len(row))):
                v = row[c]
                stack[i, r, c] = float(v) if v is not None else np.nan
    with np.errstate(invalid="ignore"):
        mean_field = np.nanmean(stack, axis=0)
    out_values = [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in mean_field
    ]
    finite = mean_field[np.isfinite(mean_field)]
    return {
        "field": {"lat": lat, "lon": lon, "values": out_values},
        "mean": float(finite.mean()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "n_finite": int(finite.size),
        "n_years": len(fields),
        "median": float(np.nanmedian(finite)) if finite.size else None,
        "q25": float(np.nanquantile(finite, 0.25)) if finite.size else None,
        "q75": float(np.nanquantile(finite, 0.75)) if finite.size else None,
    }


def aggregate_fss(per_year: Sequence[dict]) -> dict:
    if not per_year:
        return {"thresholds": [], "neighborhoods": [], "fss": [], "n_years": 0}
    thr = per_year[0]["thresholds"]
    nbr = per_year[0]["neighborhoods"]
    Nt, Nn = len(thr), len(nbr)
    stack = np.full((len(per_year), Nt, Nn), np.nan, dtype=float)
    for i, p in enumerate(per_year):
        rows = p.get("fss") or []
        for r in range(min(Nt, len(rows))):
            row = rows[r] or []
            for c in range(min(Nn, len(row))):
                v = row[c]
                stack[i, r, c] = float(v) if v is not None else np.nan
    with np.errstate(invalid="ignore"):
        mean_fss = np.nanmean(stack, axis=0)
    out = [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in mean_fss
    ]
    return {"thresholds": thr, "neighborhoods": nbr, "fss": out,
            "n_years": len(per_year)}


def aggregate_displacement(per_year: Sequence[dict]) -> dict:
    if not per_year:
        return {"thresholds": [], "n_years": 0}
    thr = per_year[0]["thresholds"]
    keys = ("delta_lat_deg", "delta_lon_deg", "great_circle_km", "area_bias_fraction")
    out = {"thresholds": list(thr), "n_years": len(per_year)}
    for k in keys:
        stk = _stack([p.get(k, []) for p in per_year])
        out[k] = _median(stk)
        out[f"{k}_q25"] = _quantiles(stk, 0.25)
        out[f"{k}_q75"] = _quantiles(stk, 0.75)
    return out


def aggregate_progression(per_year: Sequence[dict]) -> dict:
    if not per_year:
        return {"days": [], "n_years": 0}
    days = per_year[0]["days"]
    ks = ("ioe_km2", "extent_km2", "misplacement_km2", "sps_km2")
    out = {"days": list(days), "n_years": len(per_year)}
    for k in ks:
        per_year_lists = [p.get(k) for p in per_year]
        # SPS may be None for det models; drop None lists
        per_year_lists = [lst for lst in per_year_lists if lst is not None]
        if not per_year_lists:
            out[k] = None
            out[f"{k}_q25"] = None
            out[f"{k}_q75"] = None
            continue
        stk = _stack(per_year_lists)
        out[k] = _median(stk)
        out[f"{k}_q25"] = _quantiles(stk, 0.25)
        out[f"{k}_q75"] = _quantiles(stk, 0.75)

    # Season-integrated scalars: median + IQR
    season = {"n_years": len(per_year)}
    for k in ("ioe_km2_day", "extent_km2_day", "misplacement_km2_day", "sps_km2_day"):
        vals = [p["season"].get(k) for p in per_year if p.get("season")]
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            season[k] = None
            season[f"{k}_q25"] = None
            season[f"{k}_q75"] = None
        else:
            arr = np.asarray(vals, dtype=float)
            season[k] = float(np.median(arr))
            season[f"{k}_q25"] = float(np.quantile(arr, 0.25))
            season[f"{k}_q75"] = float(np.quantile(arr, 0.75))
    out["season"] = season
    return out


def expand_years(years_arg: str | None, single_year: int | None) -> list[int]:
    """Parse a CSV `years=2019,2020,2021` or fall back to `year=2023`."""
    if years_arg:
        out = []
        for tok in years_arg.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok and not tok.startswith("-"):
                lo, hi = tok.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(tok))
        return sorted(set(out))
    if single_year is not None:
        return [int(single_year)]
    raise ValueError("must supply either ?year=YYYY or ?years=YYYY[,YYYY|YYYY-YYYY]")
