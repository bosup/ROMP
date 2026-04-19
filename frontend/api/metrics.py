"""Thin wrappers that turn ROMP metric outputs into JSON-ready dicts."""
from __future__ import annotations

import numpy as np
import xarray as xr


def _as_list(a) -> list:
    arr = np.asarray(a, dtype=float)
    return [None if not np.isfinite(x) else float(x) for x in arr.ravel()]


def field_payload(da: xr.DataArray) -> dict:
    return {
        "lat": da["lat"].values.tolist(),
        "lon": da["lon"].values.tolist(),
        "values": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in np.asarray(da.values, dtype=float)
        ],
    }


def compute_crps(ens: xr.DataArray, obs: xr.DataArray, *, season_end: int) -> dict:
    from momp.metrics.crps import censored_crps_field
    crps = censored_crps_field(ens, obs, season_end=season_end)
    v = crps.values
    return {
        "field": field_payload(crps),
        "mean": float(np.nanmean(v)) if np.isfinite(v).any() else None,
        "max": float(np.nanmax(v)) if np.isfinite(v).any() else None,
        "n_finite": int(np.isfinite(v).sum()),
    }


def compute_fss(fcst: xr.DataArray, obs: xr.DataArray, *,
                thresholds, neighborhoods) -> dict:
    from momp.metrics.neighborhood import fss
    out = fss(fcst, obs, thresholds=thresholds, neighborhoods=neighborhoods)
    v = np.asarray(out.values, dtype=float)
    return {
        "thresholds": list(thresholds),
        "neighborhoods": list(neighborhoods),
        "fss": [[None if not np.isfinite(x) else float(x) for x in row] for row in v],
    }


def compute_displacement(fcst: xr.DataArray, obs: xr.DataArray, *, thresholds) -> dict:
    from momp.metrics.displacement import displacement_bias_sweep
    ds = displacement_bias_sweep(fcst, obs, thresholds=list(thresholds))
    return {
        "thresholds": list(thresholds),
        "delta_lat_deg": _as_list(ds["delta_lat_deg"].values),
        "delta_lon_deg": _as_list(ds["delta_lon_deg"].values),
        "great_circle_km": _as_list(ds["great_circle_km"].values),
        "area_bias_fraction": _as_list(ds["area_bias_fraction"].values),
    }


def compute_progression(fcst: xr.DataArray, ens: xr.DataArray | None,
                        obs: xr.DataArray, *, days) -> dict:
    from momp.metrics.progression import (
        integrated_onset_error, spatial_probability_score,
    )
    ioe = integrated_onset_error(fcst, obs, days=days)
    out = {
        "days": list(days),
        "ioe_km2": _as_list(ioe["ioe_km2"].values),
        "extent_km2": _as_list(ioe["extent_km2"].values),
        "misplacement_km2": _as_list(ioe["misplacement_km2"].values),
        "season": {
            "ioe_km2_day": float(ioe["ioe_season_km2_day"]),
            "extent_km2_day": float(ioe["extent_season_km2_day"]),
            "misplacement_km2_day": float(ioe["misplacement_season_km2_day"]),
        },
    }
    if ens is not None:
        sps = spatial_probability_score(ens, obs, days=days)
        out["sps_km2"] = _as_list(sps["sps_km2"].values)
        out["season"]["sps_km2_day"] = float(sps["sps_season_km2_day"])
    else:
        out["sps_km2"] = None
        out["season"]["sps_km2_day"] = None
    return out


def compute_isochrones(fcst: xr.DataArray, obs: xr.DataArray, *, days) -> dict:
    from momp.graphics.isochrone import (
        extract_isochrone, isochrone_distance_sweep,
    )
    entries = []
    for d in days:
        f_segs = extract_isochrone(fcst, float(d))
        o_segs = extract_isochrone(obs, float(d))
        entries.append({
            "day": int(d),
            "forecast": [s.tolist() for s in f_segs],
            "observed": [s.tolist() for s in o_segs],
        })
    sweep = isochrone_distance_sweep(fcst, obs, days=list(days))
    return {
        "isochrones": entries,
        "days": list(days),
        "hausdorff_km": _as_list(sweep["hausdorff_km"].values),
        "frechet_km": _as_list(sweep["frechet_km"].values),
        "n_segments_fcst": [int(v) for v in sweep["n_segments_fcst"].values],
        "n_segments_obs": [int(v) for v in sweep["n_segments_obs"].values],
    }


def compute_corp(ens: xr.DataArray | None, fcst: xr.DataArray,
                 obs: xr.DataArray, *, tau: int, season_end: int) -> dict:
    """Build a probability of 'onset <= tau' and decompose its Brier score."""
    from momp.graphics.corp_reliability import (
        corp_decompose_brier, _consolidate_curve,
    )
    if ens is not None:
        m = ens.values
        p = np.where(np.isnan(m) | (m > season_end), 0.0,
                     (m <= tau).astype(float)).mean(axis=0)
    else:
        f = fcst.values
        p = np.where(np.isnan(f) | (f > season_end), 0.0,
                     (f <= tau).astype(float))
    y = np.where(np.isfinite(obs.values) & (obs.values <= tau), 1.0, 0.0)
    decomp = corp_decompose_brier(p.ravel(), y.ravel())
    f_rep, c_rep = _consolidate_curve(decomp.forecast_prob, decomp.calibrated_y)
    return {
        "tau": int(tau),
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
