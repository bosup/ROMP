"""Onset-progression verification: Integrated Onset Error (IOE) and the
ensemble Spatial Probability Score (SPS).

This module treats rainy-season onset as an advancing front rather than a
static DOY map. For any calendar day ``d`` within the onset window the
binary indicator

    has_onset_by_d(x) = (onset_DOY(x) <= d)

defines a contour separating "monsoon has arrived at x" from "has not." The
symmetric difference between the forecast and observed indicators, integrated
over the domain area, is the **Integrated Onset Error** at day ``d`` — the
onset-field analogue of Goessling et al.'s Integrated Ice-Edge Error for sea
ice. Season-integration over ``d`` yields a single headline score.

The ensemble generalisation, **Spatial Probability Score**, replaces the
deterministic forecast indicator with the ensemble probability
``P_fcst(onset <= d)`` and sums a Brier score per cell. It reduces to IOE for
a deterministic forecast.

References
----------
Goessling et al. 2016, *Geophys. Res. Lett.* 43, 1642-1650 (IIEE).
Goessling & Jung 2018, *Q. J. R. Meteorol. Soc.* 144, 197-210 (SPS).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid

from momp.metrics.displacement import (
    EARTH_RADIUS_KM_DEFAULT,
    _extract_coords,
    _infer_spacing,
)


def _cell_area_km2(lat: np.ndarray, lon: np.ndarray, R_km: float) -> np.ndarray:
    """Per-cell area in km^2 on a regular lat/lon grid, shape (lat, lon)."""
    dlat = _infer_spacing(lat)
    dlon = _infer_spacing(lon)
    cos_lat = np.cos(np.deg2rad(lat))
    per_lat = (R_km**2) * np.deg2rad(dlat) * np.deg2rad(dlon) * cos_lat
    return np.broadcast_to(per_lat[:, None], (lat.size, lon.size)).copy()


def _binary_by_day(field_vals: np.ndarray, day: float) -> np.ndarray:
    return ((field_vals <= day) & np.isfinite(field_vals)).astype(float)


def _check_grids_match(forecast: xr.DataArray, observed: xr.DataArray,
                      lat_coord: str, lon_coord: str):
    lat_f, lon_f = _extract_coords(forecast, lat_coord, lon_coord)
    lat_o, lon_o = _extract_coords(observed, lat_coord, lon_coord)
    if not (np.array_equal(lat_f, lat_o) and np.array_equal(lon_f, lon_o)):
        raise ValueError("forecast and observed must share the same lat/lon grid")
    return lat_f, lon_f


def integrated_onset_error(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    *,
    days: Sequence[int],
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    earth_radius_km: float = EARTH_RADIUS_KM_DEFAULT,
) -> xr.Dataset:
    """Per-day IOE, extent-error, misplacement-error, and season-integrated totals.

    For each day d in ``days``:
        area_f(d) = area where fcst_DOY <= d
        area_o(d) = area where obs_DOY <= d
        IOE(d) = area of symmetric difference of the two masks
        extent(d) = |area_f(d) - area_o(d)|
        misplacement(d) = IOE(d) - extent(d)

    Season-integrated totals are trapezoid-integrated over ``days`` with a
    unit step of 1 day between consecutive integer days (values in km^2 * day).
    Non-uniform ``days`` are integrated using ``np.trapz`` with the given
    abscissae.

    Parameters
    ----------
    forecast, observed : xr.DataArray
        Onset-date fields with dims (lat, lon). NaT / NaN encode no onset.
    days : sequence of ints (DOY)

    Returns
    -------
    xr.Dataset
        Variables ``ioe_km2, extent_km2, misplacement_km2`` indexed by
        ``day``, plus scalar variables ``ioe_season_km2_day,
        extent_season_km2_day, misplacement_season_km2_day``.
    """
    lat, lon = _check_grids_match(forecast, observed, lat_coord, lon_coord)
    A = _cell_area_km2(lat, lon, earth_radius_km)

    fvals = np.asarray(forecast.values, dtype=float)
    ovals = np.asarray(observed.values, dtype=float)

    days_arr = np.asarray(list(days), dtype=float)
    if days_arr.ndim != 1 or days_arr.size == 0:
        raise ValueError("days must be a non-empty 1-D sequence")

    ioe = np.empty(days_arr.size, dtype=float)
    extent = np.empty(days_arr.size, dtype=float)
    misp = np.empty(days_arr.size, dtype=float)
    for k, d in enumerate(days_arr):
        f_bin = _binary_by_day(fvals, d)
        o_bin = _binary_by_day(ovals, d)
        sym_diff = np.abs(f_bin - o_bin)
        ioe[k] = float(np.sum(sym_diff * A))
        area_f = float(np.sum(f_bin * A))
        area_o = float(np.sum(o_bin * A))
        extent[k] = abs(area_f - area_o)
        misp[k] = ioe[k] - extent[k]

    ioe_season = float(trapezoid(ioe, days_arr))
    extent_season = float(trapezoid(extent, days_arr))
    misp_season = float(trapezoid(misp, days_arr))

    return xr.Dataset(
        {
            "ioe_km2": ("day", ioe),
            "extent_km2": ("day", extent),
            "misplacement_km2": ("day", misp),
            "ioe_season_km2_day": ((), ioe_season),
            "extent_season_km2_day": ((), extent_season),
            "misplacement_season_km2_day": ((), misp_season),
        },
        coords={"day": days_arr.astype(int)},
        attrs={
            "description": (
                "Integrated Onset Error (Goessling 2016 IIEE analog). "
                "IOE = area of symmetric difference between forecast and observed "
                "'onset-by-d' masks. Season-integrated via trapezoid rule over days."
            )
        },
    )


def spatial_probability_score(
    ensemble: xr.DataArray,
    observed: xr.DataArray,
    *,
    days: Sequence[int],
    member_dim: str = "member",
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    earth_radius_km: float = EARTH_RADIUS_KM_DEFAULT,
) -> xr.Dataset:
    """Per-day Spatial Probability Score, plus season integral.

    For each day d and each grid cell x:
        P_fcst(x, d) = fraction of ensemble members with onset_DOY(x) <= d
        O(x, d)      = 1 if obs_DOY(x) <= d else 0
        SPS(d) = sum_x (P_fcst(x, d) - O(x, d))^2 * A(x)

    SPS reduces to IOE when the ensemble is a single deterministic member.

    Parameters
    ----------
    ensemble : xr.DataArray
        Dims ``(member, lat, lon)`` of onset DOY. NaT encodes no onset.
    observed : xr.DataArray
        Dims ``(lat, lon)``.

    Returns
    -------
    xr.Dataset
        Variables ``sps_km2`` indexed by ``day`` and scalar
        ``sps_season_km2_day``.
    """
    if member_dim not in ensemble.dims:
        raise ValueError(f"ensemble is missing member dim '{member_dim}'")
    ens = ensemble.transpose(member_dim, ...)
    if ens.dims[1:] != observed.dims:
        raise ValueError(
            f"ensemble non-member dims {ens.dims[1:]} must match observed dims {observed.dims}"
        )
    lat, lon = _check_grids_match(ens.isel({member_dim: 0}), observed, lat_coord, lon_coord)
    A = _cell_area_km2(lat, lon, earth_radius_km)

    ens_vals = np.asarray(ens.values, dtype=float)
    obs_vals = np.asarray(observed.values, dtype=float)

    days_arr = np.asarray(list(days), dtype=float)
    if days_arr.ndim != 1 or days_arr.size == 0:
        raise ValueError("days must be a non-empty 1-D sequence")

    sps = np.empty(days_arr.size, dtype=float)
    for k, d in enumerate(days_arr):
        member_masks = _binary_by_day(ens_vals, d)  # (m, lat, lon)
        p_fcst = member_masks.mean(axis=0)          # (lat, lon)
        o_bin = _binary_by_day(obs_vals, d)         # (lat, lon)
        sps[k] = float(np.sum((p_fcst - o_bin) ** 2 * A))

    sps_season = float(trapezoid(sps, days_arr))

    return xr.Dataset(
        {
            "sps_km2": ("day", sps),
            "sps_season_km2_day": ((), sps_season),
        },
        coords={"day": days_arr.astype(int)},
        attrs={
            "description": (
                "Spatial Probability Score (Goessling & Jung 2018). "
                "Per-cell Brier score of P(onset <= d) against observed indicator, "
                "area-weighted and summed. Reduces to IOE for a deterministic ensemble."
            )
        },
    )
