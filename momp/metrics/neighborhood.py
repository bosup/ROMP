"""Neighborhood-based spatial verification for onset fields.

Implements the Fractions Skill Score (FSS) of

    Roberts & Lean 2008, "Scale-selective verification of rainfall
    accumulations from high-resolution forecasts of convective events,"
    Mon. Wea. Rev. 136, 78-97. doi:10.1175/2007MWR2123.1

Given two 2-D onset-date fields (forecast and observation), FSS is computed
by: (1) thresholding each field with ``fcst_doy <= threshold`` to produce a
binary "has onset by DOY" mask, with NaN/NaT treated as False; (2) computing
the fraction of "has onset" cells in every n-by-n neighborhood; (3) comparing
the forecast and observed fraction fields via

    FSS(tau, n) = 1 - MSE(F_f, F_o) / (mean(F_f**2) + mean(F_o**2))

where the MSE and means are taken over the whole domain. FSS = 1 is perfect,
FSS = 0 is no skill. The score is non-decreasing in the neighborhood size n.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter


def _as_binary_mask(field, threshold: float) -> np.ndarray:
    """Threshold a 2-D field to a {0, 1} float mask, NaN -> 0."""
    a = np.asarray(field, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D field, got shape {a.shape}")
    mask = (a <= threshold) & np.isfinite(a)
    return mask.astype(float)


def _fraction_field(binary: np.ndarray, neighborhood: int) -> np.ndarray:
    """Box-average binary over an ``neighborhood``-by-``neighborhood`` window.

    ``neighborhood`` must be a positive odd integer. Cells outside the domain
    are treated as 0 (Roberts-Lean convention).
    """
    if neighborhood <= 0 or neighborhood % 2 == 0:
        raise ValueError(f"neighborhood must be a positive odd integer, got {neighborhood}")
    if neighborhood == 1:
        return binary
    return uniform_filter(binary, size=neighborhood, mode="constant", cval=0.0)


def fss_single(
    forecast: np.ndarray,
    observed: np.ndarray,
    *,
    threshold: float,
    neighborhood: int,
) -> float:
    """Fractions Skill Score for one (threshold, neighborhood) pair.

    Returns NaN when the reference denominator is zero, i.e. both fields are
    identically False — the score is undefined there.
    """
    f_bin = _as_binary_mask(forecast, threshold)
    o_bin = _as_binary_mask(observed, threshold)
    if f_bin.shape != o_bin.shape:
        raise ValueError(
            f"forecast and observed shape mismatch: {f_bin.shape} vs {o_bin.shape}"
        )
    F_f = _fraction_field(f_bin, neighborhood)
    F_o = _fraction_field(o_bin, neighborhood)
    mse = float(np.mean((F_f - F_o) ** 2))
    denom = float(np.mean(F_f**2) + np.mean(F_o**2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - mse / denom


def fss(
    forecast,
    observed,
    *,
    thresholds: Sequence[float],
    neighborhoods: Sequence[int],
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> xr.DataArray:
    """Fractions Skill Score over a sweep of thresholds and neighborhood sizes.

    Parameters
    ----------
    forecast, observed : 2-D array-like or xr.DataArray
        Onset-date fields. NaN / NaT entries are treated as "no onset"
        (i.e. the threshold condition is False).
    thresholds : sequence of numeric
        DOY thresholds tau; the binary mask is ``field <= tau``.
    neighborhoods : sequence of positive odd int
        Square window sizes in grid cells.
    lat_coord, lon_coord : str
        Coordinate names used on xarray inputs; ignored for numpy arrays.

    Returns
    -------
    xr.DataArray
        Array with dims ``("threshold", "neighborhood")``.
    """
    f_arr = forecast.values if isinstance(forecast, xr.DataArray) else np.asarray(forecast)
    o_arr = observed.values if isinstance(observed, xr.DataArray) else np.asarray(observed)

    thr = np.asarray(thresholds, dtype=float)
    nbr = np.asarray(neighborhoods, dtype=int)

    out = np.full((thr.size, nbr.size), np.nan, dtype=float)
    for i, t in enumerate(thr):
        f_bin = _as_binary_mask(f_arr, float(t))
        o_bin = _as_binary_mask(o_arr, float(t))
        if f_bin.shape != o_bin.shape:
            raise ValueError(
                f"forecast and observed shape mismatch: {f_bin.shape} vs {o_bin.shape}"
            )
        for j, n in enumerate(nbr):
            F_f = _fraction_field(f_bin, int(n))
            F_o = _fraction_field(o_bin, int(n))
            mse = float(np.mean((F_f - F_o) ** 2))
            denom = float(np.mean(F_f**2) + np.mean(F_o**2))
            out[i, j] = 1.0 - mse / denom if denom > 0 else float("nan")

    return xr.DataArray(
        out,
        dims=("threshold", "neighborhood"),
        coords={"threshold": thr, "neighborhood": nbr},
        name="fss",
        attrs={
            "description": (
                "Fractions Skill Score. Roberts & Lean 2008. "
                "1 = perfect, 0 = no skill. Non-decreasing in neighborhood."
            )
        },
    )


def fss_multi_year(
    forecast_by_year: dict,
    observed_by_year: dict,
    *,
    thresholds: Sequence[float],
    neighborhoods: Sequence[int],
) -> xr.DataArray:
    """Per-year FSS across a shared set of thresholds and neighborhoods.

    ``forecast_by_year`` and ``observed_by_year`` are dicts keyed by year
    mapping to 2-D onset-date fields.

    Returns an xr.DataArray with dims ``("year", "threshold", "neighborhood")``.
    Missing years (keys in forecast but not observed, or vice versa) are
    skipped with a warning; present-in-both years are included.
    """
    shared_years = sorted(set(forecast_by_year) & set(observed_by_year))
    if not shared_years:
        raise ValueError("no overlapping years between forecast and observed dicts")

    thr = np.asarray(thresholds, dtype=float)
    nbr = np.asarray(neighborhoods, dtype=int)

    stack = np.full((len(shared_years), thr.size, nbr.size), np.nan, dtype=float)
    for k, yr in enumerate(shared_years):
        per_year = fss(
            forecast_by_year[yr],
            observed_by_year[yr],
            thresholds=thr,
            neighborhoods=nbr,
        )
        stack[k] = per_year.values

    return xr.DataArray(
        stack,
        dims=("year", "threshold", "neighborhood"),
        coords={"year": shared_years, "threshold": thr, "neighborhood": nbr},
        name="fss_multi_year",
        attrs={"description": "Per-year Fractions Skill Score (Roberts & Lean 2008)."},
    )
