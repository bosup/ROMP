"""CRPS and censored/mixed CRPS for onset-date ensemble forecasts.

Onset-date outcomes are mixed discrete-continuous: "no onset" is a positive-
probability atom, not missing data. Silently masking NaT grid cells produces
an improper score. This module implements a proper CRPS for the mixed
distribution by augmenting the sample space with a sentinel "no-onset" value
placed beyond the verification window.

Concretely, for each (grid cell, year) we treat the onset day as an element
of the extended real line ``{0, 1, ..., season_end, SENTINEL}`` where
SENTINEL = ``season_end + 1`` represents "no onset in the window." The
ensemble members and the observation are each mapped into this space and the
Hersbach (2000) ensemble CRPS is evaluated there.

This augmentation is equivalent to the censored-CRPS construction of Hemri
et al. (2014, GRL) for left-or-right censoring at a known threshold, and it
yields a proper scoring rule for the mixed onset distribution.

A diagnostic ``censored_crps_decomposition`` splits the score into a Brier
term for the onset-occurrence atom and a continuous CRPS term on the
onset-occurred members only; note that this decomposition is conceptually
useful but does NOT exactly sum to the censored CRPS (they are different
proper-score constructions).

References
----------
Hersbach 2000, *Wea. Forecasting* 15, 559-570.
Hemri et al. 2014, *Geophys. Res. Lett.* 41, 9197-9205.
Scheuerer & Hamill 2015, *Mon. Wea. Rev.* 143, 4578-4596 (CSGD).
Jordan, Krueger, Lerch 2019, *J. Stat. Softw.* 90 (scoringRules).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


def _to_doy(values, *, season_end: int) -> np.ndarray:
    """Convert datetime64 or NaT onset values to DOY floats.

    Values that are NaT (no onset) are mapped to NaN here; the caller is
    responsible for substituting the ``season_end + 1`` sentinel.
    """
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        # pandas handles NaT-aware DOY extraction with no Python loop.
        flat = pd.to_datetime(arr.ravel())
        doy = flat.dayofyear.to_numpy(dtype=float)
        doy[flat.isna()] = np.nan
        return doy.reshape(arr.shape)
    return arr.astype(float)


def _apply_sentinel(arr: np.ndarray, *, season_end: int) -> np.ndarray:
    sentinel = float(season_end) + 1.0
    out = arr.copy()
    out[~np.isfinite(out)] = sentinel
    out[out > season_end] = sentinel
    return out


def crps_ensemble(ensemble: np.ndarray, obs) -> np.ndarray:
    """Hersbach (2000) ensemble CRPS.

    Parameters
    ----------
    ensemble : array, shape (m, ...)
        Ensemble members along axis 0. Must be numeric (NaN not allowed).
    obs : scalar or array matching ``ensemble.shape[1:]``.

    Returns
    -------
    np.ndarray
        Shape ``ensemble.shape[1:]`` (or scalar if ``ensemble.ndim == 1``).

    Implementation uses the sorted-ensemble closed form

        CRPS = (1/m) sum_i |x_i - y|
             - (1/m^2) sum_{k=1}^m (2k - m - 1) x_(k).
    """
    ens = np.asarray(ensemble, dtype=float)
    if ens.ndim < 1:
        raise ValueError("ensemble must have at least 1 dimension")
    y = np.broadcast_to(np.asarray(obs, dtype=float), ens.shape[1:])
    if not np.all(np.isfinite(ens)):
        raise ValueError("ensemble must not contain NaN or inf; map no-onset to a sentinel first")
    if not np.all(np.isfinite(y)):
        raise ValueError("obs must not contain NaN or inf; map no-onset to a sentinel first")

    m = ens.shape[0]
    mae = np.mean(np.abs(ens - y), axis=0)

    sorted_ens = np.sort(ens, axis=0)
    k = np.arange(1, m + 1)
    shape = [m] + [1] * (sorted_ens.ndim - 1)
    coeff = (2 * k - m - 1).reshape(shape).astype(float)
    spread = (coeff * sorted_ens).sum(axis=0) / (m * m)

    return mae - spread


def censored_crps(
    ensemble,
    obs,
    *,
    season_end: int,
) -> np.ndarray:
    """Proper CRPS for the mixed onset distribution with a "no-onset" atom.

    Parameters
    ----------
    ensemble : array-like, shape (m, ...)
        Ensemble onset values. May contain NaN / NaT for members with no
        onset in the verification window.
    obs : scalar or array matching ``ensemble.shape[1:]``.
        Observed onset value. NaN / NaT encodes "no onset observed."
    season_end : int
        DOY upper bound of the onset verification window. No-onset entries
        are mapped to the sentinel ``season_end + 1``.
    """
    ens = _to_doy(np.asarray(ensemble), season_end=season_end)
    obs_arr = _to_doy(np.asarray(obs), season_end=season_end)
    ens_aug = _apply_sentinel(ens, season_end=season_end)
    obs_aug = _apply_sentinel(obs_arr, season_end=season_end)
    return crps_ensemble(ens_aug, obs_aug)


@dataclass(frozen=True)
class CensoredCRPSDecomposition:
    brier_atom: float          # Brier score on onset-occurrence indicator
    crps_continuous: float     # CRPS on DOY among members where onset occurred
    n_onset_members: int       # members with onset
    n_no_onset_members: int    # members with no onset
    observed_onset: bool       # True if observation had an onset


def censored_crps_decomposition(
    ensemble,
    obs,
    *,
    season_end: int,
) -> CensoredCRPSDecomposition:
    """Diagnostic decomposition of the mixed onset score.

    NOTE: ``brier_atom + crps_continuous`` does NOT equal ``censored_crps``
    in general; this helper exposes the two physical components separately
    for interpretation (calibration of the onset-occurrence probability
    vs. skill on the DOY distribution conditional on onset).
    """
    ens = _to_doy(np.asarray(ensemble), season_end=season_end)
    obs_arr = _to_doy(np.asarray(obs), season_end=season_end)
    if ens.ndim != 1 or obs_arr.ndim != 0:
        raise ValueError("decomposition is defined for a single (grid cell, year); use loops outside for fields")

    m = ens.size
    onset_members_mask = np.isfinite(ens) & (ens <= season_end)
    n_onset = int(onset_members_mask.sum())
    n_no_onset = m - n_onset

    pi_fcst = n_no_onset / m  # probability of "no onset"
    obs_no_onset = (not np.isfinite(obs_arr)) or (obs_arr > season_end)
    brier = float((pi_fcst - (1.0 if obs_no_onset else 0.0)) ** 2)

    if n_onset == 0 or obs_no_onset:
        crps_cont = 0.0
    else:
        onset_doys = ens[onset_members_mask]
        crps_cont = float(crps_ensemble(onset_doys, float(obs_arr)))

    return CensoredCRPSDecomposition(
        brier_atom=brier,
        crps_continuous=crps_cont,
        n_onset_members=n_onset,
        n_no_onset_members=n_no_onset,
        observed_onset=(not obs_no_onset),
    )


def censored_crps_skill_score(
    crps_forecast: np.ndarray,
    crps_reference: np.ndarray,
) -> np.ndarray:
    """Generic skill score: 1 - S_fcst / S_ref, NaN where S_ref == 0."""
    f = np.asarray(crps_forecast, dtype=float)
    r = np.asarray(crps_reference, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - f / r
    out = np.where(r > 0, out, np.nan)
    return out


def censored_crps_field(
    ensemble: xr.DataArray,
    observed: xr.DataArray,
    *,
    season_end: int,
    member_dim: str = "member",
) -> xr.DataArray:
    """Apply censored CRPS over a gridded onset field.

    Parameters
    ----------
    ensemble : xr.DataArray
        Onset values with ``member_dim`` plus any combination of lat / lon /
        year dims.
    observed : xr.DataArray
        Onset values sharing the non-member dims of ``ensemble``.
    season_end : int
        DOY upper bound of the onset window.

    Returns
    -------
    xr.DataArray
        CRPS per grid cell / year (same dims as ``observed``).
    """
    if member_dim not in ensemble.dims:
        raise ValueError(f"ensemble is missing member dim '{member_dim}'")
    ens = ensemble.transpose(member_dim, ...)
    values = censored_crps(ens.values, observed.values, season_end=season_end)
    out_dims = tuple(d for d in ens.dims if d != member_dim)
    return xr.DataArray(
        values,
        dims=out_dims,
        coords={d: ens.coords[d] for d in out_dims if d in ens.coords},
        name="censored_crps",
        attrs={
            "description": (
                "Censored CRPS for mixed onset distribution. "
                "No-onset atom mapped to sentinel = season_end + 1."
            ),
            "season_end": int(season_end),
        },
    )
