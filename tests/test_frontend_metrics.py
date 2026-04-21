"""Tests for frontend metric wrappers — specifically the interactions
added in the five-point audit pass (fair CRPS default, all-NaN cell
handling, shape consistency across single-year vs multi-year)."""
from __future__ import annotations

import numpy as np
import xarray as xr
import pytest

from frontend.api.metrics import (
    compute_crps, compute_progression, corp_inputs, compute_corp_pooled,
)
from frontend.api.aggregate import (
    aggregate_crps, aggregate_progression, expand_years, _stack,
)


def _field(values, lats, lons):
    return xr.DataArray(values, coords={"lat": lats, "lon": lons},
                        dims=("lat", "lon"))


def _ens(values, lats, lons, n_members=3):
    stacked = np.broadcast_to(values, (n_members,) + values.shape).copy()
    return xr.DataArray(
        stacked, dims=("member", "lat", "lon"),
        coords={"member": np.arange(n_members), "lat": lats, "lon": lons},
    )


# ---------------------------------------------------------------------------
# compute_crps: land-mask / all-NaN-cell regression (audit bug A-2)
# ---------------------------------------------------------------------------

def test_compute_crps_nulls_all_nan_cells_instead_of_counting_them_as_zero():
    """A cell where obs is NaN AND every ensemble member is NaN would map
    both sides to the same sentinel and yield CRPS = 0. Those synthetic
    zeros must NOT enter the reported mean or n_finite."""
    lats = np.arange(0.0, 4.0)
    lons = np.arange(0.0, 5.0)
    obs_vals = np.array([
        [150.0, 151.0, 152.0, 153.0, 154.0],
        [155.0, 156.0, np.nan, np.nan, np.nan],   # ocean / outside-mask
        [160.0, 161.0, np.nan, np.nan, np.nan],
        [165.0, 166.0, np.nan, np.nan, np.nan],
    ])
    ens_vals = np.broadcast_to(obs_vals + 5.0, (4,) + obs_vals.shape).copy()
    ens_vals[:, 1:, 2:] = np.nan                   # same missing mask on ens

    obs = _field(obs_vals, lats, lons)
    ens = xr.DataArray(
        ens_vals, dims=("member", "lat", "lon"),
        coords={"member": np.arange(4), "lat": lats, "lon": lons},
    )
    out = compute_crps(ens, obs, season_end=220)

    # Only the 11 "land" cells should be counted; the 9 NaN cells must
    # appear as null in the field payload, not 0.
    n_expected_land = int(np.isfinite(obs_vals).sum())
    assert out["n_finite"] == n_expected_land
    flat = [
        v for row in out["field"]["values"] for v in row
    ]
    # Every masked cell renders as None (JSON null), not 0.
    assert flat[1 * 5 + 2] is None
    assert flat[2 * 5 + 3] is None
    # Reported mean only averages real cells.
    real_cells = np.array(
        [v for v in flat if v is not None], dtype=float
    )
    assert out["mean"] == pytest.approx(float(real_cells.mean()), rel=1e-12)


def test_compute_crps_uses_fair_for_ensembles_and_raw_for_det():
    lats = np.array([0.0, 1.0]); lons = np.array([0.0, 1.0])
    obs = _field(np.full((2, 2), 150.0), lats, lons)
    ens = _ens(np.full((2, 2), 148.0), lats, lons, n_members=11)
    out = compute_crps(ens, obs, season_end=220)
    assert out["fair"] is True and out["n_members"] == 11

    det = ens.isel(member=[0])  # single-member
    out_det = compute_crps(det, obs, season_end=220)
    assert out_det["fair"] is False and out_det["n_members"] == 1


# ---------------------------------------------------------------------------
# compute_progression single-year vs aggregate_progression shape parity
# (audit bug B-2)
# ---------------------------------------------------------------------------

def test_single_and_multi_year_progression_share_season_schema():
    lats = np.arange(0.0, 5.0); lons = np.arange(0.0, 5.0)
    obs = _field(np.full((5, 5), 150.0), lats, lons)
    fcst = _field(np.full((5, 5), 155.0), lats, lons)
    single = compute_progression(fcst, None, obs, days=[140, 150, 160])
    multi = aggregate_progression([single, single, single])

    # Every *_q25 / *_q75 key present in multi must also be present in single.
    multi_season_keys = set(multi["season"].keys())
    single_season_keys = set(single["season"].keys())
    quantile_keys = {k for k in multi_season_keys if k.endswith("_q25") or k.endswith("_q75")}
    missing = quantile_keys - single_season_keys
    assert not missing, f"single-year season is missing {missing}"


# ---------------------------------------------------------------------------
# aggregate_crps: empty-fields fallback shape (audit bug B-1)
# ---------------------------------------------------------------------------

def test_aggregate_crps_empty_input_schema():
    out_empty = aggregate_crps([])
    for k in ("field", "mean", "max", "n_finite", "n_years",
              "median", "q25", "q75"):
        assert k in out_empty, f"missing key {k!r}"

    # Years requested but all yielded no field -> same schema.
    stub = {"mean": None, "max": None, "n_finite": 0}  # no "field"
    out_nofields = aggregate_crps([stub, stub])
    for k in ("field", "mean", "max", "n_finite", "n_years",
              "median", "q25", "q75"):
        assert k in out_nofields, f"missing key {k!r}"
    assert out_nofields["n_years"] == 2


# ---------------------------------------------------------------------------
# expand_years: error handling (audit bug B-3)
# ---------------------------------------------------------------------------

def test_expand_years_inverted_range_raises():
    with pytest.raises(ValueError, match="inverted"):
        expand_years("2020-2019", None)


def test_expand_years_bad_token_raises_clean():
    with pytest.raises(ValueError, match="malformed"):
        expand_years("abc", None)


def test_expand_years_bad_range_bound_raises_clean():
    with pytest.raises(ValueError, match="malformed"):
        expand_years("2019-abc", None)


def test_expand_years_accepts_mix_of_csv_and_range():
    yrs = expand_years("2019,2021-2023", None)
    assert yrs == [2019, 2021, 2022, 2023]


# ---------------------------------------------------------------------------
# _stack: ragged-input guard (audit bug B-6)
# ---------------------------------------------------------------------------

def test_stack_raises_on_ragged_input():
    with pytest.raises(ValueError, match="differing lengths"):
        _stack([[1.0, 2.0, 3.0], [4.0, 5.0]])


# ---------------------------------------------------------------------------
# corp_inputs / compute_corp_pooled: pooled CORP identity
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Land mask disambiguation (audit bug A-1)
# ---------------------------------------------------------------------------

def test_land_mask_exact_match_wins_over_substring():
    """'Niger' is a prefix of 'Nigeria' in the Natural Earth country list.
    The fix must pick exact Niger, not silently select Nigeria via substring."""
    pytest.importorskip("regionmask")
    import os
    from frontend.api import app as A

    da = xr.DataArray(
        np.zeros((3, 3)),
        coords={"lat": [12., 14., 16.], "lon": [7., 8., 9.]},  # Niger domain
        dims=("lat", "lon"),
    )

    os.environ["ROMP_LAND_MASK"] = "Niger"
    A._LAND_MASK_CACHE.clear()
    mask = A._land_mask_for(da)
    assert mask is not None, "exact 'Niger' should match"
    # The 3x3 above covers southern Niger; at least some cells should be land
    # AND all should be Niger-country (not Nigeria which is further south).
    assert mask.any(), "Niger mask should include at least one of those cells"


def test_land_mask_ambiguous_substring_rejected():
    """'Korea' is not an exact country name and matches both North and South
    Korea. The fix must reject ambiguous inputs rather than silently pick one."""
    pytest.importorskip("regionmask")
    import os
    from frontend.api import app as A

    da = xr.DataArray(
        np.zeros((3, 3)),
        coords={"lat": [36., 37., 38.], "lon": [126., 127., 128.]},
        dims=("lat", "lon"),
    )

    os.environ["ROMP_LAND_MASK"] = "Korea"
    A._LAND_MASK_CACHE.clear()
    mask = A._land_mask_for(da)
    # Ambiguous input -> ValueError is caught, warning printed, mask = None.
    assert mask is None, "ambiguous 'Korea' should be rejected, not silently picked"


# ---------------------------------------------------------------------------
# Moran's I + effective sample size (audit follow-up)
# ---------------------------------------------------------------------------

from frontend.api.metrics import moran_i_2d, effective_sample_size


def test_moran_i_random_field_near_zero():
    rng = np.random.default_rng(42)
    field = rng.standard_normal((20, 20))
    i = moran_i_2d(field)
    assert -0.2 < i < 0.2, f"random field Moran I should be near 0, got {i}"


def test_moran_i_gradient_field_strongly_positive():
    """A smooth north-south gradient is strongly spatially autocorrelated."""
    lat = np.arange(0.0, 20.0)
    field = np.broadcast_to(lat[:, None], (20, 20)).astype(float)
    i = moran_i_2d(field)
    assert i > 0.8, f"gradient field Moran I should be > 0.8, got {i}"


def test_moran_i_checkerboard_strongly_negative():
    ii, jj = np.indices((10, 10))
    field = ((ii + jj) % 2).astype(float)   # +1, 0 alternating
    i = moran_i_2d(field)
    assert i < -0.8, f"checkerboard Moran I should be < -0.8, got {i}"


def test_moran_i_nan_safety():
    field = np.full((6, 6), np.nan)
    field[0, 0] = 1.0
    field[0, 1] = 2.0
    field[1, 0] = 3.0
    field[1, 1] = 4.0
    # 4 finite cells, but contiguous — should return a real I.
    i = moran_i_2d(field)
    assert not np.isnan(i)
    # Too few finite cells -> NaN
    tiny = np.full((3, 3), np.nan)
    tiny[0, 0] = 1.0
    assert np.isnan(moran_i_2d(tiny))


def test_effective_sample_size_shrinks_with_positive_autocorrelation():
    assert effective_sample_size(100, 0.0) == 100.0
    assert effective_sample_size(100, -0.2) == 100.0   # negative I clamped to no-shrink
    eff = effective_sample_size(100, 0.5)
    assert eff == pytest.approx(100 * 0.5 / 1.5, rel=1e-9)
    assert effective_sample_size(100, 0.99) >= 1.0     # floored at 1


def test_pooled_corp_identity_holds():
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    rng = np.random.default_rng(7)
    obs_vals = rng.uniform(140, 170, size=(3, 3))
    ens_vals = obs_vals[None, :, :] + rng.normal(0, 5, size=(10, 3, 3))
    obs = _field(obs_vals, lats, lons)
    ens = xr.DataArray(
        ens_vals, dims=("member", "lat", "lon"),
        coords={"member": np.arange(10), "lat": lats, "lon": lons},
    )
    p, y = corp_inputs(ens, None, obs, tau=155, season_end=200)
    out = compute_corp_pooled(p, y, tau=155)
    assert abs(out["identity_residual"]) < 1e-9
