"""Known-answer tests for momp.metrics.neighborhood.fss.

FSS = 1 - MSE(F_f, F_o) / (mean F_f^2 + mean F_o^2).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from momp.metrics.neighborhood import fss, fss_multi_year, fss_single


RNG = np.random.default_rng(20260417)


def _random_onset_field(shape=(40, 50), fraction_onset=0.5):
    """Random DOY field with given onset fraction; NaN elsewhere."""
    doy = RNG.integers(120, 200, size=shape).astype(float)
    mask = RNG.uniform(0, 1, size=shape) < fraction_onset
    doy[~mask] = np.nan
    return doy


def test_perfect_forecast_equals_one():
    f = _random_onset_field()
    o = f.copy()
    val = fss_single(f, o, threshold=180, neighborhood=1)
    assert val == pytest.approx(1.0, abs=1e-12)
    val3 = fss_single(f, o, threshold=180, neighborhood=5)
    assert val3 == pytest.approx(1.0, abs=1e-12)


def test_forecast_says_yes_obs_says_no_is_zero():
    # Forecast has onsets everywhere by tau; obs has none.
    shape = (20, 30)
    f = np.full(shape, 150.0)
    o = np.full(shape, np.nan)
    val = fss_single(f, o, threshold=180, neighborhood=1)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_both_empty_returns_nan():
    shape = (20, 30)
    f = np.full(shape, np.nan)
    o = np.full(shape, np.nan)
    val = fss_single(f, o, threshold=180, neighborhood=1)
    assert np.isnan(val)


def test_monotone_in_neighborhood_for_shifted_field():
    # Forecast is obs shifted by 2 cells. For n >= 2*shift+1 the neighborhoods
    # overlap fully and FSS should climb toward 1.
    shape = (40, 40)
    obs = np.full(shape, np.nan)
    obs[10:30, 10:30] = 150.0
    shift = 2
    fcst = np.full(shape, np.nan)
    fcst[10 + shift : 30 + shift, 10 + shift : 30 + shift] = 150.0

    # With constant zero-padding, FSS reaches exactly 1 only when the window
    # covers the entire domain from every cell, i.e. n >= 2*L - 1 where L is
    # the domain side. For L=40 that means n=79.
    neighborhoods = (1, 3, 5, 7, 9, 15, 25, 39, 79)
    scores = [
        fss_single(fcst, obs, threshold=180, neighborhood=n) for n in neighborhoods
    ]
    # Roberts-Lean monotone-in-neighborhood property (up to floating tolerance).
    for a, b in zip(scores, scores[1:]):
        assert b >= a - 1e-10
    # Raw displacement should cost skill at small n relative to large n.
    assert scores[0] < scores[-1] - 1e-6
    # Asymptotic: a neighborhood as large as the domain drives FSS to 1.
    assert scores[-1] == pytest.approx(1.0, abs=1e-6)


def test_fss_is_symmetric_in_forecast_and_obs():
    f = _random_onset_field()
    o = _random_onset_field()
    a = fss_single(f, o, threshold=170, neighborhood=3)
    b = fss_single(o, f, threshold=170, neighborhood=3)
    assert a == pytest.approx(b, abs=1e-12)


def test_sweep_shape_and_coords():
    f = _random_onset_field()
    o = _random_onset_field()
    thresholds = [140, 160, 180]
    neighborhoods = [1, 3, 5]
    da = fss(f, o, thresholds=thresholds, neighborhoods=neighborhoods)
    assert da.dims == ("threshold", "neighborhood")
    assert da.shape == (3, 3)
    assert list(da.threshold.values) == thresholds
    assert list(da.neighborhood.values) == neighborhoods
    # FSS lies in [-inf, 1]; for well-posed cases in [0, 1].
    assert (da.values <= 1.0 + 1e-12).all()


def test_even_neighborhood_raises():
    f = _random_onset_field((10, 10))
    o = _random_onset_field((10, 10))
    with pytest.raises(ValueError):
        fss_single(f, o, threshold=180, neighborhood=4)


def test_shape_mismatch_raises():
    f = _random_onset_field((10, 10))
    o = _random_onset_field((10, 12))
    with pytest.raises(ValueError):
        fss_single(f, o, threshold=180, neighborhood=1)


def test_nan_treated_as_no_onset():
    shape = (8, 8)
    f = np.full(shape, np.nan)
    o = np.full(shape, 150.0)
    # Forecast says no onset anywhere; obs says onset everywhere by tau=180.
    # Under NaN-as-False, f's fraction is 0 and o's fraction is 1.
    val = fss_single(f, o, threshold=180, neighborhood=1)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_xarray_input_accepted():
    shape = (20, 20)
    arr_f = _random_onset_field(shape)
    arr_o = _random_onset_field(shape)
    f = xr.DataArray(arr_f, dims=("lat", "lon"))
    o = xr.DataArray(arr_o, dims=("lat", "lon"))
    da = fss(f, o, thresholds=[160, 180], neighborhoods=[1, 3])
    assert da.shape == (2, 2)


def test_multi_year_basic():
    fby = {y: _random_onset_field((20, 20)) for y in (2001, 2002, 2003)}
    oby = {y: _random_onset_field((20, 20)) for y in (2001, 2002, 2003)}
    da = fss_multi_year(fby, oby, thresholds=[160, 180], neighborhoods=[1, 3])
    assert da.dims == ("year", "threshold", "neighborhood")
    assert list(da.year.values) == [2001, 2002, 2003]
