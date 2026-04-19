"""Known-answer tests for momp.graphics.isochrone."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

from momp.graphics.isochrone import (
    extract_isochrone,
    isochrone_distance,
    isochrone_distance_sweep,
    isochrone_overlay,
)


def _ramp_field(lats: np.ndarray, lons: np.ndarray, slope=1.0, intercept=120.0):
    """Onset field with DOY = slope * lat + intercept (isochrones are horizontal lines)."""
    doy = slope * lats[:, None] + intercept + 0 * lons[None, :]
    return xr.DataArray(doy, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))


def test_identical_fields_zero_distance():
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 30, 61)
    f = _ramp_field(lats, lons)
    r = isochrone_distance(f, f, day=140.0)
    assert r["hausdorff_deg"] == pytest.approx(0.0, abs=1e-8)
    assert r["frechet_deg"] == pytest.approx(0.0, abs=1e-8) or np.isnan(r["frechet_deg"])
    assert r["n_segments_fcst"] >= 1
    assert r["n_segments_obs"] >= 1


def test_pure_northward_offset_recovered():
    # Observed: DOY = lat + 120. Forecast: DOY = (lat - 5) + 120 = lat + 115.
    # The DOY = 140 isochrone for obs is lat = 20; for fcst is lat = 25.
    lats = np.linspace(0, 30, 301)
    lons = np.linspace(0, 20, 201)
    obs = _ramp_field(lats, lons, slope=1.0, intercept=120.0)
    fcst = _ramp_field(lats, lons, slope=1.0, intercept=115.0)
    r = isochrone_distance(fcst, obs, day=140.0)
    assert r["hausdorff_deg"] == pytest.approx(5.0, abs=0.1)


def test_isochrone_missing_returns_nan():
    # DOY 300 lies outside the field range.
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons)
    fcst = _ramp_field(lats, lons)
    r = isochrone_distance(fcst, obs, day=300.0)
    assert np.isnan(r["hausdorff_deg"])
    assert r["n_segments_fcst"] == 0


def test_km_conversion_consistent_with_degrees():
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons, slope=1.0, intercept=120.0)
    fcst = _ramp_field(lats, lons, slope=1.0, intercept=115.0)
    r = isochrone_distance(fcst, obs, day=140.0)
    # KM_PER_DEG = 111.1949... so km should be ~5 deg * ~111 = ~556 km.
    assert r["hausdorff_km"] == pytest.approx(r["hausdorff_deg"] * 111.19492664455873, rel=1e-6)


def test_sweep_shape():
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons)
    fcst = _ramp_field(lats, lons, intercept=115.0)
    ds = isochrone_distance_sweep(fcst, obs, days=[130, 140, 150, 160])
    assert ds.sizes["day"] == 4
    assert "hausdorff_km" in ds


def test_extract_returns_vertex_arrays():
    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    field = _ramp_field(lats, lons)
    segs = extract_isochrone(field, day=140.0)
    assert len(segs) >= 1
    for s in segs:
        assert s.ndim == 2 and s.shape[1] == 2


def test_overlay_returns_figure_without_error():
    import matplotlib.pyplot as plt

    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons)
    fcst = _ramp_field(lats, lons, intercept=115.0)
    fig = isochrone_overlay(fcst, obs, days=[130, 150, 170], show=False)
    assert fig is not None
    assert fig.axes
    plt.close(fig)


def test_overlay_can_save_to_tmp(tmp_path):
    import matplotlib.pyplot as plt

    lats = np.linspace(0, 30, 61)
    lons = np.linspace(0, 20, 41)
    obs = _ramp_field(lats, lons)
    fcst = _ramp_field(lats, lons, intercept=115.0)
    out = tmp_path / "iso.png"
    fig = isochrone_overlay(fcst, obs, days=[130, 150], save_path=str(out), show=False)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
