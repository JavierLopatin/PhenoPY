"""Unit tests for trend analysis (Theil-Sen slope + Mann-Kendall)."""

import numpy as np
import xarray as xr

from phenopy.trends import trend


def _series(values):
    arr = np.asarray(values, dtype=float).reshape(-1, 1, 1)
    return xr.DataArray(arr, dims=("time", "y", "x"))


def test_increasing_series_is_positive_and_significant():
    out = trend(_series(np.arange(12)))
    assert float(out["slope"].isel(y=0, x=0)) > 0
    assert float(out["p_value"].isel(y=0, x=0)) < 0.05


def test_decreasing_series_is_negative():
    out = trend(_series(np.arange(12)[::-1]))
    assert float(out["slope"].isel(y=0, x=0)) < 0


def test_flat_series_has_zero_slope():
    out = trend(_series(np.ones(10)))
    assert abs(float(out["slope"].isel(y=0, x=0))) < 1e-9


def test_too_short_series_is_nan():
    out = trend(_series([1.0, 2.0]))
    assert np.isnan(float(out["slope"].isel(y=0, x=0)))
