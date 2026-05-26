"""The Numba fast path must match the pure-Python linear reconstruction."""

import numpy as np
import pandas as pd
import xarray as xr

import phenosensing  # noqa: F401  (registers the accessor)
from phenosensing import _numba


def _clean_cube(ny=4, nx=5, n=60):
    """A small (time, y, x) cube with no NaNs (so the Numba path is eligible)."""
    times = pd.date_range("2019-01-01", periods=n, freq="6D")
    doy = times.dayofyear.values
    season = 0.3 + 0.4 * np.sin((doy / 365.0) * 2 * np.pi - np.pi / 2)
    data = np.broadcast_to(season[:, None, None], (n, ny, nx)).copy()
    data += 0.01 * np.arange(ny * nx).reshape(1, ny, nx)  # per-pixel variation
    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(ny), "x": np.arange(nx)},
    )
    return da.assign_coords(doy=("time", doy), year=("time", times.year.values))


def test_numba_linear_matches_pure_python(monkeypatch):
    da = _clean_cube()

    numba_out = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=40)

    # Force the pure-Python path and compare.
    monkeypatch.setattr(_numba, "NUMBA_AVAILABLE", False)
    python_out = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=40)

    np.testing.assert_allclose(
        numba_out.values, python_out.values, rtol=1e-5, atol=1e-6, equal_nan=True
    )
