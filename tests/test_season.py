"""Tests for multi-season detection (NOS)."""

import numpy as np
import xarray as xr

from phenopy.season import n_seasons


def _shape(curve):
    xnew = np.linspace(1, 365, len(curve)).astype(int)
    return xr.DataArray(
        np.asarray(curve, dtype=float).reshape(-1, 1, 1),
        dims=("doy", "y", "x"),
        coords={"doy": xnew},
    )


def test_bimodal_two_seasons():
    x = np.linspace(1, 365, 80)
    curve = 0.2 + 0.5 * np.exp(-(((x - 100) / 25) ** 2)) + 0.5 * np.exp(-(((x - 260) / 25) ** 2))
    assert int(n_seasons(_shape(curve), prominence=0.1).isel(y=0, x=0)) == 2


def test_unimodal_one_season():
    x = np.linspace(1, 365, 80)
    curve = 0.2 + 0.6 * np.exp(-(((x - 180) / 40) ** 2))
    assert int(n_seasons(_shape(curve), prominence=0.1).isel(y=0, x=0)) == 1


def test_flat_zero_seasons():
    assert float(n_seasons(_shape(np.full(60, 0.5))).isel(y=0, x=0)) == 0.0


def test_accepts_time_dim_with_doy_coord():
    x = np.linspace(1, 365, 80)
    curve = 0.2 + 0.6 * np.exp(-(((x - 180) / 40) ** 2))
    da = xr.DataArray(
        curve.reshape(-1, 1, 1),
        dims=("time", "y", "x"),
        coords={"doy": ("time", x.astype(int))},
    )
    assert int(n_seasons(da, prominence=0.1).isel(y=0, x=0)) == 1
