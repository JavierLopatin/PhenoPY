"""Smoke tests: the package imports, registers its accessor, and the core
PhenoShape -> PhenoLSP pipeline runs end to end on a tiny synthetic cube."""

import numpy as np
import pandas as pd
import xarray as xr

import phenopy  # noqa: F401  (importing registers the `pheno` accessor)

LSP_BANDS = [
    "sos",
    "pos",
    "eos",
    "vsos",
    "vpos",
    "veos",
    "los",
    "msp",
    "mau",
    "vmsp",
    "vmau",
    "ampl",
    "ios",
    "rog",
    "ros",
    "sw",
]


def _synthetic_cube(ny=3, nx=4, years=3):
    """A small (time, y, x) cube with a clean seasonal NDVI-like signal."""
    times = pd.date_range("2019-01-01", periods=23 * years, freq="16D")
    doy = times.dayofyear.values
    season = 0.4 + 0.3 * np.sin((doy / 365.0) * 2 * np.pi - np.pi / 2)
    data = np.broadcast_to(season[:, None, None], (len(times), ny, nx)).copy()
    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(ny), "x": np.arange(nx)},
    )
    return da.assign_coords(doy=("time", doy), year=("time", times.year.values))


def test_package_has_version():
    assert isinstance(phenopy.__version__, str)


def test_accessor_is_registered():
    da = xr.DataArray(np.zeros((5, 2, 2)), dims=("time", "y", "x"))
    assert hasattr(da, "pheno")


def test_phenoshape_then_phenolsp_pipeline():
    da = _synthetic_cube()

    shape = da.pheno.PhenoShape(nGS=20, rollWindow=3)
    assert "doy" in shape.dims

    lsp = shape.pheno.PhenoLSP().compute()

    # All 16 land-surface-phenology bands are present...
    assert set(lsp.data_vars) == set(LSP_BANDS)
    # ...spatial dims are preserved...
    assert lsp.sizes["y"] == da.sizes["y"]
    assert lsp.sizes["x"] == da.sizes["x"]
    # ...and the peak-of-season falls within a plausible DOY range.
    pos = float(lsp["pos"].isel(y=0, x=0))
    assert 1 <= pos <= 366
