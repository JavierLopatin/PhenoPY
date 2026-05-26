"""Benchmark the Numba fast path for linear PhenoShape against pure-Python.

Run from the repo root in the dev env::

    python benchmarks/bench.py

PhenoShape's linear reconstruction uses a GIL-releasing, Numba-parallel kernel
when the ``[fast]`` extra (numba) is installed and the raster is in memory. This
script times that kernel against the pure-Python path on the same data and
checks the results are identical.
"""

import time

import numpy as np
import xarray as xr

import phenopy  # noqa: F401  (registers the `pheno` accessor)
from phenopy import _numba
from phenopy.io import load_sample


def tile(da, reps):
    """Tile the sample spatially to build a larger synthetic raster."""
    big = xr.concat([da] * reps, dim="y")
    big = xr.concat([big] * reps, dim="x")
    return big.assign_coords(y=np.arange(big.sizes["y"]), x=np.arange(big.sizes["x"]))


def time_phenoshape(da):
    t0 = time.perf_counter()
    out = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52)
    np.asarray(out.values)  # realise the computation
    return time.perf_counter() - t0, out


def main():
    da = tile(load_sample("SIF"), reps=5)
    npix = da.sizes["y"] * da.sizes["x"]
    print(f"raster: {dict(da.sizes)}  ({npix} pixels, {da.sizes['time']} time steps)")
    print(f"numba available: {_numba.NUMBA_AVAILABLE}")

    if _numba.NUMBA_AVAILABLE:
        # warm up the JIT so the timing reflects steady state, not compilation
        load_sample("SIF").pheno.PhenoShape(interpolType="linear", nGS=52)

    t_numba, out_numba = time_phenoshape(da)
    print(f"PhenoShape linear (numba)       : {t_numba:6.2f} s")

    orig = _numba.NUMBA_AVAILABLE
    _numba.NUMBA_AVAILABLE = False
    try:
        t_py, out_py = time_phenoshape(da)
    finally:
        _numba.NUMBA_AVAILABLE = orig
    print(f"PhenoShape linear (pure-Python) : {t_py:6.2f} s")

    if t_numba > 0:
        print(f"speedup (numba vs pure-Python)  : x{t_py / t_numba:.1f}")

    np.testing.assert_allclose(out_numba.values, out_py.values, rtol=1e-5, equal_nan=True)
    print("results identical (numba == pure-Python)")


if __name__ == "__main__":
    main()
