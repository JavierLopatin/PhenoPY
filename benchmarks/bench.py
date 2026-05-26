"""Benchmarks for PhenoPY's two scaling paths.

Run from the repo root in the dev env::

    python benchmarks/bench.py

1. **Reconstruction** (``PhenoShape``, linear): the Numba ``[fast]`` kernel vs the
   pure-Python path -- a single-core, in-memory acceleration of the hot loop.
2. **Extraction** (``PhenoLSP``): eager single-core vs chunked + the Dask
   ``processes`` scheduler -- i.e. how the *single* NumPy/scipy implementation
   scales across cores without a second code path (no Numba kernel needed).

Both checks assert the parallel result is identical to the serial one.
"""

import time

import dask
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


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    np.asarray(out.values)  # realise the computation
    return time.perf_counter() - t0, out


def bench_reconstruction(da):
    """PhenoShape(linear): Numba fast path vs pure-Python (single core, in memory)."""
    print("\n== Reconstruction: PhenoShape(linear) ==")
    params = dict(interpolType="linear", rollWindow=5, nGS=52)
    if _numba.NUMBA_AVAILABLE:
        load_sample("SIF").pheno.PhenoShape(**params)  # warm up the JIT (exclude compile time)
    t_numba, out_numba = timed(lambda: da.pheno.PhenoShape(**params))
    print(f"  numba [fast]    : {t_numba:6.2f} s")

    orig = _numba.NUMBA_AVAILABLE
    _numba.NUMBA_AVAILABLE = False
    try:
        t_py, out_py = timed(lambda: da.pheno.PhenoShape(**params))
    finally:
        _numba.NUMBA_AVAILABLE = orig
    print(f"  pure-Python     : {t_py:6.2f} s")
    if t_numba > 0:
        print(f"  speedup         : x{t_py / t_numba:.1f}")
    np.testing.assert_allclose(out_numba.values, out_py.values, rtol=1e-5, equal_nan=True)
    print("  identical       : OK")
    return out_numba  # reuse as the PhenoLSP input


def bench_extraction(shape, n_tiles=4):
    """PhenoLSP: eager (1 core) vs chunked + Dask 'processes' (multi-core)."""
    print("\n== Extraction: PhenoLSP (eager vs Dask processes) ==")
    t_eager, lsp_eager = timed(lambda: shape.pheno.PhenoLSP().compute())
    print(f"  eager (1 core)  : {t_eager:6.2f} s")

    cy = max(shape.sizes["y"] // n_tiles, 1)
    cx = max(shape.sizes["x"] // n_tiles, 1)
    shape_ch = shape.chunk({"y": cy, "x": cx})
    with dask.config.set(scheduler="processes"):
        t_par, lsp_par = timed(lambda: shape_ch.pheno.PhenoLSP().compute())
    print(f"  Dask processes  : {t_par:6.2f} s  ({n_tiles * n_tiles} tiles)")
    if t_par > 0:
        print(f"  speedup         : x{t_eager / t_par:.1f}")
    for band in lsp_eager.data_vars:
        np.testing.assert_allclose(
            lsp_eager[band].values, lsp_par[band].values, rtol=1e-5, equal_nan=True
        )
    print("  identical       : OK")


def main():
    da = tile(load_sample("SIF"), reps=8)
    npix = da.sizes["y"] * da.sizes["x"]
    print(f"raster: {dict(da.sizes)}  ({npix} pixels, {da.sizes['time']} time steps)")
    print(f"numba available: {_numba.NUMBA_AVAILABLE}")

    shape = bench_reconstruction(da)
    bench_extraction(shape)


if __name__ == "__main__":
    main()
