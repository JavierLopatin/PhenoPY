"""Quick benchmark of PhenoShape -> PhenoLSP: eager vs Dask (threads / processes).

Run from the repo root in the dev env::

    python benchmarks/bench.py

The point of this benchmark is to show (a) that the chunked Dask path produces
identical results to the eager path, and (b) the real wall-clock trade-offs.
Because the per-pixel kernels are pure-Python (numpy.vectorize), the *threaded*
scheduler mainly helps memory, not speed (the GIL); a *process* scheduler gives
CPU parallelism at the cost of serialisation overhead. A Numba ``[fast]`` kernel
(GIL-releasing) is on the roadmap for in-thread speedups.
"""

import time

import dask
import numpy as np
import xarray as xr

import phenopy  # noqa: F401  (registers the `pheno` accessor)
from phenopy.io import load_sample


def tile(da, reps):
    """Tile the sample spatially to build a larger synthetic raster."""
    big = xr.concat([da] * reps, dim="y")
    big = xr.concat([big] * reps, dim="x")
    return big.assign_coords(y=np.arange(big.sizes["y"]), x=np.arange(big.sizes["x"]))


def run(da, chunks=None):
    t0 = time.perf_counter()
    shape = da.pheno.PhenoShape(rollWindow=5, nGS=52, chunks=chunks)
    lsp = shape.pheno.PhenoLSP().compute()
    return time.perf_counter() - t0, lsp


def main():
    da = tile(load_sample("SIF"), reps=4)
    npix = da.sizes["y"] * da.sizes["x"]
    print(f"raster: {dict(da.sizes)}  ({npix} pixels, {da.sizes['time']} time steps)")

    t_eager, lsp_eager = run(da)
    print(f"eager (numpy)              : {t_eager:6.2f} s")

    t_threads, lsp_threads = run(da, chunks={"x": 20, "y": 20})
    print(f"dask threads (20x20 chunks): {t_threads:6.2f} s")

    with dask.config.set(scheduler="processes"):
        t_proc, _ = run(da, chunks={"x": 20, "y": 20})
    print(f"dask processes (20x20)     : {t_proc:6.2f} s   speedup x{t_eager / t_proc:.2f}")

    np.testing.assert_allclose(lsp_eager["sos"].values, lsp_threads["sos"].values, equal_nan=True)
    print("results identical (eager == dask)")


if __name__ == "__main__":
    main()
