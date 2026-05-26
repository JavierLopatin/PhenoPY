# Performance & scaling

PhenoPY maps the per-pixel reconstruction and metric extraction along the time
axis with `xarray.apply_ufunc`, which gives two ways to scale.

## Out-of-core / chunked processing

Pass `chunks` to `PhenoShape` (or chunk the input yourself) to process a raster
in spatial tiles instead of loading everything into memory at once:

```python
shape = da.pheno.PhenoShape(interpolType="linear", chunks={"x": 512, "y": 512})
lsp = shape.pheno.PhenoLSP()   # still lazy (Dask-backed)
lsp = lsp.compute()            # realises the result, tile by tile
```

The time axis is always kept whole (each pixel needs its full series); only the
spatial dimensions are tiled. This keeps peak memory bounded and lets you
process rasters larger than RAM.

## Parallelism and the GIL

The per-pixel kernels are currently pure-Python (via `numpy.vectorize`), so the
default *threaded* Dask scheduler does not give CPU speed-up — the GIL
serialises Python work, so its benefit is **memory**, not wall-clock. For CPU
parallelism today, use a *process*-based scheduler:

```python
import dask

with dask.config.set(scheduler="processes"):
    lsp = shape.pheno.PhenoLSP().compute()
```

A Numba-accelerated kernel (which releases the GIL and gives in-thread
speed-ups) is on the roadmap as the `[fast]` extra.

## Choosing chunk sizes

`phenopy.utils.computeChunkSize` suggests a chunking that targets ~100 MB tiles
while keeping the time axis whole:

```python
from phenopy.utils import computeChunkSize

chunks = computeChunkSize(da, sizeMB=100)
shape = da.pheno.PhenoShape(chunks=chunks)
```

See `benchmarks/bench.py` for a runnable comparison of the eager and Dask paths.
