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

## Numba fast path (linear)

When the `[fast]` extra (numba) is installed, **linear** `PhenoShape` on an
in-memory raster automatically uses a GIL-releasing, Numba-parallel kernel — a
real multi-core speed-up over the pure-Python path, with identical results. No
code change is needed; it is used whenever the inputs are eligible (linear
method, in memory, no NaNs, no custom params).

```python
shape = da.pheno.PhenoShape(interpolType="linear")  # uses the Numba kernel if available
```

See `benchmarks/bench.py` for a numba-vs-pure-Python comparison.

## Parallelism and the GIL (other methods)

Non-linear reconstruction methods and the metric extraction still run a
pure-Python kernel (via `numpy.vectorize`). There, the GIL means the default
*threaded* Dask scheduler helps **memory**, not wall-clock; use a
*process*-based scheduler for CPU parallelism:

```python
import dask

with dask.config.set(scheduler="processes"):
    lsp = shape.pheno.PhenoLSP().compute()
```

Numba acceleration of the metric extraction and the other reconstruction
methods is planned.

## Choosing chunk sizes

`phenopy.utils.computeChunkSize` suggests a chunking that targets ~100 MB tiles
while keeping the time axis whole:

```python
from phenopy.utils import computeChunkSize

chunks = computeChunkSize(da, sizeMB=100)
shape = da.pheno.PhenoShape(chunks=chunks)
```

See `benchmarks/bench.py` for a runnable comparison of the eager and Dask paths.
