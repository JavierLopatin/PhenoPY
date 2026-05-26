# Performance & scaling

PhenoPY keeps **one readable reference implementation** per axis (NumPy/SciPy,
mapped per pixel with `xarray.apply_ufunc`) and scales it two ways — it never
ships a second copy of an algorithm. The two knobs:

## TL;DR — which knob when

| Situation | Use |
|---|---|
| In-memory raster, **linear** reconstruction | Nothing — the Numba `[fast]` kernel is used automatically (~12× over pure Python) |
| Raster **larger than RAM** | `chunks=` — process in spatial tiles, bounded memory |
| **Large** raster, want multi-core extraction / non-linear methods | Chunk + the Dask `processes` scheduler |
| Small / medium in-memory raster | Eager is usually fastest — skip the scheduler overhead |

## Out-of-core / chunked processing

Pass `chunks` to `PhenoShape` (or chunk the input yourself) to process a raster
in spatial tiles instead of loading it all into memory:

```python
shape = da.pheno.PhenoShape(interpolType="linear", chunks={"x": 512, "y": 512})
lsp = shape.pheno.PhenoLSP()   # still lazy (Dask-backed)
lsp = lsp.compute()            # realises the result, tile by tile
```

The time axis is always kept whole (each pixel needs its full series); only the
spatial dims are tiled. This keeps peak memory bounded and lets you process
rasters larger than RAM.

## Numba fast path (linear reconstruction)

With the `[fast]` extra (numba) installed, **linear** `PhenoShape` on an
in-memory raster automatically uses a GIL-releasing, Numba-parallel kernel —
about **12× faster** than the pure-Python path on a multi-core machine, with
identical results. No code change is needed; it is used whenever the inputs are
eligible (linear method, in memory, no NaNs, no custom params) and falls back to
the pure-Python path otherwise.

```python
shape = da.pheno.PhenoShape(interpolType="linear")  # Numba kernel if available
```

## Scaling extraction and the other methods (Dask)

Metric extraction (`PhenoLSP`) and the non-linear reconstruction methods run the
single NumPy/SciPy implementation via `numpy.vectorize`. Because that path holds
the GIL, the default *threaded* Dask scheduler helps **memory** (out-of-core) but
not wall-clock; for CPU parallelism use the *process* scheduler on a chunked
input:

```python
import dask

shape = da.pheno.PhenoShape(chunks={"x": 256, "y": 256})
with dask.config.set(scheduler="processes"):
    lsp = shape.pheno.PhenoLSP().compute()
```

**Honest expectation:** the `processes` scheduler pays a per-worker start-up cost
(each process re-imports the scientific stack and tiles are pickled to it), so on
*small / medium in-memory* rasters the speed-up is modest and eager may even win.
The benefit grows with raster size — where per-tile compute dwarfs the start-up
cost — and the primary reason to chunk at all is **bounded memory** (processing
larger-than-RAM data), not raw speed. Benchmark on *your* data and hardware.

## Why there is no Numba kernel for extraction

A deliberate design choice for robustness. Extraction calls SciPy
(`scipy.stats.skew`, `scipy.integrate.trapezoid`) and dispatches through the
Python extractor registry (axis 3) — neither of which Numba's `@njit` can
compile. A Numba kernel would therefore be a **second implementation** of the
18-metric logic that must stay byte-for-byte identical to the reference one
forever: a maintenance and drift hazard for a marginal, default-path-only gain.

So PhenoPY keeps **one** implementation and scales it with Dask, and reserves
Numba for the place it is a clean win: the hot, simple, SciPy-free linear
reconstruction loop. (A pure-NumPy micro-optimisation of `_getLSPmetrics2` — one
implementation, faster for every backend — is the preferred next step if
extraction proves to be a bottleneck.)

## Choosing chunk sizes

`phenopy.utils.computeChunkSize` suggests a chunking that targets ~100 MB tiles
while keeping the time axis whole:

```python
from phenopy.utils import computeChunkSize

chunks = computeChunkSize(da, sizeMB=100)
shape = da.pheno.PhenoShape(chunks=chunks)
```

## Benchmarks

`benchmarks/bench.py` (run from the repo root in the dev env) times both paths on
a tiled copy of the bundled SIF sample and **asserts the parallel result equals
the serial one**:

1. `PhenoShape(linear)` — Numba `[fast]` vs pure-Python.
2. `PhenoLSP` — eager (1 core) vs chunked + Dask `processes`.
