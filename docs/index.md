# PhenoSensing

Land surface phenology metrics from satellite image time series, built on
[xarray](https://docs.xarray.dev) and Dask.

PhenoSensing reconstructs a smoothed seasonal *phenological shape* for every pixel of
a vegetation-index time series and extracts **18 land-surface-phenology (LSP)
metrics** from it — start/peak/end of season, rates, integrals, amplitude and more —
with first-class support for **interannual** analysis and the **Southern Hemisphere**.

## Composable 3-axis design

Any combination of the three axes is valid; new methods plug *into* an axis:

1. **Reconstruction** (`PhenoShape`) — `linear`, `RBF`, `savgol`, `whittaker`,
   `dlog_beck`, `dlog_elmore`, `agauss`, `upper_envelope`, `KDE`.
2. **Extraction** (`PhenoLSP`) — `seasonal_median`, `trs`, `der`, `curvature`.
3. **Temporal** (`get_timeseries_metrics`) — climatology, per-year, or a moving
   multi-year window.

See **[Methods & references](methods.md)** for the algorithm and primary paper behind
each method.

## Highlights

- A clean `xarray` accessor: `da.pheno.PhenoShape(...).pheno.PhenoLSP()`.
- **18 LSP metrics** (SOS, POS, EOS, MOS; vSOS/vPOS/vEOS, trough; LOS, amplitude,
  integral; spring/autumn midpoints; greening/senescence rates; skewness).
- **Interannual time series** via a moving multi-year window.
- **Segmented RMSE** as an interannual-stability metric (Lopatin, 2023).
- **Trends** (Theil–Sen + Mann–Kendall), **anomalies** (npphen-style),
  **multi-season** counts, **bootstrap uncertainty**.
- **QA weighting** of any quality band into per-observation weights.
- **Southern-Hemisphere** day-of-year reordering.
- **Dask-aware** with an optional **Numba** fast path.

See **[Quickstart](quickstart.md)** to get going in a few lines, or the
**[API reference](api.md)** for the full surface.

!!! note
    PhenoSensing is the modernized successor to **PhenoPY**, renamed because `phenopy`
    is taken on PyPI. It imports as `phenosensing` (the accessor stays `da.pheno`); a
    PyPI/conda release is in preparation.
