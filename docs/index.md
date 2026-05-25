# PhenoPY

Land surface phenology metrics from satellite image time series, built on
[xarray](https://docs.xarray.dev) and Dask.

PhenoPY reconstructs a smoothed seasonal *phenological shape* for every pixel of
a vegetation-index time series and extracts 16 land-surface-phenology (LSP)
metrics from it — start/peak/end of season, rates, integrals, amplitude and
more — with first-class support for **interannual** analysis and the **Southern
Hemisphere**.

## Highlights

- A clean `xarray` accessor: `da.pheno.PhenoShape(...).pheno.PhenoLSP()`.
- 16 LSP metrics (SOS, POS, EOS, vSOS/vPOS/vEOS, LOS, MSP, MAU, amplitude,
  integral, rates of greening/senescence, skewness).
- Interannual time series via a moving multi-year window
  (`get_timeseries_metrics`).
- Curvature and (segmented) RMSE goodness-of-fit.
- Southern-Hemisphere day-of-year reordering.
- Dask-aware for larger-than-memory rasters.

See **[Quickstart](quickstart.md)** to get going in a few lines, or the
**[API reference](api.md)** for the full surface.

!!! note
    PhenoPY is being modernized toward a PyPI/conda release. The core accessor
    API is stable, while new reconstruction and extraction methods are being
    added.
