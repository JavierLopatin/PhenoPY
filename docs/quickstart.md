# Quickstart

```python
import phenopy
from phenopy import load_sample

# Bundled example: SIF time series over Chile (2001-2020)
da = load_sample("SIF")          # (time, y, x) with doy/year coords

# 1. Reconstruct a smoothed phenological shape per pixel
shape = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52)

# 2. Extract the 16 land-surface-phenology metrics
lsp = shape.pheno.PhenoLSP()     # xarray.Dataset: sos, pos, eos, ...

# 3. Goodness-of-fit (optionally segmented by SOS / POS / EOS)
rmse = shape.pheno.RMSE(da, LSP_stack=lsp, normalized=True, segment=True)

# 4. Interannual time series via a moving multi-year window
ts = da.pheno.get_timeseries_metrics(window_length=3, metric=["sos", "pos", "eos"])
```

## The metrics

`PhenoLSP` returns an `xarray.Dataset` with 16 variables:

| Group       | Variables                          |
| ----------- | ---------------------------------- |
| Timing      | `sos`, `pos`, `eos`                |
| Values      | `vsos`, `vpos`, `veos`             |
| Season      | `los`, `ampl`, `ios`               |
| Spring/Aut. | `msp`, `mau`, `vmsp`, `vmau`       |
| Rates/shape | `rog`, `ros`, `sw`                 |

## Trends and anomalies

```python
from phenopy import anomaly, trend

# Interannual trend of start-of-season: Theil-Sen slope + Mann-Kendall p-value
ts = da.pheno.get_timeseries_metrics(window_length=3, metric=["sos"])
sos_trend = trend(ts["sos"])

# Per-observation phenological anomalies + RFD percentile (npphen-style)
an = anomaly(da)   # -> Dataset with `anomaly`, `z`, `rfd`
```

## Southern Hemisphere

For Southern-Hemisphere sites, reorder the day-of-year so the growing season is
contiguous before computing the shape:

```python
from phenopy import reorder_southern_hemisphere

positions, da_sh = reorder_southern_hemisphere(da)
shape = da_sh.pheno.PhenoShape()
```
