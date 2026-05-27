# Quickstart

```python
import phenosensing
from phenosensing import load_sample

# Bundled example: SIF time series over Chile (2001-2020)
da = load_sample("SIF")          # (time, y, x) with doy/year coords

# 1. Reconstruct a smoothed phenological shape per pixel
shape = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52)

# 2. Extract the 18 land-surface-phenology metrics
lsp = shape.pheno.PhenoLSP()     # xarray.Dataset: sos, pos, eos, ...

# 3. Goodness-of-fit / interannual stability (optionally segmented by phenophase)
rmse = shape.pheno.RMSE(da, LSP_stack=lsp, normalized=True, segment=True)

# 4. Interannual time series via a moving multi-year window
ts = da.pheno.get_timeseries_metrics(window_length=3, metric=["sos", "pos", "eos"])
```

## Choosing a reconstruction / extraction

The reconstruction (axis 1) and extraction (axis 3) are independent — mix freely:

```python
from phenosensing import list_reconstructors, list_extractors

list_reconstructors()  # linear, RBF, savgol, whittaker, dlog_beck, dlog_elmore, agauss, upper_envelope, KDE
list_extractors()      # seasonal_median, trs, der, curvature

shape = da.pheno.PhenoShape(interpolType="whittaker", recon_params={"lmbd": 15})
lsp   = shape.pheno.PhenoLSP(extraction="trs", extract_params={"threshold": 0.5})
```

See **[Methods & references](methods.md)** for what each method does and its paper.

## The metrics

`PhenoLSP` returns an `xarray.Dataset` with 18 variables:

| Group | Variables |
| ----- | --------- |
| Timing | `sos`, `pos`, `eos`, `mos` |
| Values | `vsos`, `vpos`, `veos`, `trough` |
| Season | `los`, `ampl`, `ios` |
| Spring / autumn | `msp`, `mau`, `vmsp`, `vmau` |
| Rates / shape | `rog`, `ros`, `sw` |

## Trends, anomalies, uncertainty

```python
from phenosensing import anomaly, n_seasons, trend, uncertainty

# Interannual trend of start-of-season: Theil-Sen slope + Mann-Kendall p-value
ts = da.pheno.get_timeseries_metrics(window_length=3, metric=["sos"])
sos_trend = trend(ts["sos"])

# Per-observation phenological anomalies + RFD percentile (npphen-style)
an = anomaly(da)   # -> Dataset with `anomaly`, `z`, `rfd`

# Number of growing seasons per pixel (multi-cropping / bimodal vegetation)
nos = n_seasons(da.pheno.PhenoShape())

# Bootstrap uncertainty (std) of each of the 18 metrics, per pixel
unc = uncertainty(da, n_boot=50)
```

## QA-weighted reconstruction

Decode any quality band into per-observation weights and feed them to the
reconstruction, so noisy/cloudy samples count less:

```python
from phenosensing import qa_to_weight, list_qa_specs

list_qa_specs()                       # MOD13Q1, LANDSAT_C2, S2_SCL
w = qa_to_weight(qa_band, "MOD13Q1")  # or a custom dict / callable
shape = da.pheno.PhenoShape(weights=w)
```

The QA decoder is source-agnostic; Earth Engine is only used in the
[example notebook](https://github.com/JavierLopatin/PhenoSensing/blob/master/examples/earthengine_torres_del_paine.ipynb),
not as a dependency.

## Southern Hemisphere

For Southern-Hemisphere sites, reorder the day-of-year so the growing season is
contiguous before computing the shape:

```python
from phenosensing import reorder_southern_hemisphere

positions, da_sh = reorder_southern_hemisphere(da)
shape = da_sh.pheno.PhenoShape()
```
