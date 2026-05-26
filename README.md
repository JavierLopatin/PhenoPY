<h1 align="center">
<a href='https://github.com/JavierLopatin/PhenoSensing'><img src='phenosensing/data/logo.svg' align="right" height="300" /></a>
PhenoSensing
</h1>

<h4 align="center">Land surface phenology from satellite image time series, built on xarray and Dask</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>

PhenoSensing reconstructs a smoothed seasonal *phenological shape* for every pixel of
a vegetation-index (NDVI, EVI, kNDVI, SIF, …) time series and extracts **16
land-surface-phenology (LSP) metrics** from it, with first-class support for
**interannual** analysis and the **Southern Hemisphere**.

## Highlights

- A clean `xarray` accessor: `da.pheno.PhenoShape(...).pheno.PhenoLSP()`.
- **16 LSP metrics**: SOS, POS, EOS, vSOS/vPOS/vEOS, LOS, MSP, MAU, vMSP/vMAU,
  amplitude, integral, rates of greening/senescence, and skewness.
- **Interannual time series** via a moving multi-year window
  (`get_timeseries_metrics`).
- **Curvature** and **(segmented) RMSE** goodness-of-fit.
- **Southern-Hemisphere** day-of-year reordering.
- **Dask-aware** for larger-than-memory rasters.

> [!NOTE]
> PhenoSensing is the modernized successor to **PhenoPY**, renamed because
> `phenopy` is taken on PyPI by an unrelated clinical-phenotyping tool. It
> installs and imports as **`phenosensing`** (the xarray accessor stays
> `da.pheno`). A PyPI/conda release is in preparation.

## Installation

From source for now (a PyPI/conda release is in progress):

```bash
git clone https://github.com/JavierLopatin/PhenoSensing.git
cd PhenoSensing

# Recommended: conda dev environment (reliable geospatial stack)
conda env create -f environment-dev.yml
conda activate phenosensing
pip install -e ".[fit,plot,test]"
```

Optional extras: `plot` (matplotlib/folium/pyproj), `fit`
(whittaker-eilers/pymannkendall), `kde` (KDEpy), `fast` (numba),
`test`, `docs`.

## Quickstart

```python
import phenosensing
from phenosensing import load_sample

# Bundled example: SIF time series over Chile (2001-2020)
da = load_sample("SIF")          # (time, y, x) with doy/year coords

# 1. Reconstruct a smoothed phenological shape per pixel
shape = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52)

# 2. Extract the 16 land-surface-phenology metrics
lsp = shape.pheno.PhenoLSP()     # xarray.Dataset: sos, pos, eos, ...

# 3. Interannual time series via a moving multi-year window
ts = da.pheno.get_timeseries_metrics(window_length=3, metric=["sos", "pos", "eos"])
```

A full walk-through is in the example notebook:
[`phenosensing/ExampleData.ipynb`](https://github.com/JavierLopatin/PhenoSensing/blob/master/phenosensing/ExampleData.ipynb),
and an **executed tutorial exercising every function** is in
[`examples/tutorial.ipynb`](examples/tutorial.ipynb).
The example SIF data is a small sample over Chile, derived from
[Chen et al. (2022)](https://www.nature.com/articles/s41597-022-01520-1).

## License

MIT — see [LICENSE.txt](LICENSE.txt).
