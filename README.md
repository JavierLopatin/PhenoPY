<p align="center">
  <a href="https://github.com/JavierLopatin/PhenoSensing">
    <img src="phenosensing/data/logo.svg" height="240" alt="PhenoSensing logo">
  </a>
</p>

<h1 align="center">PhenoSensing</h1>

<h4 align="center">Land surface phenology from satellite image time series, built on xarray and Dask</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>

PhenoSensing reconstructs a smoothed seasonal *phenological shape* for every pixel of
a vegetation-index (NDVI, EVI, kNDVI, SIF, …) time series and extracts **18
land-surface-phenology (LSP) metrics** from it, with first-class support for
**interannual** analysis and the **Southern Hemisphere**. It is `xarray`-native and
Dask-aware, so the same code runs on a single pixel or a larger-than-memory raster.

PhenoSensing is a clean, installable, tested, MIT-licensed Python library with a
composable, `xarray`-native and Dask-aware workflow; its land-surface-phenology methods
are reimplemented from the primary literature.

What's distinctive about PhenoSensing:

- a **composable 3-axis pipeline** — any reconstruction × extraction × temporal mode;
- **interannual** moving-window metrics (`get_timeseries_metrics`);
- **segmented-RMSE interannual stability** (`rmse_sos`/`rmse_pos`/`rmse_eos`, with an RGB
  composite map), after Lopatin (2023);
- source-agnostic **QA weighting** — decode any quality band into per-observation weights;
- first-class **Southern-Hemisphere** support; and an `xarray` accessor with a **Numba**
  fast path and **Dask** out-of-core for larger-than-memory rasters.

Several widely used phenology toolkits (e.g. phenofit, npphen, greenbrown, TIMESAT) are
written in R or MATLAB — see [Methods & references](docs/methods.md) for related projects,
in Python and beyond.

## Composable 3-axis design

Phenology extraction is decomposed into **three orthogonal axes** — any combination is
valid, and new methods plug *into* an axis rather than replacing the pipeline:

1. **Reconstruction** — smooth/fit the noisy series (`PhenoShape`): `linear`, `RBF`,
   `savgol`, `whittaker`, `dlog_beck`, `dlog_elmore`, `agauss`, `upper_envelope`
   (wTSM/TIMESAT), `KDE`.
2. **Extraction** — locate SOS/POS/EOS on the curve (`PhenoLSP`): `seasonal_median`,
   `trs`, `der`, `curvature`.
3. **Temporal** — pool a climatology, a single year, or a **moving multi-year window**
   (`get_timeseries_metrics`).

Trends, anomalies, multi-season counts, per-metric uncertainty and QA weighting are
**layers on top**. See **[Methods & references](docs/methods.md)** for the algorithm
behind each method and its primary paper.

## Highlights

- A clean `xarray` accessor: `da.pheno.PhenoShape(...).pheno.PhenoLSP()`.
- **18 LSP metrics**: SOS, POS, EOS, MOS; vSOS/vPOS/vEOS, trough; LOS, amplitude,
  integral; spring/autumn midpoints & values; greening/senescence rates; skewness.
- **Interannual time series** via a moving multi-year window.
- **Segmented RMSE** (`rmse_sos`/`rmse_pos`/`rmse_eos`) as an interannual-stability
  metric (Lopatin, 2023).
- **Trends** (Theil–Sen + Mann–Kendall), **anomalies** (npphen-style RFD),
  **multi-season** counts, and **bootstrap uncertainty** per metric.
- **QA weighting**: decode any quality band (MODIS / Landsat / Sentinel-2 or a custom
  spec) into per-observation weights for a QA-weighted reconstruction.
- **Southern-Hemisphere** day-of-year reordering.
- **Dask-aware** with an optional **Numba** fast path.


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
(whittaker-eilers/pymannkendall), `kde` (KDEpy), `fast` (numba), `gee`
(earthengine-api/xee — for the example notebook only), `test`, `docs`.

## Quickstart

```python
import phenosensing
from phenosensing import load_sample

# Bundled example: SIF time series over Chile (2001-2020)
da = load_sample("SIF")          # (time, y, x) with doy/year coords

# 1. Reconstruct a smoothed phenological shape per pixel (pick any reconstructor)
shape = da.pheno.PhenoShape(interpolType="whittaker", rollWindow=5, nGS=52)

# 2. Extract the 18 land-surface-phenology metrics
lsp = shape.pheno.PhenoLSP()     # xarray.Dataset: sos, pos, eos, ...

# 3. Goodness of fit / interannual stability, segmented by phenophase
rmse = shape.pheno.RMSE(da, LSP_stack=lsp, normalized=True, segment=True)

# 4. Interannual time series via a moving multi-year window
ts = da.pheno.get_timeseries_metrics(window_length=3, metric=["sos", "pos", "eos"])
```

Analysis layers:

```python
from phenosensing import trend, anomaly, n_seasons, uncertainty, qa_to_weight

sos_trend = trend(ts["sos"])              # Theil-Sen slope + Mann-Kendall p
an        = anomaly(da)                   # anomaly, z, RFD percentile (npphen-style)
nos       = n_seasons(shape)              # growing seasons per pixel
unc       = uncertainty(da, n_boot=50)    # bootstrap std of each metric

# QA-weighted reconstruction (decode a quality band -> weights)
w     = qa_to_weight(qa_band, "MOD13Q1")  # or LANDSAT_C2 / S2_SCL / custom spec
shape = da.pheno.PhenoShape(weights=w)
```

## Documentation & examples

- **[Quickstart](docs/quickstart.md)** · **[Methods & references](docs/methods.md)** ·
  **[Performance](docs/performance.md)** · **[API reference](docs/api.md)**
  (build the site with `mkdocs serve`).
- An **executed full-API tutorial** exercising every function:
  [`examples/tutorial.ipynb`](examples/tutorial.ipynb).
- A **Google Earth Engine** example pulling real cloud-contaminated MODIS / Landsat /
  Sentinel-2 data and validating QA weighting:
  [`examples/earthengine_torres_del_paine.ipynb`](examples/earthengine_torres_del_paine.ipynb).

The example SIF data is a small sample over Chile, derived from
[Chen et al. (2022)](https://www.nature.com/articles/s41597-022-01520-1).

## Citation

If you use the segmented-RMSE interannual-variability approach, please cite:

> Lopatin, J. (2023). Interannual Variability of Remotely Sensed Phenology Relates to
> Plant Communities. *IEEE Geoscience and Remote Sensing Letters*, 20, 1–5.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
