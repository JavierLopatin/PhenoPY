# Methods & references

PhenoSensing is organised around **three orthogonal, composable axes**. Any
reconstruction can be combined with any extraction and any temporal mode — new
methods plug *into* an axis rather than replacing the pipeline.

| Axis | What it decides | Where |
| ---- | --------------- | ----- |
| **1. Reconstruction** | how the noisy series is smoothed / fitted into a seasonal curve | `PhenoShape(interpolType=…, recon_params=…)` |
| **2. Extraction** | how SOS / POS / EOS and the other metrics are located on the curve | `PhenoLSP(extraction=…, extract_params=…)` |
| **3. Temporal** | which years are pooled (climatology, per-year, or a moving multi-year window) | `get_timeseries_metrics(window_length=…)` |

Trends, anomalies, multi-season counts, per-metric uncertainty and QA weighting are
**layers on top** of these axes.

!!! note "Licensing"
    The GPL toolkits (phenofit, npphen, greenbrown, TIMESAT) were **not copied**; the
    algorithms below are reimplemented from the primary literature. PhenoSensing is
    MIT-licensed.

## Axis 1 — Reconstruction

`list_reconstructors()` → `linear, RBF, savgol, whittaker, dlog_beck, dlog_elmore, agauss, upper_envelope, KDE`

| Method | Idea | Reference |
| ------ | ---- | --------- |
| `linear` | piece-wise linear interpolation + rolling mean (default; Numba fast path) | — |
| `RBF` | radial-basis-function interpolation | — |
| `savgol` | Savitzky–Golay polynomial filter | Savitzky & Golay (1964); for NDVI: Chen et al. (2004) |
| `whittaker` | Whittaker–Eilers penalised-least-squares smoother | Eilers (2003) |
| `dlog_beck` | double-logistic fit | Beck et al. (2006) |
| `dlog_elmore` | double-logistic fit (Elmore parameterisation) | Elmore et al. (2012) |
| `agauss` | asymmetric-Gaussian fit | Jönsson & Eklundh (2002, 2004) |
| `upper_envelope` | iterative weighted upper-envelope (wTSM/TIMESAT); pulls the curve toward cloud-free maxima — needs no QA band | Chen et al. (2004); Jönsson & Eklundh (2004) |
| `KDE` | kernel-density non-parametric reconstruction (npphen-style; needs the `[kde]` extra) | Chávez et al. (2023) |

**Weighted reconstruction.** `PhenoShape(weights=…)` threads per-observation weights
into a QA-weighted Whittaker, so noisy/cloudy samples count less. Weights usually come
from [`qa_to_weight`](#axis-3-temporal-and-analysis-layers). With `weights=None` the
output is byte-identical to the unweighted path.

## Axis 2 — Extraction

`list_extractors()` → `seasonal_median, trs, der, curvature`

| Method | Idea | Reference |
| ------ | ---- | --------- |
| `seasonal_median` | threshold relative to the seasonal median (historical default) | — |
| `trs` | fixed amplitude threshold (`extract_params={"threshold": 0.5}`) for SOS/EOS | — |
| `der` | derivative extrema (steepest greening / senescence) | — |
| `curvature` | rate of change of curvature (inflection points) | Zhang et al. (2003) |

The **18 LSP metrics** returned by `PhenoLSP`:

| Group | Variables |
| ----- | --------- |
| Timing | `sos`, `pos`, `eos`, `mos` (middle-of-season) |
| Values | `vsos`, `vpos`, `veos`, `trough` (base) |
| Season | `los`, `ampl`, `ios` |
| Spring / autumn | `msp`, `mau`, `vmsp`, `vmau` |
| Rates / shape | `rog`, `ros`, `sw` |

## Axis 3 — Temporal and analysis layers

- **`get_timeseries_metrics`** slides a moving multi-year window, producing a time
  series of every metric — PhenoSensing's signature interannual feature.
- **`RMSE(segment=True)`** — RMSE of the observations against the fitted curve, split
  by phenophase (`rmse_sos` / `rmse_pos` / `rmse_eos`). Against a multi-year
  climatology this is an **interannual-stability** metric, composited as an RGB map
  (R/G/B = Beginning/Middle/End) — Lopatin (2023).
- **`trend`** — Theil–Sen slope + Mann–Kendall significance over the metric series
  (via `pymannkendall`).
- **`anomaly`** — non-parametric departure from the climatological shape + RFD
  percentile, npphen-style — Chávez et al. (2023).
- **`n_seasons`** — number of growing seasons per pixel (multi-cropping / bimodal).
- **`uncertainty`** — bootstrap standard deviation of each metric per pixel.
- **`qa_to_weight`** — source-agnostic decoder turning any quality band into
  per-observation weights (registry keys `MOD13Q1`, `LANDSAT_C2`, `S2_SCL`, or a custom
  dict / callable). Earth Engine is used only in the
  [example notebook](https://github.com/JavierLopatin/PhenoSensing/blob/master/examples/earthengine_torres_del_paine.ipynb),
  never as a dependency.

## Related libraries

| Library | Lang. | License | Notes |
| ------- | ----- | ------- | ----- |
| [phenofit](https://github.com/eco-hydro/phenofit) | R | GPL | rich curve-fitting + weight updating (Kong et al., 2022) |
| [npphen](https://github.com/labGRS/npphen) | R | GPL | non-parametric phenology + anomaly mapping (Chávez et al., 2023) |
| [greenbrown](https://greenbrown.r-forge.r-project.org/) | R | GPL | LSP + trend/breakpoint analysis (Forkel et al., 2013) |
| TIMESAT | C/MATLAB | — | SG / asym.-Gaussian / double-logistic + wTSM (Jönsson & Eklundh, 2004) |
| [Phenolopy](https://github.com/lewistrotter/Phenolopy) | Python | Apache-2.0 | xarray-based, unpackaged |
| [pyPhenology](https://github.com/sdtaylor/pyPhenology) | Python | MIT | species-level phenology models (not LSP from imagery) |

PhenoSensing's niche: a clean, installed, tested, **xarray-/Dask-native** Python LSP
library — the most complete alternatives are R/MATLAB, and the closest Python one is
unpackaged.

## References

- Beck, P. S. A., Atzberger, C., Høgda, K. A., Johansen, B., & Skidmore, A. K. (2006).
  Improved monitoring of vegetation dynamics at very high latitudes. *Remote Sensing of
  Environment*, 100, 321–334.
  [doi:10.1016/j.rse.2005.10.021](https://doi.org/10.1016/j.rse.2005.10.021)
- Chávez, R. O., Estay, S. A., Lastra, J. A., Riquelme, C. G., Olea, M., Aguayo, J., &
  Decuyper, M. (2023). npphen: An R-package for detecting and mapping extreme vegetation
  anomalies. *Remote Sensing*, 15, 73.
  [doi:10.3390/rs15010073](https://doi.org/10.3390/rs15010073)
- Chen, J., Jönsson, P., Tamura, M., Gu, Z., Matsushita, B., & Eklundh, L. (2004). A
  simple method for reconstructing a high-quality NDVI time-series data set based on the
  Savitzky–Golay filter. *Remote Sensing of Environment*, 91, 332–344.
  [doi:10.1016/j.rse.2004.03.014](https://doi.org/10.1016/j.rse.2004.03.014)
- Eilers, P. H. C. (2003). A perfect smoother. *Analytical Chemistry*, 75, 3631–3636.
  [doi:10.1021/ac034173t](https://doi.org/10.1021/ac034173t)
- Elmore, A. J., Guinn, S. M., Minsley, B. J., & Richardson, A. D. (2012). Landscape
  controls on the timing of spring, autumn, and growing season length in mid-Atlantic
  forests. *Global Change Biology*, 18, 656–674.
  [doi:10.1111/j.1365-2486.2011.02521.x](https://doi.org/10.1111/j.1365-2486.2011.02521.x)
- Forkel, M., Carvalhais, N., Verbesselt, J., Mahecha, M. D., Neigh, C. S. R., &
  Reichstein, M. (2013). Trend change detection in NDVI time series. *Remote Sensing*, 5,
  2113–2144. [doi:10.3390/rs5052113](https://doi.org/10.3390/rs5052113)
- Jönsson, P., & Eklundh, L. (2002). Seasonality extraction by function fitting to
  time-series of satellite sensor data. *IEEE Transactions on Geoscience and Remote
  Sensing*, 40, 1824–1832.
  [doi:10.1109/TGRS.2002.802519](https://doi.org/10.1109/TGRS.2002.802519)
- Jönsson, P., & Eklundh, L. (2004). TIMESAT — a program for analyzing time-series of
  satellite sensor data. *Computers & Geosciences*, 30, 833–845.
  [doi:10.1016/j.cageo.2004.05.006](https://doi.org/10.1016/j.cageo.2004.05.006)
- Kong, D., McVicar, T. R., Xiao, M., Zhang, Y., Peña-Arancibia, J. L., Filippa, G.,
  Xie, Y., & Gu, X. (2022). phenofit: An R package for extracting vegetation phenology
  from time series remote sensing. *Methods in Ecology and Evolution*, 13, 1508–1527.
  [doi:10.1111/2041-210X.13870](https://doi.org/10.1111/2041-210X.13870)
- Lopatin, J. (2023). Interannual variability of remotely sensed phenology relates to
  plant communities. *IEEE Geoscience and Remote Sensing Letters*, 20, 1–5.
  [IEEE Xplore](https://ieeexplore.ieee.org/document/10128132)
- Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of data by
  simplified least squares procedures. *Analytical Chemistry*, 36, 1627–1639.
  [doi:10.1021/ac60214a047](https://doi.org/10.1021/ac60214a047)
- Zhang, X., Friedl, M. A., Schaaf, C. B., Strahler, A. H., Hodges, J. C. F., Gao, F.,
  Reed, B. C., & Huete, A. (2003). Monitoring vegetation phenology using MODIS. *Remote
  Sensing of Environment*, 84, 471–475.
  [doi:10.1016/S0034-4257(02)00135-9](https://doi.org/10.1016/S0034-4257(02)00135-9)
- Mann–Kendall test (Mann, 1945; Kendall, 1975) and Theil–Sen slope (Sen, 1968),
  computed via [pymannkendall](https://github.com/mmhs013/pyMannKendall).
