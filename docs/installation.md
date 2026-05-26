# Installation

PhenoSensing is being prepared for a PyPI/conda release. For now, install from source.

## Conda development environment (recommended)

The geospatial stack (GDAL / rasterio / rioxarray) resolves most reliably on
conda-forge:

```bash
git clone https://github.com/JavierLopatin/PhenoSensing.git
cd PhenoSensing
conda env create -f environment-dev.yml
conda activate phenosensing
pip install -e ".[fit,plot,test]"
```

## Plain pip

```bash
pip install -e ".[fit,plot]"
```

## Optional extras

| Extra  | Adds                                                |
| ------ | --------------------------------------------------- |
| `plot` | matplotlib, folium, pyproj (plotting / maps)        |
| `fit`  | whittaker-eilers, pymannkendall (smoothing, trends) |
| `kde`  | KDEpy (non-parametric reconstruction)               |
| `fast` | numba (accelerated kernels)                         |
| `test` | pytest, pytest-cov, ruff                            |
| `docs` | mkdocs-material, mkdocstrings                        |
