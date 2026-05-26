"""Smoke tests for the plotting helpers (Agg backend, no display)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from phenosensing.io import load_sample  # noqa: E402
from phenosensing.plotting import PhenoPlot  # noqa: E402


def _xy(da):
    return float(da.x[da.sizes["x"] // 2]), float(da.y[da.sizes["y"] // 2])


def test_phenoplot_southern_many_years_uses_colorbar():
    sif = load_sample("SIF")  # 20 years -> continuous palette + colorbar
    X, Y = _xy(sif)
    ax = PhenoPlot(sif, X=X, Y=Y, rollWindow=5, ylab="SIF", southern=True)
    assert ax.get_xlabel().lower().startswith("day of season")
    assert ax.get_legend() is None  # many years -> colorbar, not a legend
    plt.close("all")


def test_phenoplot_few_years_uses_legend():
    ndvi = load_sample("ndvi")  # 2 years -> per-year legend
    X, Y = _xy(ndvi)
    ax = PhenoPlot(ndvi, X=X, Y=Y, rollWindow=3)
    assert ax.get_legend() is not None
    plt.close("all")
