"""Temporal trend analysis on phenological metric time series.

Computes, per pixel, the Theil-Sen slope and the Mann-Kendall significance of a
metric time series (e.g. one of the arrays returned by
:meth:`~phenopy.phenopy.Pheno.get_timeseries_metrics`). This is the layer that
turns the interannual moving-window series into a statement such as *"is the
start of season getting earlier, and is it significant?"*.

The Mann-Kendall test is implemented directly (with the standard tie
correction), so trend analysis only needs numpy + scipy.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import norm, theilslopes


def _mann_kendall_p(y: np.ndarray) -> float:
    """Two-sided p-value of the Mann-Kendall trend test for a 1D series."""
    n = y.size
    s = 0.0
    for k in range(n - 1):
        s += np.sum(np.sign(y[k + 1 :] - y[k]))
    _, counts = np.unique(y, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    if var_s <= 0:
        return np.nan
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    return float(2 * (1 - norm.cdf(abs(z))))


def _slope_pvalue(series: np.ndarray):
    s = np.asarray(series, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() < 4:  # too few points for a meaningful trend
        return np.nan, np.nan
    t = np.arange(s.size)[mask]
    slope = float(theilslopes(s[mask], t)[0])
    return slope, _mann_kendall_p(s[mask])


def trend(da: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Per-pixel Theil-Sen slope and Mann-Kendall p-value along ``dim``.

    Parameters
    ----------
    da:
        A metric time series, e.g. one of the arrays returned by
        ``get_timeseries_metrics`` (dims ``(time, y, x)``).
    dim:
        Name of the temporal dimension. Default ``"time"``.

    Returns
    -------
    xarray.Dataset
        ``slope`` (metric units per time step, Theil-Sen estimator) and
        ``p_value`` (Mann-Kendall, two-sided). Pixels with fewer than four
        finite observations are NaN.
    """
    slope, pval = xr.apply_ufunc(
        _slope_pvalue,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    return xr.Dataset({"slope": slope, "p_value": pval})
