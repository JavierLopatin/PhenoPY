"""Non-parametric phenological anomaly detection (npphen-style).

For each pixel this compares the observed vegetation index against the
climatological phenological cycle (the :meth:`PhenoShape`) and expresses how
anomalous each observation is:

- ``anomaly`` -- the raw deviation (observed - expected);
- ``z`` -- the anomaly standardised by the per-pixel anomaly spread;
- ``rfd`` -- a Reference Frequency Distribution percentile in (0, 1]: where the
  anomaly falls within the pixel's historical anomaly distribution. Values near
  0 are extreme negative anomalies (e.g. drought / dieback), near 1 extreme
  positive (e.g. unusual greening), near 0.5 typical.

Reference: Estay & Chavez, ``npphen`` (non-parametric vegetation phenology and
anomaly detection).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import rankdata


def _rfd_1d(a):
    """Empirical percentile (RFD) of each value among the finite values of ``a``."""
    out = np.full(a.shape, np.nan)
    mask = np.isfinite(a)
    n = int(mask.sum())
    if n < 2:
        return out
    out[mask] = rankdata(a[mask]) / n
    return out


def anomaly(
    da: xr.DataArray,
    reference: xr.DataArray | None = None,
    interpolType: str = "linear",
    nGS: int = 52,
    rollWindow: int = 5,
    standardize: bool = True,
) -> xr.Dataset:
    """Phenological anomalies of ``da`` against a climatological reference.

    Parameters
    ----------
    da:
        ``(time, y, x)`` vegetation-index cube with a ``doy`` coordinate.
    reference:
        Cube used to build the climatology (default: ``da`` itself). Pass a
        held-out reference period to assess a target period against it.
    interpolType:
        Reconstruction method for the climatology. ``"KDE"`` gives a fully
        non-parametric, npphen-style reference (needs the ``[kde]`` extra).
    nGS, rollWindow:
        Passed through to ``PhenoShape`` when building the climatology.
    standardize:
        Also return a z-score (anomaly divided by the per-pixel anomaly std).

    Returns
    -------
    xarray.Dataset
        ``anomaly`` (observed - expected), ``rfd`` (empirical percentile in
        (0, 1]) and, when ``standardize``, ``z`` -- all with dims
        ``(time, y, x)``.
    """
    ref = da if reference is None else reference
    shape = ref.pheno.PhenoShape(interpolType=interpolType, nGS=nGS, rollWindow=rollWindow)

    # expected vegetation index at each observation's day-of-year
    expected = shape.interp(doy=da["doy"])
    anomaly_da = (da - expected).rename("anomaly")

    rfd = xr.apply_ufunc(
        _rfd_1d,
        anomaly_da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).rename("rfd")

    data = {"anomaly": anomaly_da, "rfd": rfd}
    if standardize:
        data["z"] = (anomaly_da / anomaly_da.std("time", skipna=True)).rename("z")
    return xr.Dataset(data)
