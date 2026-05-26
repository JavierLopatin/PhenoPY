"""Multi-season detection (NOS, Number of Seasons).

Counts the growing cycles in a reconstructed phenological curve per pixel, to
distinguish single-cycle vegetation from multi-cropping / bimodal systems.
Peaks are found with ``scipy.signal.find_peaks`` using a prominence threshold
relative to the per-pixel amplitude, so noise is not counted as a season.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import find_peaks


def _count_peaks(curve, prominence_frac, distance, min_height_frac):
    c = np.asarray(curve, dtype=float)
    if np.isfinite(c).sum() < 3:
        return np.nan
    cmin = np.nanmin(c)
    amp = np.nanmax(c) - cmin
    if amp <= 0:
        return 0.0
    prominence = prominence_frac * amp
    height = cmin + min_height_frac * amp if min_height_frac is not None else None
    peaks, _ = find_peaks(c, prominence=prominence, distance=distance, height=height)
    return float(peaks.size)


def n_seasons(phenoshape, prominence=0.1, distance=None, min_height=None) -> xr.DataArray:
    """Number of growing seasons (NOS) per pixel from a reconstructed curve.

    Parameters
    ----------
    phenoshape:
        A PhenoShape output with a ``doy`` dimension (or a ``doy`` coordinate on
        a ``time`` dimension).
    prominence:
        Minimum peak prominence as a fraction of the per-pixel amplitude
        (default 0.1 = 10%). Higher values count only the stronger seasons.
    distance:
        Optional minimum spacing between peaks, in samples of the ``doy`` grid.
    min_height:
        Optional minimum peak height as a fraction of the amplitude above the
        per-pixel trough.

    Returns
    -------
    xarray.DataArray
        Per-pixel season count, dims ``(y, x)``.
    """
    if "doy" not in phenoshape.dims:
        if "doy" in phenoshape.coords and "time" in phenoshape.dims:
            phenoshape = phenoshape.swap_dims({"time": "doy"})
        else:
            raise ValueError("The provided DataArray must have 'doy' as a dimension or coordinate.")

    return xr.apply_ufunc(
        _count_peaks,
        phenoshape,
        input_core_dims=[["doy"]],
        kwargs={
            "prominence_frac": prominence,
            "distance": distance,
            "min_height_frac": min_height,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
