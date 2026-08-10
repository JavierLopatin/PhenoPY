import sys
from functools import reduce

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from scipy.stats import skew

from .extraction import get_extractor
from .phase import DEFAULT_STRENGTH_THRESH, _unrotate_doy, resolve_anchor
from .reconstruction import get_reconstructor

# Canonical order of the 18 land-surface-phenology bands from _getLSPmetrics2.
LSP_BANDS = [
    "sos",
    "pos",
    "eos",
    "vsos",
    "vpos",
    "veos",
    "los",
    "msp",
    "mau",
    "vmsp",
    "vmau",
    "ampl",
    "ios",
    "rog",
    "ros",
    "sw",
    "trough",
    "mos",
]


def reorder_southern_hemisphere(img: xr.Dataset) -> tuple:
    """
    Reorder the DOY of the img for the Southern Hemisphere to ensure
    peak summer is in the middle. This reordering is critical for some applications
    like phenological studies.

    This is the *global, manual* special case of phase anchoring: it shifts the
    whole cube by ~half a year (the austral year). For heterogeneous or
    cross-equator scenes — where some pixels peak in austral summer and others in
    austral winter — prefer the *per-pixel* ``PhenoLSP(hemisphere="auto")`` /
    :func:`phenosensing.phase.season_phase`, which anchors each pixel
    individually. ``hemisphere="south"`` reproduces this global behaviour.

    :param img: xarray Dataset with `time` and `doy` dimensions/coordinates.
    :return: The input xarray Dataset reordered by DOY.
    """
    doy = img.doy.values
    time_values = img.time.values

    arr1 = np.linspace(183, 365, int(365 - 180)).astype(int)
    arr2 = np.linspace(1, 185, 180).astype(int)
    result = np.concatenate((arr1, arr2))
    positions = [np.where(result == value)[0][0] for value in doy]

    # Create a mapping of DOY value to its position in 'result'
    position_map = {value: np.where(result == value)[0][0] for value in doy}

    # Reorder the time values based on the mapping
    reordered_times = sorted(
        time_values, key=lambda x: position_map[img.sel(time=x).doy.values.item()]
    )

    # Re-index the img using the reordered time values
    da = img.sel(time=reordered_times)
    da.doy.values = np.linspace(1, 365, len(da.time.values))  # np.sort(doy)#

    return positions, da


def _getPheno(y, x, nGS, interpolType, recon_params=None, weights=None):
    """
    Reconstruct a regular phenological curve from an irregular (doy, value) series.

    x: DOY values
    y: ndarray with VI values
    interpolType: reconstruction method name (see ``phenosensing.reconstruction``)
    recon_params: optional dict of method-specific parameters
    weights: optional per-observation weights in [0, 1] (e.g. from a QA band),
        passed to reconstructors that support them (e.g. ``whittaker``).
    """
    inds = np.isnan(y)  # check if array has NaN values
    if np.sum(inds) == len(y):  # check if all values are NaN
        return y[0:nGS]
    try:
        xnew = np.linspace(np.min(x), np.max(x), nGS, dtype=np.int32)
        if inds.any():  # if inds have at least one True
            y = _fillNaN(y)
            _replaceElements(x)  # replace doy values when they are the same
        reconstruct = get_reconstructor(interpolType)
        # only forward ``weights`` when given, so the unweighted call is unchanged
        extra = {} if weights is None else {"weights": weights}
        ynew = reconstruct(x, y, xnew, **extra, **(recon_params or {}))
        return ynew
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.full(nGS, np.nan)


def _getPheno0(y, doy, interpolType, nan_replace, rollWindow, nGS, recon_params=None,
               weights=None, rollMode="wrap"):
    # replace nan_replace values by NaN
    if nan_replace is not None:
        y = np.where(y == nan_replace, np.nan, y)

    # sort values by DOY (and the weights alongside, when provided)
    idx = doy.argsort()
    y = y[idx]
    w = weights[idx] if weights is not None else None

    # get phenological shape (reconstruction)
    phen = _getPheno(y, doy[idx], nGS, interpolType, recon_params=recon_params, weights=w)

    # rolling average using moving window
    if rollWindow is not None:
        phen = _moving_average(phen, rollWindow, mode=rollMode)

    return phen


def _getPheno0_weighted(y, weights, doy, interpolType, nan_replace, rollWindow, nGS,
                        recon_params=None, rollMode="wrap"):
    """``_getPheno0`` with the weights as a second positional array, for the
    two-input ``xr.apply_ufunc`` path used when ``PhenoShape(weights=...)``."""
    return _getPheno0(
        y, doy, interpolType, nan_replace, rollWindow, nGS,
        recon_params=recon_params, weights=weights, rollMode=rollMode,
    )


def _getLSPmetrics2(
    phen,
    xnew,
    nGS,
    bands,
    phentype=1,
    extraction=None,
    extract_params=None,
    hemisphere="north",
    offset=None,
    strength_thresh=DEFAULT_STRENGTH_THRESH,
):
    """
    Obtain land surfurface phenology metrics

    Parameters
    ----------
    - phen: 1D array
        PhenoShape data
    - xnew: 1D array
        DOY values for PhenoShape data
    - bands: string list
        Name of the output bands (soft requirement)
    - hemisphere: {"north", "south", "auto"}
        Per-pixel seasonal-phase anchoring. ``"north"`` (default) does not rotate
        and is byte-identical to the historical behaviour. ``"south"`` rotates by
        a fixed half year (austral year, ~1 July). ``"auto"`` detects the pixel's
        season from its first annual harmonic and rotates it into a centered
        "phenological-year" frame, mapping the DOY-valued metrics back to calendar
        DOY; aseasonal pixels (weak annual cycle) are left unrotated.
        See :mod:`phenosensing.phase`.
    - offset: float, optional
        Explicit anchor DOY; overrides ``hemisphere`` (used internally / in tests).

    Outputs
    -------
    - 2D array with the following variables:

        SOS = DOY of start of season
        POS = DOY of peak of season
        EOS = DOY of end of season
        vSOS = Value at start of season
        vPOS = Value at peak of season
        vEOS = Value at end of season
        LOS = Length of season (DOY)
        MSP = Mean spring (DOY)
        MAU = Mean autum (DOY)
        vMSP = Mean spring value
        vMAU = Mean autum value
        AOS = Amplitude of season (in value units)
        IOS = Integral of season (SOS-EOS)
        ROG = Rate of greening [slope SOS-POS]
        ROS = Rate of senescence [slope POS-EOS]
        SW = Skewness of growing season
        TROUGH = Trough / base value (minimum of the curve, in value units)
        MOS = Middle of season (DOY, midpoint between SOS and EOS)
    """
    inds = np.isnan(phen)  # check if array has NaN values
    if inds.any():  # check is all values are NaN
        return np.repeat(np.nan, len(bands))

    # Per-pixel phase anchoring (axis: hemisphere). For the default "north" path
    # nothing is rotated, so the metrics below are byte-identical to the historical
    # code. Otherwise rotate the pixel into a centered "phenological-year" frame,
    # run the metric body on the rotated curve, and map the DOY-valued metrics back
    # to calendar DOY at the end. Aseasonal pixels resolve to ``None`` (no rotation).
    anchor_doy = None
    if not (hemisphere == "north" and offset is None):
        anchor = resolve_anchor(phen, xnew, hemisphere, offset, strength_thresh)
        if anchor is not None:
            xnew_arr = np.asarray(xnew, dtype=float)
            roll_k = int(np.argmin(np.abs(xnew_arr - anchor)))
            anchor_doy = float(xnew_arr[roll_k])
            phen = np.roll(phen, -roll_k)

    # basic variables
    vpos = np.max(phen)
    trough = np.min(phen)
    ampl = vpos - trough

    # get position of seasonal peak and trough
    ipos = np.where(phen == vpos)[0]
    pos = xnew[ipos]

    # scale annual time series to 0-1
    ratio = (phen - trough) / ampl

    # separate greening from senesence values
    dev = np.gradient(ratio)  # first derivative
    greenup = np.zeros([ratio.shape[0]], dtype=bool)
    greenup[dev > 0] = True

    # determine SOS / EOS via the selected extraction method (axis 3)
    if extraction is None:
        extraction = "seasonal_median"  # historical phentype 1/2 behavior
    sos, eos, isos, ieos = get_extractor(extraction)(
        phen, xnew, ratio, greenup, ipos, **(extract_params or {})
    )
    if sos is None:
        isos = 0
        sos = xnew[isos]
    if eos is None:
        ieos = len(xnew) - 1
        eos = xnew[ieos]

    # los: length of season
    los = eos - sos
    if los < 0:
        los = np.nan

    # get MSP, MAU (independent from SOS and EOS)

    # mean spring
    idx = np.mean(xnew[(xnew > sos) & (xnew < pos[0])])
    idx = (np.abs(xnew - idx)).argmin()  # indexing value
    msp = xnew[idx]  # DOY of MGS
    vmsp = phen[idx]  # mgs value

    # mean autum
    idx = np.mean(xnew[(xnew < eos) & (xnew > pos[0])])
    idx = (np.abs(xnew - idx)).argmin()  # indexing value
    mau = xnew[idx]  # DOY of MGS
    vmau = phen[idx]  # mgs value

    # doy of growing season
    green = xnew[(xnew > sos) & (xnew < eos)]
    id_ = []
    for i in range(len(green)):
        id_.append((xnew == green[i]).nonzero()[0])
    # TODO: move id_ generation to a list comprehension -> id_ = [(xnew == green[i]).nonzero()[0] for i in range(len(green))]

    # index of growing season
    id = np.array([item for sublist in id_ for item in sublist])

    # get intergral of green season
    ios = trapezoid(phen[id], xnew[id]) if len(id) > 0 else np.nan

    # skewness of growing season
    sw = skew(phen[id]) if len(id) > 0 else np.nan

    # rate of greening [slope SOS-POS]
    rog = (vpos - phen[isos]) / (pos - sos)

    # rate of senescence [slope POS-EOS]
    ros = (phen[ieos] - vpos) / (eos - pos)

    # middle of season: midpoint DOY between SOS and EOS (NaN when LOS is)
    mos = (sos + eos) / 2.0 if not np.isnan(los) else np.nan
    # trough/base value already computed above as ``trough = np.min(phen)``

    metrics = np.array(
        (
            sos,
            pos[0],
            eos,
            phen[isos][0],
            vpos,
            phen[ieos][0],
            los,
            msp,
            mau,
            vmsp,
            vmau,
            ampl,
            ios,
            rog[0],
            ros[0],
            sw,
            trough,
            mos,
        )
    )

    # map the DOY-valued metrics from the rotated frame back to calendar DOY
    # (values, durations and rates are frame-invariant, so they are left as is)
    if anchor_doy is not None:
        for _name in ("sos", "pos", "eos", "msp", "mau", "mos"):
            _j = bands.index(_name)
            if np.isfinite(metrics[_j]):
                metrics[_j] = _unrotate_doy(metrics[_j], anchor_doy)

    return metrics


def _rmse(computed_stack, original_stack, normalized=False):
    # Compute RMSE
    N = len(computed_stack["doy"])
    squared_difference = (original_stack - computed_stack) ** 2
    mean_squared_error = squared_difference.sum("doy", keep_attrs=True, skipna=True) / N
    rmse = mean_squared_error**0.5

    if normalized:
        minn = original_stack.min(dim="doy", skipna=True)
        maxx = original_stack.max(dim="doy", skipna=True)
        return rmse / (maxx - minn)
    else:
        return rmse


def computeChunkSize(arr, sizeMB=100, Z="time"):
    """
    Return a per-dimension chunk dict targeting ~``sizeMB`` per chunk.

    The ``Z`` axis (default ``'time'``) is kept whole and the spatial
    dimensions are split into roughly square tiles to hit the target size.

    :param arr: 3D xarray.DataArray.
    :param sizeMB: approximate desired chunk size in MB.
    :param Z: name of the Z axis. By default, 'time'.
    """
    bmod = arr.dtype.itemsize
    shape = arr.shape
    if len(shape) != 3:
        raise ValueError(f"DataArray dimensions should be 3, not {len(shape)}")
    total_sizeMB = reduce(lambda a, b: a * b, shape) / 1000**2 * bmod
    if total_sizeMB <= sizeMB:
        return dict(zip(arr.dims, shape))
    # keep the Z axis whole, split the spatial dims into ~square tiles
    z_len = arr.sizes[Z]
    spatial_dims = [d for d in arr.dims if d != Z]
    bytes_per_col = bmod * z_len
    target_pixels = max(1, int(sizeMB * 1000**2 / bytes_per_col))
    side = max(1, int(target_pixels**0.5))
    chunk = {Z: z_len}
    for d in spatial_dims:
        chunk[d] = min(arr.sizes[d], side)
    return chunk


def _moving_average(a, n=3, mode="wrap"):
    """Centred moving average of width ``n``, with a real edge neighbourhood.

    ``mode`` is forwarded to :func:`numpy.pad` and decides what the window sees past the
    ends of the array:

    ``"wrap"``
        the series is one closed cycle, so the last step is adjacent to the first. This is
        the correct choice for a phenological year and is the default: ``PhenoShape``
        reconstructs exactly one cycle onto ``nGS`` points.
    ``"reflect"``
        for an open series that is not periodic.
    ``"legacy"``
        the historical behaviour, kept only to reproduce output from before this fix.

    Why the default changed. The historical implementation convolved in ``"valid"`` mode
    and filled the resulting gap by **copying the raw, unsmoothed input** back into the
    first and last ``n // 2`` positions::

        out = np.convolve(a, np.ones(n), "valid") / n
        return np.concatenate([a[: n // 2], out, a[-(n // 2) :]])

    With the default ``rollWindow=5`` that left 4 of 52 steps unsmoothed while the other 48
    were averaged over 5 neighbours -- a different noise level at the two ends of the
    curve, which for a phenological year are the two sides of the *same* boundary. Measured
    on 1,082 Landsat plots in central Chile, the resulting step between DOY 364 and DOY 1
    was ~4x the typical week-to-week change and negative in 67-71% of plots: a systematic
    edge bias, not noise. Any downstream consumer that treats the curve as cyclic -- a
    circular padding in a CNN, an FFT, a harmonic fit -- reads that step as a real event.
    """
    n = int(n)
    a = np.asarray(a, dtype=float)
    if n <= 1:
        return a
    half = n // 2
    if mode == "legacy":
        out = np.convolve(a, np.ones(n), "valid") / n
        return np.concatenate([a[:half], out, a[-half:]])
    padded = np.pad(a, (half, n - 1 - half), mode=mode)
    return np.convolve(padded, np.ones(n), "valid") / n


def _fillNaN(x):
    # Fill NaN data by linear interpolation
    mask = np.isnan(x)
    x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    return x


def _replaceElements(arr):
    """
    Replace monotonic vector values to avoid
    interpolation errors
    """
    s = []
    for i in range(len(arr)):
        # check whether the element
        # is repeated or not
        if arr[i] not in s:
            s.append(arr[i])
        else:
            # find the next greatest element
            for j in range(arr[i] + 1, sys.maxsize):
                if j not in s:
                    arr[i] = j
                    s.append(j)
                    break
