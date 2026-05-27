"""Per-pixel seasonal-phase detection and anchoring.

Land-surface-phenology extraction assumes a single growing-season bump that sits
*centered* inside the day-of-year window [1, 365]. The real failure mode is not
"hemisphere" but the **phase of each pixel's season relative to the 1-Jan cut**:
when a season wraps the year boundary (e.g. austral-summer vegetation peaking in
January), the pooled climatology becomes a U-shape and SOS/EOS/LOS come out
wrong. A single global Southern-Hemisphere reorder cannot serve heterogeneous
scenes (e.g. Chile holds both winter-peaking Mediterranean shrubland and
summer-peaking vegetation in one image).

This module estimates each pixel's seasonal phase from the **first annual
Fourier harmonic** and derives a per-pixel *anchor* (the dormancy DOY) that
defines a centered "phenological-year" frame. ``PhenoLSP(hemisphere="auto")``
rotates each pixel into that frame, extracts the metrics there, and maps the
DOY-valued ones back to calendar DOY. The estimator is one small least-squares
per pixel, so it vectorises through ``xarray.apply_ufunc``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["season_phase"]

# Seasonality strength below which a pixel is treated as aseasonal (no rotation).
# The first-harmonic amplitude is taken relative to the series mean; 0.15 means
# the annual cycle must be at least ~15% of the mean. Tunable per call.
DEFAULT_STRENGTH_THRESH = 0.15


def _harmonic_phase(x, y):
    """Phase and strength of the first annual harmonic of a per-pixel series.

    Fits ``a0 + a1*cos(2*pi*d/365) + b1*sin(2*pi*d/365)`` by least squares and
    returns ``(peak_doy, strength)`` where ``peak_doy`` is the day-of-year of the
    harmonic maximum and ``strength = sqrt(a1**2 + b1**2) / |a0|`` measures how
    seasonal the pixel is. Returns ``(nan, nan)`` when fewer than 3 finite
    samples are available, matching the rest of the kernel.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    xx = x[mask]
    yy = y[mask]
    w = 2.0 * np.pi * xx / 365.0
    design = np.column_stack((np.ones_like(xx), np.cos(w), np.sin(w)))
    a0, a1, b1 = np.linalg.lstsq(design, yy, rcond=None)[0]
    # a1*cos(w) + b1*sin(w) = R*cos(w - phi); maximum at w = phi = atan2(b1, a1)
    peak = float((np.arctan2(b1, a1) * 365.0 / (2.0 * np.pi)) % 365.0)
    amp = float(np.hypot(a1, b1))
    strength = amp / max(abs(float(a0)), 1e-9)
    return peak, strength


def _anchor_from_peak(peak_doy):
    """Dormancy DOY (start of the phenological year): half a year from the peak."""
    return (peak_doy + 182.5) % 365.0


def _unrotate_doy(r, anchor):
    """Map a DOY ``r`` from the rotated phenological-year frame (which starts at
    ``anchor``) back to a calendar DOY in [1, 365]."""
    return ((anchor - 1.0) + (r - 1.0)) % 365.0 + 1.0


def resolve_anchor(phen, xnew, hemisphere, offset=None, strength_thresh=DEFAULT_STRENGTH_THRESH):
    """Per-pixel anchor (dormancy DOY) for the rotation, or ``None`` to skip it.

    - an explicit ``offset`` wins (used by tests and the future weighted path);
    - ``"south"`` → a fixed austral anchor (DOY 183, ~1 July);
    - ``"auto"`` → harmonic peak + half a year, unless the pixel is aseasonal
      (``strength < strength_thresh``), in which case ``None`` (no rotation).

    ``"north"`` is handled by the caller (no rotation) and never reaches here.
    """
    if offset is not None:
        return float(offset)
    if hemisphere == "south":
        return 183.0
    if hemisphere == "auto":
        peak, strength = _harmonic_phase(xnew, phen)
        if not np.isfinite(peak) or strength < strength_thresh:
            return None
        return _anchor_from_peak(peak)
    raise ValueError(f"hemisphere must be 'north', 'south' or 'auto'; got {hemisphere!r}.")


def _season_phase_1d(y, x, strength_thresh):
    peak, strength = _harmonic_phase(x, y)
    anchor = _anchor_from_peak(peak) if np.isfinite(peak) else np.nan
    aseasonal = (not np.isfinite(strength)) or (strength < strength_thresh)
    return np.array([peak, anchor, strength, float(aseasonal)], dtype=float)


def season_phase(da: xr.DataArray, strength_thresh: float = DEFAULT_STRENGTH_THRESH) -> xr.Dataset:
    """Per-pixel seasonal phase from the first annual harmonic.

    Parameters
    ----------
    da:
        Either a ``(time, y, x)`` vegetation-index cube with a ``doy`` coordinate,
        or a :meth:`~phenosensing.accessor.Pheno.PhenoShape` output with a ``doy``
        dimension.
    strength_thresh:
        Seasonality strength below which a pixel is flagged ``aseasonal``
        (default :data:`DEFAULT_STRENGTH_THRESH`).

    Returns
    -------
    xarray.Dataset
        Per-pixel ``(y, x)`` variables:

        - ``peak_doy`` — day-of-year of the seasonal peak,
        - ``anchor`` — dormancy DOY (= ``peak_doy + 182.5``), the start of the
          phenological year and the rotation used by ``PhenoLSP(hemisphere="auto")``,
        - ``strength`` — first-harmonic amplitude relative to the mean,
        - ``aseasonal`` — ``1`` where ``strength < strength_thresh``.

    Mapping ``anchor`` (or ``peak_doy``) shows *when* each pixel's growing season
    occurs, independent of any global hemisphere assumption.
    """
    if "doy" in da.dims:
        core = "doy"
    elif "doy" in da.coords and "time" in da.dims:
        core = "time"
    else:
        raise ValueError("season_phase needs a 'doy' dimension or a 'doy' coordinate on 'time'.")

    x = np.asarray(da["doy"].values, dtype=float)
    out = xr.apply_ufunc(
        _season_phase_1d,
        da,
        input_core_dims=[[core]],
        output_core_dims=[["phase_var"]],
        exclude_dims={core},
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"phase_var": 4}},
        output_dtypes=[float],
        kwargs={"x": x, "strength_thresh": strength_thresh},
    )
    out = out.assign_coords(phase_var=["peak_doy", "anchor", "strength", "aseasonal"])
    return out.to_dataset("phase_var")
