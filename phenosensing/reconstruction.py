"""Reconstruction (smoothing / curve-fitting) methods -- axis 1 of the
composable pipeline.

A *reconstructor* maps an irregular ``(doy, value)`` series onto a regular grid
``xnew`` and returns the reconstructed values. All share the signature::

    reconstruct(x, y, xnew, **params) -> ynew

where ``x`` is the (sorted) day-of-year, ``y`` the vegetation-index values and
``xnew`` the regular output grid.

New methods are registered in :data:`RECONSTRUCTORS`; unknown names fall back to
``scipy.interpolate.interp1d`` with the name used as the spline ``kind`` (for
backwards compatibility with the historical ``interpolType`` argument).

``harmonic`` is the only periodic method: it is the one to use when the output has to
close the year (circular padding downstream, FFT, phase estimation).

Parametric fits (double-logistic, asymmetric Gaussian) are reimplemented from
the primary literature and return an all-NaN curve when the per-pixel fit fails
to converge, so they are safe to map over a whole raster.

References
----------
- Beck et al. (2006), Remote Sensing of Environment 100:321-334.
- Elmore et al. (2012), Global Change Biology 18:656-674.
- Jonsson & Eklundh (2002), IEEE TGRS 40:1824-1832 (asymmetric Gaussian).
- Eilers (2003), Analytical Chemistry 75:3631-3636 (Whittaker smoother).
- Roerink et al. (2000), Int. J. Remote Sensing 21:1911-1917 (HANTS / harmonic).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import Rbf, interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# --------------------------------------------------------------------------- #
# Periodic fits
# --------------------------------------------------------------------------- #


def _harmonic(x, y, xnew, n_harmonics=3, period=365.25, **params):
    """Harmonic (Fourier) regression. Periodic by construction: ``f(t) = f(t + period)``.

    The only reconstructor in the registry that closes the year. Every other method treats
    DOY as an *open* interval, so nothing ties ``f(1)`` to ``f(365)`` and the two ends of
    the fitted curve are free to disagree -- which they do. Measured on 1,082 Landsat plots
    in central Chile with ``interpolType="linear"``, the step between DOY 364 and DOY 1 was
    ~4x the typical week-to-week change and negative in 67-71% of plots.

    That matters for anything downstream that assumes the cycle closes: circular padding in
    a convolutional model, an FFT, a phase estimate, or simply reading the curve as a year.

    The design matrix is ``1, cos(k w t), sin(k w t)`` for ``k = 1..n_harmonics`` with
    ``w = 2 pi / period``. Three harmonics resolve an annual cycle plus a semi-annual and a
    four-monthly component, which covers bimodal and shoulder-season phenologies; more
    harmonics track finer structure at the cost of fitting noise. This is the standard
    device in the satellite-phenology literature (HANTS; the harmonic series of Zhu &
    Woodcock 2014).

    :param n_harmonics: number of sine/cosine pairs. 3 by default.
    :param period: length of the cycle in the units of ``x``. 365.25 days by default.

    References
    ----------
    - Roerink, Menenti & Verhoef (2000), Int. J. Remote Sensing 21:1911-1917 (HANTS).
    - Zhu & Woodcock (2014), Remote Sensing of Environment 152:217-234.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    k = int(n_harmonics)

    def design(t):
        w = 2.0 * np.pi * np.asarray(t, dtype=float) / float(period)
        cols = [np.ones_like(w)]
        for h in range(1, k + 1):
            cols += [np.cos(h * w), np.sin(h * w)]
        return np.column_stack(cols)

    ok = np.isfinite(y)
    # 2k+1 coefficients need at least that many observations; fall back rather than return
    # a rank-deficient fit that looks plausible and is not
    if ok.sum() < 2 * k + 1:
        return np.interp(xnew, x[ok], y[ok]) if ok.any() else np.full(len(xnew), np.nan)
    beta, *_ = np.linalg.lstsq(design(x[ok]), y[ok], rcond=None)
    return design(xnew) @ beta


# --------------------------------------------------------------------------- #
# Interpolators (historical methods)
# --------------------------------------------------------------------------- #


def _linear(x, y, xnew, **params):
    return np.interp(xnew, x, y)


def _rbf(x, y, xnew, function="cubic", **params):
    return Rbf(x, y, function=function)(xnew)


def _kde(x, y, xnew, **params):
    return _KDE(x, y, len(xnew))


# --------------------------------------------------------------------------- #
# Smoothers (operate on a linearly-resampled regular grid)
# --------------------------------------------------------------------------- #


def _odd_window(window_length, n, polyorder):
    """Coerce ``window_length`` to a valid odd Savitzky-Golay window for ``n`` points."""
    wl = int(min(window_length, n if n % 2 else n - 1))
    wl = max(wl, polyorder + 1)
    if wl % 2 == 0:
        wl += 1
    return min(wl, n if n % 2 else n - 1)


def _savgol(x, y, xnew, window_length=7, polyorder=2, **params):
    base = np.interp(xnew, x, y)
    wl = _odd_window(window_length, len(base), polyorder)
    po = min(polyorder, wl - 1)
    return savgol_filter(base, wl, po)


def _whittaker(x, y, xnew, lmbd=10.0, order=2, weights=None, **params):
    from whittaker_eilers import WhittakerSmoother

    if weights is None:
        base = np.interp(xnew, x, y)
        smoother = WhittakerSmoother(lmbda=lmbd, order=order, data_length=len(base))
        return np.asarray(smoother.smooth(base), dtype=float)

    # Weighted: use per-observation weights (e.g. from a QA band). Pooled
    # multi-year input has duplicate DOYs, so collapse each unique DOY to its
    # weighted mean (with total weight) before smoothing the irregular series,
    # then resample to the regular grid.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    ux, inv = np.unique(x, return_inverse=True)
    sw = np.bincount(inv, weights=w, minlength=ux.size)
    swy = np.bincount(inv, weights=w * np.nan_to_num(y), minlength=ux.size)
    uy = np.where(sw > 0, swy / np.where(sw > 0, sw, 1.0), np.interp(ux, x, y))
    smoother = WhittakerSmoother(
        lmbda=lmbd,
        order=order,
        data_length=ux.size,
        x_input=ux.tolist(),
        weights=np.maximum(sw, 1e-6).tolist(),
    )
    smoothed = np.asarray(smoother.smooth(uy.tolist()), dtype=float)
    return np.interp(xnew, ux, smoothed)


def _upper_envelope(x, y, xnew, base="whittaker", n_iter=3, base_params=None, **params):
    """Iterative upper-envelope reconstruction (Chen et al. 2004 / TIMESAT wTSM).

    Fits a base smoother, then repeatedly lifts below-fit samples -- assumed
    cloud-contaminated, since clouds bias optical VIs *downward* -- up to the
    fitted curve and refits, so the result tracks the noise-free upper envelope.
    Needs no QA band; ``base`` is any registered reconstructor and ``base_params``
    its parameters (e.g. ``{"lmbd": 50}``).
    """
    recon = get_reconstructor(base)
    bp = dict(base_params or {})
    grid = np.interp(xnew, x, y)  # work on the regular grid (no duplicate DOYs)
    envelope = grid.copy()
    for _ in range(max(int(n_iter), 1)):
        fit = recon(xnew, envelope, xnew, **bp)
        envelope = np.maximum(grid, fit)  # keep where obs >= fit, else lift to the fit
    return recon(xnew, envelope, xnew, **bp)


# --------------------------------------------------------------------------- #
# Parametric models (fitted per pixel)
# --------------------------------------------------------------------------- #


def _fit(model, x, y, xnew, p0, bounds, maxfev=10000):
    """Fit ``model`` to ``(x, y)`` and evaluate at ``xnew``; NaN on failure."""
    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
        return model(xnew, *popt)
    except Exception:
        return np.full(len(xnew), np.nan)


def _guess(x, y):
    mn, mx = float(np.nanmin(y)), float(np.nanmax(y))
    amp = max(mx - mn, 1e-6)
    half = mn + 0.5 * amp
    above = x[y >= half]
    sos0 = float(above.min()) if above.size else float(x[0])
    eos0 = float(above.max()) if above.size else float(x[-1])
    tpos = float(x[int(np.argmax(y))])
    return mn, mx, amp, sos0, eos0, tpos


def _beck(t, mn, mx, sos, rsp, eos, rau):
    """Beck et al. (2006) double-logistic."""
    return mn + (mx - mn) * (
        1.0 / (1.0 + np.exp(-rsp * (t - sos))) + 1.0 / (1.0 + np.exp(rau * (t - eos))) - 1.0
    )


def _dlog_beck(x, y, xnew, **params):
    mn, mx, amp, sos0, eos0, _ = _guess(x, y)
    p0 = [mn, mx, sos0, 0.1, eos0, 0.1]
    lo = [mn - amp, mn, float(x.min()), 0.0, float(x.min()), 0.0]
    hi = [mx, mx + amp, float(x.max()), 1.0, float(x.max()), 1.0]
    return _fit(_beck, x, y, xnew, p0, (lo, hi))


def _elmore(t, m1, m2, m3, m4, m5, m6, m7):
    """Elmore et al. (2012) seven-parameter double-logistic."""
    return m1 + (m2 - m7 * t) * (
        1.0 / (1.0 + np.exp((m3 - t) / m4)) - 1.0 / (1.0 + np.exp((m5 - t) / m6))
    )


def _dlog_elmore(x, y, xnew, **params):
    mn, mx, amp, sos0, eos0, _ = _guess(x, y)
    p0 = [mn, amp, sos0, 5.0, eos0, 5.0, 0.0]
    lo = [mn - amp, 0.0, float(x.min()), 1e-2, float(x.min()), 1e-2, -abs(amp)]
    hi = [mx, 2 * amp, float(x.max()), 1e3, float(x.max()), 1e3, abs(amp)]
    return _fit(_elmore, x, y, xnew, p0, (lo, hi))


def _asym_gauss(t, base, amp, t0, sig_l, sig_r):
    """Asymmetric Gaussian: different widths on each side of the peak."""
    g = np.where(
        t <= t0,
        np.exp(-(((t - t0) / sig_l) ** 2)),
        np.exp(-(((t - t0) / sig_r) ** 2)),
    )
    return base + amp * g


def _agauss(x, y, xnew, **params):
    mn, mx, amp, _, _, tpos = _guess(x, y)
    span = max(float(x.max() - x.min()), 1.0)
    p0 = [mn, amp, tpos, span / 4, span / 4]
    lo = [mn - amp, 0.0, float(x.min()), 1e-2, 1e-2]
    hi = [mx, 2 * amp, float(x.max()), span, span]
    return _fit(_asym_gauss, x, y, xnew, p0, (lo, hi))


# --------------------------------------------------------------------------- #
# Registry / dispatch
# --------------------------------------------------------------------------- #

RECONSTRUCTORS = {
    "linear": _linear,
    "harmonic": _harmonic,
    "RBF": _rbf,
    "KDE": _kde,
    "savgol": _savgol,
    "whittaker": _whittaker,
    "dlog_beck": _dlog_beck,
    "dlog_elmore": _dlog_elmore,
    "agauss": _agauss,
    "upper_envelope": _upper_envelope,
}


def list_reconstructors() -> list:
    """Return the names of the registered reconstruction methods."""
    return sorted(RECONSTRUCTORS)


def get_reconstructor(name: str):
    """Return the reconstructor callable for ``name``.

    Unknown names fall back to ``scipy.interpolate.interp1d`` using ``name`` as
    the spline ``kind`` (e.g. ``"nearest"``, ``"quadratic"``, ``"cubic"``).
    """
    if name in RECONSTRUCTORS:
        return RECONSTRUCTORS[name]

    def _interp1d_kind(x, y, xnew, _kind=name, **params):
        return interp1d(x, y, kind=_kind)(xnew)

    return _interp1d_kind


def _KDE(x, y, nGS):
    """Bivariate KDE reconstruction using KDEpy (optional dependency)."""
    try:
        from KDEpy import FFTKDE
    except ImportError as exc:
        raise ImportError(
            "KDE reconstruction requires the optional 'KDEpy' package. "
            "Install it with `pip install KDEpy` (or the project's [kde] extra)."
        ) from exc

    grid_points_x, grid_points_y = nGS + 6, 2**8
    data = np.vstack((x, y)).T
    kde = FFTKDE(bw=0.025).fit(data)
    grid, points = kde.evaluate((grid_points_x, grid_points_y))
    x2, y2 = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points_x, grid_points_y)
    y_pred = np.sum((z.T / np.sum(z, axis=1)).T * y2, axis=1)
    idx = np.where(x2 < np.min(x))
    x2 = np.delete(x2, idx)
    y_pred = np.delete(y_pred, idx)
    idx = np.where(x2 > np.max(x))
    y_pred = np.delete(y_pred, idx)
    return y_pred
