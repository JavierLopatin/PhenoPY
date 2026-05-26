"""SOS / EOS extraction methods -- axis 3 of the composable pipeline.

An *extractor* locates the start and end of season on a reconstructed,
peak-normalised curve. All share the signature::

    extract(phen, xnew, ratio, greenup, ipos, **params) -> (sos, eos, isos, ieos)

where ``phen`` is the reconstructed curve, ``xnew`` the DOY grid, ``ratio`` the
curve scaled to [0, 1], ``greenup`` a boolean mask of positive first-derivative
samples and ``ipos`` the index/indices of the seasonal peak. ``sos``/``eos`` are
DOY values and ``isos``/``ieos`` their indices (as length-1 arrays, matching the
historical convention used by the metric calculator).

References
----------
- Threshold (TRS): White et al. (1997); Jonsson & Eklundh (2004).
- Derivative / inflection: Tateishi & Ebata (2004); Gu et al. (2009).
"""

from __future__ import annotations

import numpy as np


def _seasonal_median(phen, xnew, ratio, greenup, ipos, **params):
    """Historical ``phentype=1``: SOS/EOS as the median DOY of the
    greening / senescing phase around the peak."""
    i = np.median(xnew[: ipos[0]][greenup[: ipos[0]]])
    ii = np.median(xnew[ipos[0] :][~greenup[ipos[0] :]])
    sos = xnew[(np.abs(xnew - i)).argmin()]
    eos = xnew[(np.abs(xnew - ii)).argmin()]
    isos = np.where(xnew == int(sos))[0]
    ieos = np.where(xnew == eos)[0]
    return sos, eos, isos, ieos


def _trs(phen, xnew, ratio, greenup, ipos, threshold=0.5, **params):
    """Relative-amplitude threshold: SOS/EOS where the scaled curve crosses
    ``threshold`` (default 0.5) on the rising / falling side of the peak."""
    p = int(ipos[0])
    rise = np.where(ratio[: p + 1] >= threshold)[0]
    isos = int(rise[0]) if rise.size else 0
    fall = np.where(ratio[p:] >= threshold)[0]
    ieos = int(p + fall[-1]) if fall.size else len(xnew) - 1
    return xnew[isos], xnew[ieos], np.array([isos]), np.array([ieos])


def _derivative(phen, xnew, ratio, greenup, ipos, **params):
    """Derivative method: SOS at the maximum greening rate and EOS at the
    maximum senescence rate (extrema of the first derivative)."""
    dev = np.gradient(ratio)
    p = int(ipos[0])
    isos = int(np.argmax(dev[: p + 1])) if p > 0 else 0
    ieos = int(p + np.argmin(dev[p:])) if p < len(xnew) - 1 else len(xnew) - 1
    return xnew[isos], xnew[ieos], np.array([isos]), np.array([ieos])


def _curvature(phen, xnew, ratio, greenup, ipos, **params):
    """Inflection method: SOS/EOS at the maximum curvature (extreme of the
    second derivative) on each side of the peak."""
    curv = np.gradient(np.gradient(ratio))
    p = int(ipos[0])
    isos = int(np.argmax(curv[: p + 1])) if p > 0 else 0
    ieos = int(p + np.argmax(curv[p:])) if p < len(xnew) - 1 else len(xnew) - 1
    return xnew[isos], xnew[ieos], np.array([isos]), np.array([ieos])


EXTRACTORS = {
    "seasonal_median": _seasonal_median,
    "trs": _trs,
    "der": _derivative,
    "curvature": _curvature,
}


def list_extractors() -> list:
    """Return the names of the registered SOS/EOS extraction methods."""
    return sorted(EXTRACTORS)


def get_extractor(name: str):
    """Return the extractor callable for ``name``."""
    try:
        return EXTRACTORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown extraction method {name!r}; choose from {list_extractors()}."
        ) from None
