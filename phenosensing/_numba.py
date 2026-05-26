"""Optional Numba-accelerated kernels (the ``[fast]`` extra).

These provide a GIL-releasing, parallel fast path for the common *linear*
reconstruction. PhenoShape uses them automatically when Numba is installed and
the inputs are provably equivalent to the pure-Python path (linear method,
in-memory array, no NaNs, no custom params); otherwise it falls back to the
pure-Python implementation, so results are identical either way.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the [fast] extra
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        # Support both @njit and @njit(...) when numba is absent.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(func):
            return func

        return deco


@njit(cache=True)
def _mov_avg(a, n):
    """Moving average matching utils._moving_average for odd ``n``."""
    m = a.shape[0]
    half = n // 2
    valid_len = m - n + 1
    out = np.empty(m, dtype=np.float64)
    for k in range(half):
        out[k] = a[k]
    for k in range(valid_len):
        s = 0.0
        for w in range(n):
            s += a[k + w]
        out[half + k] = s / n
    for k in range(half):
        out[half + valid_len + k] = a[m - half + k]
    return out


@njit(parallel=True, cache=True)
def linear_phenoshape(arr, idx, doy, xnew, roll_window):
    """Parallel linear reconstruction of a ``(time, y, x)`` cube.

    Mirrors the pure-Python linear path (reorder by the precomputed DOY sort
    ``idx``, ``np.interp`` onto ``xnew``, optional moving average) but loops
    over pixels in compiled, GIL-released code. ``idx`` is computed by NumPy
    so the tie-ordering of duplicate DOYs matches the pure-Python path exactly.
    """
    T = arr.shape[0]
    Y = arr.shape[1]
    X = arr.shape[2]
    nGS = xnew.shape[0]
    doy_s = np.empty(T, dtype=np.float64)
    for t in range(T):
        doy_s[t] = doy[idx[t]]
    out = np.empty((nGS, Y, X), dtype=np.float64)
    for j in prange(Y):
        for k in range(X):
            y_s = np.empty(T, dtype=np.float64)
            for t in range(T):
                y_s[t] = arr[idx[t], j, k]
            interp = np.interp(xnew, doy_s, y_s)
            if roll_window >= 2:
                interp = _mov_avg(interp, roll_window)
            for g in range(nGS):
                out[g, j, k] = interp[g]
    return out
