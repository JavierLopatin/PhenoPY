"""Per-metric uncertainty via bootstrap resampling.

For each pixel, the observations are resampled with replacement ``n_boot``
times; the 18 land-surface-phenology metrics are recomputed for every replicate
and their spread (standard deviation) is returned as the per-metric
uncertainty. This is a non-parametric estimate of the sampling uncertainty of
SOS/POS/EOS and the other metrics.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .utils import LSP_BANDS, _getLSPmetrics2, _getPheno0


def _bootstrap_lsp_std(
    vi,
    doy,
    n_boot,
    interpolType,
    nan_replace,
    rollWindow,
    nGS,
    phentype,
    extraction,
    extract_params,
    seed,
    hemisphere,
):
    n_bands = len(LSP_BANDS)
    if np.isfinite(vi).sum() < 3:
        return np.full(n_bands, np.nan)
    rng = np.random.default_rng(seed)
    T = vi.shape[0]
    boot = np.full((n_boot, n_bands), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, T, T)
        d = doy[idx]
        phen = _getPheno0(vi[idx], d, interpolType, nan_replace, rollWindow, nGS)
        xnew = np.linspace(np.min(d), np.max(d), nGS, dtype=np.int32)
        boot[b] = _getLSPmetrics2(
            phen, xnew, nGS, LSP_BANDS, phentype, extraction, extract_params, hemisphere=hemisphere
        )
    return np.nanstd(boot, axis=0)


def uncertainty(
    da: xr.DataArray,
    n_boot: int = 50,
    interpolType: str = "linear",
    nan_replace: float | None = None,
    rollWindow: int = 5,
    nGS: int = 52,
    phentype: int = 1,
    extraction: str | None = None,
    extract_params: dict | None = None,
    seed: int = 0,
    hemisphere: str = "north",
) -> xr.Dataset:
    """Per-pixel bootstrap uncertainty (std) of each land-surface-phenology metric.

    Parameters
    ----------
    da:
        ``(time, y, x)`` vegetation-index cube with a ``doy`` coordinate.
    n_boot:
        Number of bootstrap replicates (resamples of the observations).
    interpolType, nan_replace, rollWindow, nGS, phentype, extraction,
    extract_params, hemisphere:
        Passed through to the reconstruction/extraction for each replicate
        (``hemisphere="auto"`` anchors each pixel's phase before extraction; see
        :func:`phenosensing.phase.season_phase`).
    seed:
        Seed for the per-pixel resampling (reproducible).

    Returns
    -------
    xarray.Dataset
        The bootstrap standard deviation of each of the 18 metrics, dims
        ``(y, x)`` per variable.
    """
    doy = np.asarray(da["doy"].values)
    out = xr.apply_ufunc(
        _bootstrap_lsp_std,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["LSP_bands"]],
        exclude_dims={"time"},
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"LSP_bands": len(LSP_BANDS)}},
        output_dtypes=[float],
        kwargs={
            "doy": doy,
            "n_boot": n_boot,
            "interpolType": interpolType,
            "nan_replace": nan_replace,
            "rollWindow": rollWindow,
            "nGS": nGS,
            "phentype": phentype,
            "extraction": extraction,
            "extract_params": extract_params,
            "seed": seed,
            "hemisphere": hemisphere,
        },
    )
    return out.assign_coords(LSP_bands=list(LSP_BANDS)).to_dataset("LSP_bands")
