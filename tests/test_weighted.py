"""Weighted reconstruction: QA-weighted Whittaker + upper-envelope (wTSM/Chen).

Uses a synthetic clean season with injected *downward* "cloud" contamination
(the asymmetric noise QA weighting is meant to fix), so it needs no Earth Engine.
"""

import numpy as np
import xarray as xr

import phenopy  # noqa: F401  (registers the .pheno accessor)
from phenopy.reconstruction import get_reconstructor, list_reconstructors


def _truth(doy):
    """A clean single-season bump in [0.2, 0.9]."""
    doy = np.asarray(doy, dtype=float)
    return 0.2 + 0.7 * np.exp(-(((doy - 200.0) / 45.0) ** 2))


def _contaminated(seed=0, n_years=5, frac=0.4):
    """Pooled multi-year series (duplicate DOYs); a fraction of points are pulled
    downward (clouds) and flagged with weight 0."""
    rng = np.random.default_rng(seed)
    doy = np.sort(np.tile(np.arange(8, 361, 8, dtype=float), n_years))
    y = _truth(doy)
    w = np.ones_like(y)
    mask = rng.random(y.size) < frac
    y = y.copy()
    y[mask] -= 0.3 + 0.4 * rng.random(int(mask.sum()))  # clouds bias NDVI down
    w[mask] = 0.0
    return doy, y, w


def _rmse(a, b):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


XNEW = np.arange(8, 361, 4, dtype=float)
TRUTH = _truth(XNEW)


def test_upper_envelope_is_registered():
    assert "upper_envelope" in list_reconstructors()


def test_upper_envelope_recovers_from_downward_noise():
    doy, y, _ = _contaminated()
    env = get_reconstructor("upper_envelope")(
        doy, y, XNEW, base="whittaker", n_iter=5, base_params={"lmbd": 50}
    )
    lin = get_reconstructor("linear")(doy, y, XNEW)
    assert _rmse(env, TRUTH) < _rmse(lin, TRUTH)


def test_weighted_whittaker_beats_unweighted():
    doy, y, w = _contaminated()
    wht = get_reconstructor("whittaker")
    weighted = wht(doy, y, XNEW, lmbd=50, weights=w)
    plain = wht(doy, y, XNEW, lmbd=50)
    assert _rmse(weighted, TRUTH) < _rmse(plain, TRUTH)


def _cube(doy, y, w, ny=2, nx=2):
    da = xr.DataArray(
        np.broadcast_to(y[:, None, None], (y.size, ny, nx)).astype(float),
        dims=["time", "y", "x"],
        coords={"doy": ("time", doy), "year": ("time", np.zeros(doy.size, int))},
    )
    wda = xr.DataArray(
        np.broadcast_to(w[:, None, None], (w.size, ny, nx)).astype(float),
        dims=["time", "y", "x"],
        coords={"doy": ("time", doy)},
    )
    return da, wda


def test_phenoshape_weights_runs_and_improves_fit():
    doy, y, w = _contaminated()
    da, wda = _cube(doy, y, w)
    sw = da.pheno.PhenoShape(
        interpolType="whittaker", recon_params={"lmbd": 50}, rollWindow=None, weights=wda
    )
    su = da.pheno.PhenoShape(interpolType="whittaker", recon_params={"lmbd": 50}, rollWindow=None)
    assert dict(sw.sizes)["doy"] == 52
    assert not np.allclose(sw.values, su.values)  # weighting changed the result
    truth = _truth(sw["doy"].values)
    assert _rmse(sw.isel(y=0, x=0).values, truth) < _rmse(su.isel(y=0, x=0).values, truth)


def test_phenoshape_default_byte_identical_to_weights_none():
    doy, y, w = _contaminated()
    da, _ = _cube(doy, y, w)
    xr.testing.assert_identical(da.pheno.PhenoShape(), da.pheno.PhenoShape(weights=None))
