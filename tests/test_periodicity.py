"""A phenological year has to close.

Every test here guards the same property from a different side: the curve `PhenoShape`
returns describes one cycle, so its last step is adjacent to its first. Nothing in the
library used to enforce that, and the two ends drifted apart in a way that looked like a
real vegetation event to any consumer treating the series as cyclic -- a circular padding,
an FFT, a phase estimate.

The bias these tests pin was measured on 1,082 Landsat plots in central Chile: with
``interpolType="linear"`` and the historical edge handling, the step between DOY 364 and
DOY 1 was ~4x the typical week-to-week change, and negative in 67-71% of plots. Two
independent causes, one test group each.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import phenosensing  # noqa: F401  (registers the accessor)
from phenosensing import _numba
from phenosensing.reconstruction import get_reconstructor, list_reconstructors
from phenosensing.utils import _moving_average


# --------------------------------------------------------------------------- #
# cause 1: the moving average used to leave the ends unsmoothed
# --------------------------------------------------------------------------- #


def test_moving_average_smooths_the_ends_too():
    """The historical implementation copied the raw input into the first and last n//2
    positions, so 4 of 52 steps carried a different noise level from the other 48."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=52)
    out = _moving_average(a, 5)
    assert out.shape == a.shape
    assert not np.allclose(out[:2], a[:2]), "first two steps came through unsmoothed"
    assert not np.allclose(out[-2:], a[-2:]), "last two steps came through unsmoothed"


def test_moving_average_wraps_the_window():
    """With mode='wrap' the window at index 0 reaches back into the tail."""
    a = np.zeros(10)
    a[-1] = 10.0                       # a single spike at the very end
    out = _moving_average(a, 3)
    # a centred width-3 window at index 0 covers indices [-1, 0, 1] -> it must see the spike
    assert out[0] == pytest.approx(10.0 / 3)


def test_moving_average_preserves_the_mean_under_wrap():
    """A circular moving average is mean-preserving; the legacy one is not, because the
    copied endpoints are counted with a different weight."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=52) + 5.0
    assert _moving_average(a, 5).mean() == pytest.approx(a.mean())


def test_legacy_mode_still_reproduces_the_old_output():
    """Kept so results computed before the fix remain reproducible."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=52)
    legacy = _moving_average(a, 5, mode="legacy")
    np.testing.assert_allclose(legacy[:2], a[:2])
    np.testing.assert_allclose(legacy[-2:], a[-2:])


def test_numba_and_numpy_moving_averages_agree():
    """`PhenoShape` picks one or the other depending on whether numba is installed. A
    divergence between them would make results depend on the environment."""
    rng = np.random.default_rng(3)
    for n in (3, 5, 7):
        a = rng.normal(size=52)
        np.testing.assert_allclose(_moving_average(a, n), _numba._mov_avg(a, n))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
def test_moving_average_keeps_length_for_any_window(n):
    a = np.arange(20, dtype=float)
    assert _moving_average(a, n).shape == a.shape


# --------------------------------------------------------------------------- #
# cause 2: no reconstructor was periodic
# --------------------------------------------------------------------------- #


def _irregular_year(seed=0, n_obs=80, amp=0.2, noise=0.05):
    rng = np.random.default_rng(seed)
    doy = np.sort(rng.choice(np.arange(1, 366), n_obs, replace=False)).astype(float)
    y = 0.4 + amp * np.sin(2 * np.pi * (doy - 100) / 365.25) + noise * rng.normal(size=n_obs)
    return doy, y


def test_harmonic_is_registered():
    assert "harmonic" in list_reconstructors()


def test_harmonic_closes_the_year():
    """The property that motivates the method: f(1) == f(366) to within nothing."""
    doy, y = _irregular_year()
    xnew = np.linspace(1, 365, 52)
    f = get_reconstructor("harmonic")(doy, y, xnew, n_harmonics=3)
    step = np.median(np.abs(np.diff(f)))
    assert abs(f[-1] - f[0]) < 0.1 * step


def test_harmonic_recovers_a_known_harmonic_signal():
    """With the signal in the model's span, the fit has to be near-exact."""
    doy = np.arange(1, 366, 4, dtype=float)
    truth = 0.5 + 0.3 * np.cos(2 * np.pi * doy / 365.25) + 0.1 * np.sin(4 * np.pi * doy / 365.25)
    out = get_reconstructor("harmonic")(doy, truth, doy, n_harmonics=3)
    np.testing.assert_allclose(out, truth, atol=1e-6)


def test_harmonic_falls_back_rather_than_returning_a_rank_deficient_fit():
    """2k+1 coefficients need at least that many observations. Returning a fit from fewer
    would look plausible and be meaningless."""
    doy = np.array([10.0, 100.0, 200.0])
    y = np.array([0.2, 0.6, 0.3])
    out = get_reconstructor("harmonic")(doy, y, np.linspace(1, 365, 20), n_harmonics=3)
    assert out.shape == (20,)
    assert np.all(np.isfinite(out))


def test_n_harmonics_controls_what_the_fit_can_represent():
    """A signal made of harmonics 1 and 3: k<3 cannot represent it, k>=3 fits it exactly."""
    doy, y = _irregular_year(seed=5, noise=0.0, amp=0.25)
    y = y + 0.08 * np.sin(6 * np.pi * doy / 365.25)      # a three-per-year component
    err = {k: float(np.mean((get_reconstructor("harmonic")(doy, y, doy, n_harmonics=k) - y) ** 2))
           for k in (1, 2, 3, 6)}
    assert err[2] < err[1]                    # harmonic 2 helps a little
    assert err[3] < 1e-20                     # harmonic 3 completes the span: exact
    assert err[3] < 1e-3 * err[2]             # and the drop is not marginal
    assert err[6] < 1e-20                     # more harmonics cannot hurt an exact fit


# --------------------------------------------------------------------------- #
# the property end to end, through PhenoShape
# --------------------------------------------------------------------------- #


def _cube(ny=3, nx=3, n=70, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2019-01-01", periods=n, freq="5D")
    doy = times.dayofyear.values
    season = 0.35 + 0.25 * np.sin((doy / 365.25) * 2 * np.pi - np.pi / 2)
    data = np.broadcast_to(season[:, None, None], (n, ny, nx)).copy()
    data = data + 0.03 * rng.normal(size=data.shape)
    da = xr.DataArray(
        data, dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(ny), "x": np.arange(nx)})
    return da.assign_coords(doy=("time", doy), year=("time", times.year.values))


@pytest.mark.parametrize("recon", ["linear", "harmonic"])
def test_phenoshape_curve_closes_the_year(recon):
    """The regression guard the library was missing.

    For every reconstructed pixel, the wrap-around step may not exceed the typical step
    inside the curve. This is the check that would have caught the original bug.
    """
    out = _cube().pheno.PhenoShape(interpolType=recon, rollWindow=5, nGS=52).values
    flat = out.reshape(out.shape[0], -1).T
    wrap = np.abs(flat[:, 0] - flat[:, -1])
    typical = np.median(np.abs(np.diff(flat, axis=1)), axis=1)
    assert np.all(wrap <= 3.0 * typical), (
        f"{int((wrap > 3.0 * typical).sum())} of {len(wrap)} pixels have a wrap-around step "
        f"more than 3x the typical step (max ratio {np.max(wrap / typical):.1f}x)")


def test_the_fix_actually_reduces_the_wrap_step(monkeypatch):
    """Directly contrasts the two edge modes on the same data, so the improvement is
    attributable to the fix and not to the test data being easy.

    Numba has to be disabled first. When it is available `PhenoShape` runs `_numba._mov_avg`
    and never touches `utils._moving_average`, so patching only the latter silently compares
    the fixed path against itself -- which is exactly what this test did on the first
    attempt, and it is the same trap that would have let a fix land in one path only.
    """
    from phenosensing import utils

    monkeypatch.setattr(_numba, "NUMBA_AVAILABLE", False)
    da = _cube(seed=7)
    fixed = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52).values

    orig = utils._moving_average
    try:
        utils._moving_average = lambda a, n=3, mode="wrap": orig(a, n, mode="legacy")
        legacy = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52).values
    finally:
        utils._moving_average = orig

    def wrap_step(arr):
        f = arr.reshape(arr.shape[0], -1).T
        return float(np.median(np.abs(f[:, 0] - f[:, -1])))

    assert wrap_step(fixed) < wrap_step(legacy)
