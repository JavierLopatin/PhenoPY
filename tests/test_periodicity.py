"""The ends of a phenological curve: reproducible, evenly smoothed, and free to differ.

An earlier version of this file asserted that the curve must *close* -- that ``f(1)`` should
equal ``f(365)``. That was wrong, and the correction is the point of these tests. When
`PhenoShape` fits a composite over several years, the end of the composite joins the start
of the *next* year, not its own. If productivity changed between years the two ends
genuinely differ, and that difference is signal.

What was actually broken was three separate things, measured on 1,082 Landsat plots in
central Chile:

1. **Order dependence.** `_getPheno0` sorted by DOY with an unstable quicksort, so
   observations sharing a DOY were ordered arbitrarily -- and `_fillNaN` interpolates over
   array positions, so a different tie order produced a different curve. All 690 plots with
   no tied DOY reproduced bit for bit across machines; 375 of the 392 that do have ties did
   not, median discrepancy 0.068. Ties affect 36% of plots.
2. **Uneven smoothing.** `_moving_average` left the first and last ``n // 2`` steps raw
   while the rest were averaged over ``n`` neighbours -- 4 of 52 steps with a different
   noise level, at the two sides of the year boundary.
3. **A bad remedy.** Closing the year (with ``mode="wrap"`` or the ``harmonic``
   reconstructor) does remove the artefact, and also removes the real interannual signal.
   The tests below pin that the signal survives.
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


def test_moving_average_wraps_the_window_when_asked():
    """`wrap` is still available for a genuine single cycle: the window at index 0 then
    reaches back into the tail. It is no longer the default -- see the module docstring."""
    a = np.zeros(10)
    a[-1] = 10.0                       # a single spike at the very end
    out = _moving_average(a, 3, mode="wrap")
    # a centred width-3 window at index 0 covers indices [-1, 0, 1] -> it must see the spike
    assert out[0] == pytest.approx(10.0 / 3)


def test_moving_average_preserves_the_mean_under_wrap():
    """A circular moving average is mean-preserving; the legacy one is not, because the
    copied endpoints are counted with a different weight."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=52) + 5.0
    assert _moving_average(a, 5, mode="wrap").mean() == pytest.approx(a.mean())


def test_shrink_touches_only_the_ends():
    """The property that makes this a surgical fix rather than a new curve.

    Wherever the full window fits -- steps ``half`` to ``n - half - 1`` -- `shrink` is the
    same "valid" convolution the library always did. Only the ends change.
    """
    rng = np.random.default_rng(7)
    for n in (3, 5, 7):
        half = n // 2
        a = rng.normal(size=52)
        legacy = _moving_average(a, n, mode="legacy")
        shrink = _moving_average(a, n, mode="shrink")
        np.testing.assert_allclose(legacy[half:-half], shrink[half:-half], atol=1e-12)
        differ = np.flatnonzero(np.abs(legacy - shrink) > 1e-12)
        assert set(differ) <= set(list(range(half)) + list(range(52 - half, 52)))


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


def test_phenoshape_is_invariant_to_the_order_of_the_observations():
    """The reproducibility guard the library was missing.

    Feeding the same observations in a different order must give the same curve. It did not:
    with tied DOYs the unstable sort picked an arbitrary order and `_fillNaN`, which
    interpolates over array positions, filled the gaps differently.
    """
    rng = np.random.default_rng(0)
    da = _cube(seed=3)
    ref = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52).values
    for _ in range(5):
        p = rng.permutation(da.sizes["time"])
        shuffled = da.isel(time=p)
        got = shuffled.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52).values
        np.testing.assert_allclose(got, ref, atol=1e-12, equal_nan=True)


def test_tied_doys_do_not_change_the_curve():
    """The same property where it actually bites: duplicated days of year.

    Three years of a 16-day revisit put ties in 36% of real plots, and every plot that
    failed to reproduce across machines had them.
    """
    da = _cube(seed=4, n=60)
    doy = da["doy"].values.copy()
    doy[10] = doy[3]                       # force a tie
    doy[25] = doy[7]
    tied = da.assign_coords(doy=("time", doy))
    a = tied.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52).values
    order = np.argsort(-np.arange(tied.sizes["time"]))     # reverse the input order
    b = tied.isel(time=order).pheno.PhenoShape(
        interpolType="linear", rollWindow=5, nGS=52).values
    np.testing.assert_allclose(a, b, atol=1e-12, equal_nan=True)


def test_the_interannual_trend_survives_the_smoothing():
    """A composite must NOT be forced to close.

    With a genuine interannual trend injected, the year-boundary step has to reflect it.
    The compositing predicts step ~ -slope: DOY 1 averages observations about a year earlier
    in real time than DOY 364. A reconstructor or an edge mode that flattens this is
    deleting information, not cleaning it.
    """
    from phenosensing.utils import _getPheno, _moving_average

    rng = np.random.default_rng(11)
    doy = np.sort(rng.choice(np.arange(1, 366), 90, replace=False)).astype(int)
    t = np.tile(np.arange(3), 30)[: len(doy)]              # three years
    steps = []
    for slope in (-0.10, 0.0, 0.10):
        y = 0.4 + 0.15 * np.sin(2 * np.pi * (doy - 100) / 365.25) + slope * t
        i = np.lexsort((y, doy))
        c = _moving_average(_getPheno(y[i].copy(), doy[i], 52, "linear"), 5, mode="shrink")
        steps.append(c[0] - c[-1])
    # monotone in the slope, and the sign is the one the compositing predicts
    assert steps[0] > steps[1] > steps[2], f"the trend did not reach the boundary: {steps}"


def test_the_ends_are_smoothed_as_much_as_the_middle(monkeypatch):
    """The actual goal: the tails should be as smoothed as the centre -- not forced to meet.

    Legacy left the first and last ``n // 2`` steps raw, so they carried the full
    observation noise while their neighbours carried a fifth of it. Roughness here is the
    mean absolute second difference, which measures exactly that.

    Numba has to be disabled first. When it is available `PhenoShape` runs `_numba._mov_avg`
    and never touches `utils._moving_average`, so patching only the latter silently compares
    the fixed path against itself -- which is what this test did on its first version, and
    the same trap that would let a fix land in one path only.
    """
    from phenosensing import utils

    monkeypatch.setattr(_numba, "NUMBA_AVAILABLE", False)
    da = _cube(seed=7)

    def edge_vs_middle(arr):
        f = arr.reshape(arr.shape[0], -1).T
        r = np.abs(np.diff(f, n=2, axis=1))          # local roughness per step
        return float(np.median(r[:, [0, 1, -2, -1]])) / float(np.median(r[:, 2:-2]))

    fixed = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52).values
    orig = utils._moving_average
    try:
        utils._moving_average = lambda a, n=3, mode="shrink": orig(a, n, mode="legacy")
        legacy = da.pheno.PhenoShape(interpolType="linear", rollWindow=5, nGS=52).values
    finally:
        utils._moving_average = orig

    assert edge_vs_middle(legacy) > 2.0, "the legacy ends should be visibly rougher"
    assert edge_vs_middle(fixed) < edge_vs_middle(legacy)
