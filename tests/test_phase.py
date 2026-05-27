"""Tests for per-pixel seasonal-phase detection and anchoring (``hemisphere=``).

The headline property is that no single *global* hemisphere flag works for a
heterogeneous scene: ``"north"`` breaks austral-summer pixels (the season wraps
the 1-Jan cut) and ``"south"`` breaks austral-winter pixels, while the per-pixel
``"auto"`` mode recovers a plausible season in every regime. We use the season
*length* (LOS) as the discriminator because it is an accuracy signal — bootstrap
uncertainty is not, since a mis-centered extraction tends to be *stably* wrong.
"""

import numpy as np
import pytest
import xarray as xr

import phenosensing  # noqa: F401  (importing registers the `pheno` accessor)
from phenosensing import season_phase
from phenosensing.io import load_sample
from phenosensing.phase import _unrotate_doy


def _circular_gauss(doy, peak, sigma=40.0, base=0.2, amp=0.6):
    """A seasonal bump on the DOY circle (so it can wrap the year boundary)."""
    d = np.abs(np.asarray(doy, dtype=float) - peak)
    d = np.minimum(d, 365.0 - d)
    return base + amp * np.exp(-((d / sigma) ** 2))


def _shape(*curves, n=120):
    """Stack 1-D curves into a ``(doy, y=1, x=len(curves))`` PhenoShape array."""
    x = np.linspace(1, 365, n).astype(np.int32)
    data = np.stack([np.asarray(c, dtype=float) for c in curves], axis=-1)[:, None, :]
    return xr.DataArray(data, dims=("doy", "y", "x"), coords={"doy": x})


def _circ_dist(a, b):
    d = abs(a - b) % 365
    return min(d, 365 - d)


def test_north_is_byte_identical_to_default():
    da = load_sample("SIF")
    shape = da.pheno.PhenoShape(rollWindow=5, nGS=52)
    default = shape.pheno.PhenoLSP().compute()
    north = shape.pheno.PhenoLSP(hemisphere="north").compute()
    for band in default.data_vars:
        np.testing.assert_array_equal(north[band].values, default[band].values)


def test_invalid_hemisphere_raises():
    shape = _shape(_circular_gauss(np.linspace(1, 365, 120), 180))
    with pytest.raises(ValueError, match="hemisphere"):
        shape.pheno.PhenoLSP(hemisphere="east")


def test_wrapping_summer_recovered_by_auto():
    x = np.linspace(1, 365, 120)
    shape = _shape(_circular_gauss(x, peak=15, sigma=35))
    north = shape.pheno.PhenoLSP(hemisphere="north", extraction="trs")
    auto = shape.pheno.PhenoLSP(hemisphere="auto", extraction="trs")

    # north cannot represent the wrap: it sees a year-long, edge-pinned "season"
    assert float(north.los.isel(y=0, x=0)) > 300
    # auto centers the pixel and recovers a compact austral-summer season
    pos_a = float(auto.pos.isel(y=0, x=0))
    los_a = float(auto.los.isel(y=0, x=0))
    sos_a = float(auto.sos.isel(y=0, x=0))
    eos_a = float(auto.eos.isel(y=0, x=0))
    assert _circ_dist(pos_a, 15) < 25  # peak recovered (circularly) near DOY 15
    assert 30 < los_a < 150  # compact, plausible season length
    assert sos_a > 300  # season starts the previous December
    assert eos_a < 90  # ...and ends in late summer


def test_aseasonal_pixel_flagged_and_left_unrotated():
    rng = np.random.default_rng(0)
    flat = 0.5 + 0.01 * rng.standard_normal(120)
    shape = _shape(flat)
    sp = season_phase(shape)
    assert bool(sp["aseasonal"].isel(y=0, x=0))  # weak annual cycle -> aseasonal
    # an aseasonal pixel is not rotated, so "auto" == "north" exactly
    auto = shape.pheno.PhenoLSP(hemisphere="auto")
    north = shape.pheno.PhenoLSP(hemisphere="north")
    for band in north.data_vars:
        np.testing.assert_allclose(auto[band].values, north[band].values, equal_nan=True)


def test_multi_season_pixel_flagged():
    x = np.linspace(1, 365, 120)
    bimodal = (
        0.2 + 0.5 * np.exp(-(((x - 100) / 25) ** 2)) + 0.5 * np.exp(-(((x - 260) / 25) ** 2))
    )
    auto = _shape(bimodal).pheno.PhenoLSP(hemisphere="auto")
    assert bool(auto.coords["multi_season"].isel(y=0, x=0))


def test_auto_matches_north_on_centered_season():
    x = np.linspace(1, 365, 120)
    shape = _shape(_circular_gauss(x, peak=180, sigma=40))
    north = shape.pheno.PhenoLSP(hemisphere="north")
    auto = shape.pheno.PhenoLSP(hemisphere="auto")
    for band in ("sos", "pos", "eos", "los"):
        np.testing.assert_allclose(
            float(auto[band].isel(y=0, x=0)),
            float(north[band].isel(y=0, x=0)),
            atol=8,  # within ~1 DOY grid step (the anchor snaps to the grid)
        )


def test_no_single_global_hemisphere_wins_every_regime():
    # one scene, three regimes: austral summer (wraps the 1-Jan cut), austral
    # winter (~DOY 190) and spring. A global flag fixes one and breaks another;
    # per-pixel "auto" gives a plausible season length for all three.
    x = np.linspace(1, 365, 120)
    shape = _shape(
        _circular_gauss(x, 15),  # summer -> "north" breaks this
        _circular_gauss(x, 190),  # winter -> "south" breaks this
        _circular_gauss(x, 250),  # spring -> both global flags are fine
    )
    los = {
        h: shape.pheno.PhenoLSP(hemisphere=h, extraction="trs").los.isel(y=0)
        for h in ("north", "south", "auto")
    }
    summer, winter, spring = 0, 1, 2

    assert float(los["north"].isel(x=summer)) > 300  # north breaks austral summer
    assert float(los["south"].isel(x=winter)) > 300  # south breaks austral winter
    # auto yields a plausible (~2-month) season for every regime
    for px in (summer, winter, spring):
        assert 30 < float(los["auto"].isel(x=px)) < 150


def test_unrotate_doy_is_circular():
    # rotated day 1 maps back to the anchor (start of the phenological year)
    assert _unrotate_doy(1.0, 183.0) == pytest.approx(183.0)
    # a day deep in the phenological year wraps past New Year
    assert _unrotate_doy(250.0, 183.0) == pytest.approx(67.0)


def test_season_phase_structure_and_accepts_both_layouts():
    da = load_sample("SIF").isel(y=slice(0, 4), x=slice(0, 4))
    sp = season_phase(da)  # (time, y, x) with a doy coordinate
    assert set(sp.data_vars) == {"peak_doy", "anchor", "strength", "aseasonal"}
    assert set(sp["anchor"].dims) == {"y", "x"}

    shape = da.pheno.PhenoShape(rollWindow=5, nGS=40)
    sp2 = season_phase(shape)  # (doy, y, x)
    assert set(sp2["anchor"].dims) == {"y", "x"}
