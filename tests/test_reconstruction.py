"""Unit tests for the reconstruction registry (axis 1)."""

import numpy as np

from phenopy.reconstruction import _beck, get_reconstructor, list_reconstructors


def test_registry_lists_expected_methods():
    names = list_reconstructors()
    for m in ["linear", "RBF", "KDE", "savgol", "whittaker", "dlog_beck", "dlog_elmore", "agauss"]:
        assert m in names


def test_linear_matches_numpy_interp():
    x = np.array([1.0, 10, 20, 40, 60])
    y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
    xnew = np.linspace(1, 60, 30)
    out = get_reconstructor("linear")(x, y, xnew)
    np.testing.assert_allclose(out, np.interp(xnew, x, y))


def test_unknown_name_falls_back_to_interp1d_kind():
    x = np.array([1.0, 10, 20, 40, 60])
    y = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
    xnew = np.linspace(1, 60, 10)
    out = get_reconstructor("nearest")(x, y, xnew)
    assert out.shape == xnew.shape
    assert np.all(np.isfinite(out))


def test_savgol_reduces_noise():
    rng = np.random.default_rng(0)
    x = np.arange(1.0, 101.0)
    clean = 0.5 + 0.4 * np.sin((x / 100) * 2 * np.pi - np.pi / 2)
    y = clean + rng.normal(0, 0.05, x.size)
    out = get_reconstructor("savgol")(x, y, x, window_length=11, polyorder=2)
    assert np.var(np.diff(out)) < np.var(np.diff(y))


def test_dlog_beck_recovers_known_curve():
    t = np.linspace(1, 365, 60)
    truth = _beck(t, 0.1, 0.8, 120, 0.1, 270, 0.1)
    out = get_reconstructor("dlog_beck")(t, truth, t)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, truth, atol=0.05)


def test_agauss_recovers_peak():
    t = np.linspace(1, 365, 60)
    truth = 0.1 + 0.7 * np.exp(-(((t - 180) / 40) ** 2))
    out = get_reconstructor("agauss")(t, truth, t)
    assert np.all(np.isfinite(out))
    assert abs(t[np.argmax(out)] - 180) < 30
