"""Unit tests for the SOS/EOS extraction registry (axis 3)."""

import numpy as np
import pytest

import phenopy  # noqa: F401  (registers the accessor)
from phenopy.extraction import get_extractor, list_extractors
from phenopy.io import load_sample

METHODS = ["seasonal_median", "trs", "der", "curvature"]


def _synthetic_curve():
    xnew = np.linspace(1, 365, 60).astype(int)
    phen = 0.2 + 0.6 * np.exp(-(((xnew - 180) / 45) ** 2))
    trough = phen.min()
    ampl = phen.max() - trough
    ratio = (phen - trough) / ampl
    greenup = np.gradient(ratio) > 0
    ipos = np.where(phen == phen.max())[0]
    return phen, xnew, ratio, greenup, ipos


def test_registry_lists_expected_methods():
    for m in METHODS:
        assert m in list_extractors()


@pytest.mark.parametrize("method", METHODS)
def test_sos_before_pos_before_eos(method):
    phen, xnew, ratio, greenup, ipos = _synthetic_curve()
    sos, eos, _, _ = get_extractor(method)(phen, xnew, ratio, greenup, ipos)
    pos = xnew[ipos[0]]
    assert sos <= pos <= eos


def test_trs_crosses_threshold_at_sos():
    phen, xnew, ratio, greenup, ipos = _synthetic_curve()
    _, _, isos, _ = get_extractor("trs")(phen, xnew, ratio, greenup, ipos, threshold=0.5)
    assert ratio[int(isos[0])] >= 0.5 - 1e-9


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        get_extractor("not-a-method")


@pytest.mark.parametrize("method", METHODS)
def test_phenolsp_runs_with_each_extraction(method):
    da = load_sample("ndvi")
    shape = da.pheno.PhenoShape(rollWindow=3, nGS=40)
    lsp = shape.pheno.PhenoLSP(extraction=method).compute()
    assert {"sos", "pos", "eos"}.issubset(set(map(str, lsp.data_vars)))
