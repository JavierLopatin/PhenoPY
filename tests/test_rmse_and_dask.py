"""RMSE regression tests and verification that the Dask (chunked) path matches
the eager path after the apply_ufunc refactor."""

import dask
import numpy as np
import pytest

import phenopy  # noqa: F401  (registers the accessor)
from phenopy.io import load_sample


def test_rmse_overall_and_segmented():
    da = load_sample("ndvi")
    shape = da.pheno.PhenoShape(rollWindow=3, nGS=40)
    lsp = shape.pheno.PhenoLSP()

    overall = shape.pheno.RMSE(da, LSP_stack=lsp, normalized=True)
    assert "rmse" in overall.data_vars

    seg = shape.pheno.RMSE(da, LSP_stack=lsp, normalized=True, segment=True)
    assert {"rmse", "rmse_sos", "rmse_pos", "rmse_eos"}.issubset(set(map(str, seg.data_vars)))


def test_rmse_requires_phenoshape_output():
    da = load_sample("ndvi")  # has a 'time' dim, not a 'doy' dim
    with pytest.raises(ValueError):
        da.pheno.RMSE(da, LSP_stack=None)


def test_chunked_phenoshape_is_dask_and_matches_eager():
    da = load_sample("ndvi")
    eager = da.pheno.PhenoShape(rollWindow=3, nGS=40)
    lazy = da.pheno.PhenoShape(rollWindow=3, nGS=40, chunks={"x": 5, "y": 5})

    assert dask.is_dask_collection(lazy.data)
    np.testing.assert_allclose(eager.values, lazy.compute().values, equal_nan=True)


def test_chunked_phenolsp_matches_eager():
    da = load_sample("ndvi")
    eager_lsp = da.pheno.PhenoShape(rollWindow=3, nGS=40).pheno.PhenoLSP()
    lazy_lsp = (
        da.pheno.PhenoShape(rollWindow=3, nGS=40, chunks={"x": 5, "y": 5})
        .pheno.PhenoLSP()
        .compute()
    )
    for band in eager_lsp.data_vars:
        np.testing.assert_allclose(
            eager_lsp[band].values, lazy_lsp[band].values, rtol=1e-5, equal_nan=True
        )
