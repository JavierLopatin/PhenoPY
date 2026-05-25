"""Golden regression tests.

Freeze the 16 land-surface-phenology metrics computed by the *current* code on
the bundled sample datasets. The upcoming composable refactor (Phase 2) must
reproduce these numbers within tolerance, so any silent numerical drift fails
the test. Regenerate the ``.npz`` snapshots with ``tests/golden/_generate.py``
only when a change to the output is intended and reviewed.
"""

from pathlib import Path

import numpy as np
import pytest

import phenopy  # noqa: F401  (importing registers the `pheno` accessor)
from phenopy.io import load_sample

GOLDEN_DIR = Path(__file__).parent / "golden"
PHENOSHAPE_PARAMS = dict(interpolType="linear", rollWindow=5, nGS=52)


@pytest.mark.parametrize("name", ["SIF", "ndvi"])
def test_lsp_matches_golden(name):
    da = load_sample(name)
    lsp = da.pheno.PhenoShape(**PHENOSHAPE_PARAMS).pheno.PhenoLSP().compute()

    golden = np.load(GOLDEN_DIR / f"lsp_{name}.npz")
    assert set(golden.files) == {str(v) for v in lsp.data_vars}

    for band in golden.files:
        np.testing.assert_allclose(
            lsp[band].values,
            golden[band],
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
            err_msg=f"{name}/{band} drifted from the golden snapshot",
        )
