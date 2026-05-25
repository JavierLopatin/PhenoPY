"""Regenerate the golden LSP snapshots used by ``tests/test_golden.py``.

Run from the repo root::

    python tests/golden/_generate.py

Only regenerate when a change to the numerical output is *intended* and
reviewed; otherwise the golden snapshots are the regression baseline.
"""

from pathlib import Path

import numpy as np

import phenopy  # noqa: F401  (registers the `pheno` accessor)
from phenopy.io import load_sample

OUT = Path(__file__).parent
PHENOSHAPE_PARAMS = dict(interpolType="linear", rollWindow=5, nGS=52)


def main() -> None:
    for name in ["SIF", "ndvi"]:
        da = load_sample(name)
        lsp = da.pheno.PhenoShape(**PHENOSHAPE_PARAMS).pheno.PhenoLSP().compute()
        arrs = {str(b): lsp[b].values for b in lsp.data_vars}
        np.savez_compressed(OUT / f"lsp_{name}.npz", **arrs)
        print(f"wrote lsp_{name}.npz ({len(arrs)} metrics)")


if __name__ == "__main__":
    main()
