"""Tests for bootstrap per-metric uncertainty."""

import numpy as np

import phenosensing  # noqa: F401  (registers the accessor)
from phenosensing.io import load_sample
from phenosensing.uncertainty import uncertainty
from phenosensing.utils import LSP_BANDS


def test_uncertainty_structure_and_reproducible():
    da = load_sample("ndvi").isel(y=slice(0, 4), x=slice(0, 4))

    u1 = uncertainty(da, n_boot=12, nGS=30, seed=0)
    assert set(LSP_BANDS).issubset(set(map(str, u1.data_vars)))
    assert set(u1["sos"].dims) == {"y", "x"}
    # a standard deviation is non-negative (where finite)
    assert float(np.nanmin(u1["sos"].values)) >= 0.0

    # same seed -> identical result
    u2 = uncertainty(da, n_boot=12, nGS=30, seed=0)
    np.testing.assert_allclose(u1["sos"].values, u2["sos"].values, equal_nan=True)
