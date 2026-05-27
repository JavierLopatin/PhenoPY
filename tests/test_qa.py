"""Tests for the source-agnostic QA -> weight decoder (no Earth Engine needed)."""

import numpy as np
import pytest
import xarray as xr

from phenosensing.qa import list_qa_specs, qa_to_weight


def _da(vals):
    return xr.DataArray(np.array(vals, dtype=float), dims=["time"], name="qa")


def test_remap_modis_summaryqa():
    # SummaryQA: 0 good -> 1, 1 marginal -> 0.5, 2 snow / 3 cloud -> 0
    w = qa_to_weight(_da([0, 1, 2, 3, 0]), "MOD13Q1")
    assert list(w.values) == [1.0, 0.5, 0.0, 0.0, 1.0]
    assert w.name == "weight"


def test_good_classes_s2_scl():
    # SCL: 4 veg, 5 bare, 6 water are kept; 8/9 cloud, 3 shadow, 11 snow dropped
    w = qa_to_weight(_da([4, 5, 6, 8, 9, 3, 11]), "S2_SCL")
    assert list(w.values) == [1, 1, 1, 0, 0, 0, 0]


def test_bad_bits_landsat_qa_pixel():
    # bad_bits = [1,2,3,4,5]; bit 6 is not flagged -> stays good
    qa = _da([0, 1 << 3, 1 << 4, 1 << 6, 1 << 1])
    w = qa_to_weight(qa, "LANDSAT_C2")
    assert list(w.values) == [1, 0, 0, 1, 0]


def test_threshold_dict_cloud_probability():
    # e.g. an S2 cloud-probability band: keep <= 30 %
    w = qa_to_weight(_da([10, 30, 50, np.nan]), {"max_value": 30})
    assert list(w.values) == [1, 1, 0, 0]


def test_custom_callable():
    w = qa_to_weight(_da([0, 1, 2]), lambda q: q * 0 + 0.7)
    assert float(w.mean()) == pytest.approx(0.7)


def test_nan_is_always_bad():
    assert list(qa_to_weight(_da([0, np.nan]), "MOD13Q1").values) == [1.0, 0.0]


def test_weights_preserve_dims_and_coords():
    qa = xr.DataArray(
        np.zeros((3, 2, 2)), dims=["time", "y", "x"], coords={"time": [1, 2, 3]}, name="SCL"
    )
    w = qa_to_weight(qa, "S2_SCL")
    assert w.dims == qa.dims
    assert list(w["time"].values) == [1, 2, 3]


def test_unknown_key_raises_listing_options():
    with pytest.raises(ValueError, match="Unknown QA spec"):
        qa_to_weight(_da([0]), "NOPE")
    assert "S2_SCL" in list_qa_specs()


def test_bad_dict_raises():
    with pytest.raises(ValueError, match="must contain one of"):
        qa_to_weight(_da([0]), {"nonsense": 1})
