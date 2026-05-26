"""Decode a per-observation quality (QA) band into reconstruction weights.

QA encodings are sensor-specific, but almost all reduce to one of four patterns,
each expressible as a small declarative spec (a ``dict``) -- or, for anything
exotic, a custom callable:

- ``remap``        : map ordinal QA values to weights (e.g. MODIS ``SummaryQA``).
- ``good_classes`` : keep a set of class labels (e.g. Sentinel-2 ``SCL``).
- ``bad_bits``     : a packed bitmask (e.g. Landsat ``QA_PIXEL``).
- ``max_value`` / ``min_value`` : a continuous threshold (e.g. a cloud-probability
  band).

The decoder is deliberately **source-agnostic**: it operates on any xarray QA
``DataArray`` -- from Earth Engine, local rasters, NetCDF, ... -- so Earth Engine
is never a dependency of PhenoPY. ``examples/`` shows a GEE -> xarray bridge that
feeds this function. Missing QA (``NaN``) is always treated as bad (weight 0).

The resulting weights (in ``[0, 1]``) are what the weighted reconstruction
consumes; PhenoPY's algorithms never need to know which sensor produced them.
"""

from __future__ import annotations

import xarray as xr

# Built-in specs for common products. The value is the decode rule; ``band`` is a
# hint for which raster band holds the QA when loading the data (it is ignored by
# ``qa_to_weight``, which receives the band directly).
QA_SPECS: dict[str, dict] = {
    "MOD13Q1": {"band": "SummaryQA", "remap": {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}},
    "LANDSAT_C2": {"band": "QA_PIXEL", "bad_bits": [1, 2, 3, 4, 5]},
    "S2_SCL": {"band": "SCL", "good_classes": [4, 5, 6]},
}


def list_qa_specs() -> list[str]:
    """Return the names of the built-in QA specs."""
    return sorted(QA_SPECS)


def get_qa_spec(name: str) -> dict:
    """Return the built-in QA spec registered under ``name``."""
    try:
        return QA_SPECS[name]
    except KeyError:
        raise ValueError(
            f"Unknown QA spec {name!r}; choose from {list_qa_specs()} or pass a dict/callable."
        ) from None


def _remap(qa: xr.DataArray, mapping: dict, default: float) -> xr.DataArray:
    out = xr.full_like(qa, float(default), dtype=float)
    for value, weight in mapping.items():
        out = out.where(qa != value, float(weight))  # set out=weight where qa==value
    return out


def _good_classes(qa: xr.DataArray, classes) -> xr.DataArray:
    return qa.isin(list(classes)).astype(float)  # NaN is not in the list -> 0


def _bad_bits(qa: xr.DataArray, bits) -> xr.DataArray:
    mask = int(sum(1 << int(b) for b in bits))
    qi = qa.fillna(-1).astype("int64")  # -1 has all bits set -> flagged bad
    good = (qi & mask) == 0
    return good.astype(float).where(qa.notnull(), 0.0)


def _threshold(qa: xr.DataArray, lo, hi) -> xr.DataArray:
    out = xr.ones_like(qa, dtype=float)
    if hi is not None:
        out = out.where(qa <= hi, 0.0)
    if lo is not None:
        out = out.where(qa >= lo, 0.0)
    return out.where(qa.notnull(), 0.0)


def qa_to_weight(qa: xr.DataArray, spec) -> xr.DataArray:
    """Decode a QA ``DataArray`` into weights in ``[0, 1]`` (same shape as ``qa``).

    Parameters
    ----------
    qa:
        The raw quality band, as an xarray ``DataArray``.
    spec:
        How to decode it -- one of:

        - a built-in registry key (see :func:`list_qa_specs`), e.g. ``"S2_SCL"``;
        - a declarative ``dict`` with exactly one of ``remap`` (+ optional
          ``default``), ``good_classes``, ``bad_bits``, or ``max_value`` /
          ``min_value``;
        - a callable ``qa -> weights`` for encodings that fit no pattern.

    Returns
    -------
    xarray.DataArray
        Weights in ``[0, 1]`` named ``"weight"``, aligned to ``qa``.
    """
    if callable(spec):
        return spec(qa)
    if isinstance(spec, str):
        spec = get_qa_spec(spec)
    if not isinstance(spec, dict):
        raise TypeError("spec must be a registry key (str), a dict, or a callable")

    if "remap" in spec:
        out = _remap(qa, spec["remap"], spec.get("default", 0.0))
    elif "good_classes" in spec:
        out = _good_classes(qa, spec["good_classes"])
    elif "bad_bits" in spec:
        out = _bad_bits(qa, spec["bad_bits"])
    elif "max_value" in spec or "min_value" in spec:
        out = _threshold(qa, spec.get("min_value"), spec.get("max_value"))
    else:
        raise ValueError(
            "spec dict must contain one of: 'remap', 'good_classes', 'bad_bits', "
            f"'max_value'/'min_value'; got keys {list(spec)}."
        )
    return out.rename("weight")
