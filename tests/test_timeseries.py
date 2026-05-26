"""Smoke test for the interannual moving-window orchestrator and that the
reconstruction/extraction axes are threaded through it."""

import numpy as np

import phenosensing  # noqa: F401  (registers the accessor)
from phenosensing.io import load_sample


def test_get_timeseries_metrics_runs_and_threads_extraction():
    da = load_sample("SIF").isel(y=slice(0, 3), x=slice(0, 3))
    years = np.unique(da.year.values)[:5]
    da = da.where(da.year.isin(years), drop=True)

    ts = da.pheno.get_timeseries_metrics(
        window_length=3,
        metric=["sos", "pos", "eos"],
        nGS=30,
        extraction="trs",
    )

    assert set(ts) == {"sos", "pos", "eos"}
    for key in ts:
        assert "time" in ts[key].dims
        # one value per sliding 3-year window over 5 years -> 3 windows
        assert ts[key].sizes["time"] == 3
