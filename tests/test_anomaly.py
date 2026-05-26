"""Tests for non-parametric phenological anomaly detection."""

import numpy as np
import pandas as pd
import xarray as xr

import phenopy  # noqa: F401  (registers the accessor)
from phenopy.anomaly import anomaly


def _cube_with_low_year(ny=2, nx=2, anomalous_year=2019):
    """5 years of weekly data; one year is anomalously low everywhere."""
    times = pd.date_range("2017-01-01", periods=5 * 46, freq="8D")
    doy = times.dayofyear.values
    year = times.year.values
    season = 0.3 + 0.4 * np.sin((doy / 365.0) * 2 * np.pi - np.pi / 2)
    base = np.broadcast_to(season[:, None, None], (len(times), ny, nx)).copy()
    base[year == anomalous_year] -= 0.2  # a clear negative anomaly
    da = xr.DataArray(
        base,
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(ny), "x": np.arange(nx)},
    )
    return da.assign_coords(doy=("time", doy), year=("time", year))


def test_anomaly_outputs():
    da = _cube_with_low_year()
    out = anomaly(da, nGS=46, rollWindow=3)
    assert {"anomaly", "rfd", "z"}.issubset(set(map(str, out.data_vars)))
    for v in ("anomaly", "rfd", "z"):
        assert set(out[v].dims) == {"time", "y", "x"}
    # RFD is a percentile in (0, 1]
    rfd = out["rfd"].values
    assert np.nanmin(rfd) >= 0 and np.nanmax(rfd) <= 1


def test_anomaly_detects_low_year():
    da = _cube_with_low_year(anomalous_year=2019)
    out = anomaly(da, nGS=46, rollWindow=3)

    a2019 = out["anomaly"].where(da.year == 2019, drop=True).mean().item()
    rfd2019 = out["rfd"].where(da.year == 2019, drop=True).mean().item()
    rfd_other = out["rfd"].where(da.year != 2019, drop=True).mean().item()

    assert a2019 < -0.05  # clearly negative deviation in the dry year
    assert rfd2019 < rfd_other  # and it ranks as more extreme (lower percentile)
