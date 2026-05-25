"""I/O helpers for PhenoPY, including the small bundled example datasets."""

from importlib.resources import files

import pandas as pd
import rioxarray
import xarray as xr

__all__ = ["load_sample", "list_samples"]

# name -> (raster filename, paired date CSV filename)
_SAMPLES = {
    "SIF": ("SIF_sample.tif", "SIF_sample_dates.csv"),
    "ndvi": ("ndvi_test.tif", "dates_ndvi.csv"),
}


def list_samples() -> list:
    """Return the names of the bundled example datasets."""
    return sorted(_SAMPLES)


def load_sample(name: str = "SIF") -> xr.DataArray:
    """Load a bundled example vegetation-index time series.

    Parameters
    ----------
    name:
        Which bundled dataset to load; one of :func:`list_samples`
        (``"SIF"`` or ``"ndvi"``).

    Returns
    -------
    xarray.DataArray
        A ``(time, y, x)`` array with ``doy`` and ``year`` coordinates,
        ready to be used through the ``.pheno`` accessor.
    """
    try:
        tif_name, csv_name = _SAMPLES[name]
    except KeyError:
        raise ValueError(f"Unknown sample {name!r}; choose from {list_samples()}.") from None

    data_dir = files("phenopy").joinpath("data")
    dates = pd.to_datetime(pd.read_csv(str(data_dir.joinpath(csv_name)), header=None).iloc[:, 0])

    da = rioxarray.open_rasterio(str(data_dir.joinpath(tif_name)))
    da = da.rename({"band": "time"}).assign_coords(time=dates.values)
    da = da.assign_coords(
        doy=("time", da["time"].dt.dayofyear.values),
        year=("time", da["time"].dt.year.values),
    )
    return da
