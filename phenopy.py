

# import datetime

from matplotlib import pyplot as plt


def PhenoShape(inData, dates=None, rollWindow=None, nGS=46, nan_replace=None, saveRaster=None):
    """
    Explanaition here....
    """
    import xarray as xr
    import pandas as pd
    import numpy as np

    # get variables
    if isinstance(inData, str):
        try:
            import rasterio
            # load raster as a xarray
            xarray = xr.open_rasterio(inData).rename({'band': 'time'})
            # load also metadata
            with rasterio.open(inData) as img:
                meta = img.profile
            # assing dates as time values
            xarray.time.values = dates
            # attribites
            attrs = xarray.attrs
            # add day of the year coordinates
            xarray.coords['doy'] = xarray.time.dt.dayofyear
        except RasterioIOError:
            print('ERROR: data must be a GDAL format')
    elif isinstance(xarray, xr.DataArray):
        try:
            # add day of the year coordinates
            xarray.coords['doy'] = xarray.time.dt.dayofyear
            # attribites
            attrs = xarray.attrs
        except TypeError:
            print('ERROR: The xarray do not have "time" dimention in datetime64 format')
    else:
        print('ERROR: inData must be either a string with the path to a'
              ' raster image or a xarray with time dimension with dates.')

    # sort basds according to day-of-the-year
    xarray = xarray.sortby('doy')
    # turn a value to NaN
    if nan_replace is not None:
        xarray = xarray.where(xarray.values != nan_replace)
    # rolling average using moving window
    if rollWindow is not None:
        xarray = xarray.rolling(time=rollWindow, center=True).mean()

    # prepare inputs to getPheno
    x = xarray.doy.values
    y = xarray.values

    def getPheno(y, x):
        """
        Apply linear interpolation in the 'time' axis
        x: DOY values
        y: ndarray with VI values
        """
        inds = np.isnan(y)  # check if array has NaN values
        if inds.any(): # if inds have at least one True
            x = x[~inds]
            y = y[~inds]
            xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
            ynew = np.interp(xnew, x, y)

        else:
            xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
            ynew = np.interp(xnew, x, y)

        return xnew, ynew

    # get phenology shape accross the time axis
    xnew, phen = np.apply_along_axis(getPheno, 0, y, x)
    # add phenology into an xarray format according to the input xarray
    phen = xr.DataArray(phen, dims=xarray.dims, coords={
                        'time': xnew[:, 0, 0],
                        'y': xarray.coords['y'],
                        'x': xarray.coords['x']},
                        attrs=attrs)
    if saveRaster is None:
        return phen
    elif isinstance(saveRaster, str):
        # edit metadata before save the raster
        meta.update(count=nGS, dtype=phen.values.dtype)
        # save results
        with rasterio.open(saveRaster, "w", **meta) as dst:
            dst.write(phen)
