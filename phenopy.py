def PhenoShape(inData, saveRaster, dates=None, nan_replace=None,
               rollWindow=None, nGS=46, chuckSize=256):
    """
    Explanaition here....
    """
    import xarray as xr
    import numpy as np
    from numba import jit
    import rasterio
    from dash import ProgressBar

    # get variables
    try:
        # load raster as a xarray
        xarray = xr.open_rasterio(inData, chunks={'x': chuckSize, 'y': chuckSize}).rename({'band': 'time'})
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
    y = xarray.values  # [:,100,100]

    @jit
    def getPheno(y, x):
        """
        Apply linear interpolation in the 'time' axis
        x: DOY values
        y: ndarray with VI values
        """
        inds = np.isnan(y)  # check if array has NaN values
        if np.sum(inds) == len(x):  # check is all values are NaN
            return y[0:nGS]
        else:
            if inds.any():  # if inds have at least one True
                x = x[~inds]
                y = y[~inds]
                xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
                ynew = np.interp(xnew, x, y)

            else:
                xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
                ynew = np.interp(xnew, x, y)

            return ynew

    # get phenology shape accross the time axis
    phen = np.apply_along_axis(getPheno, 0, y, x)
    # add phenology into an xarray format according to the input xarray
    phen = xr.DataArray(phen, dims=xarray.dims, coords={
                        'time': np.linspace(np.min(x), np.max(x), nGS, dtype='int16'),
                        'y': xarray.coords['y'],
                        'x': xarray.coords['x']},
                        attrs=attrs)
    # phen[10,:,:].plot()

    # save to disk

    # edit metadata before save the raster
    meta.update(count=nGS, dtype=phen.values.dtype)
    # save results
    with rasterio.open(saveRaster, "w", **meta) as dst:
        dst.write(phen)
