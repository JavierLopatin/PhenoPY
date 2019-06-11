################################################################################
#
# PhenoPy
#
################################################################################

from rasterstats import point_query
from shapely.geometry import Point
import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from numba import jit

def PhenoPlot(X, Y, inData, dates, saveFigure=None, ylim=None, rollWindow=None,
              nGS=46, ylab='NDVI'):
    """
    Plot the PhenoShape curve along with the yearly data

    Args:
        - X: X coordinates
        - Y: Y coordinates
        - inData: Absolute path to the original timeseries data
        - dates: Dates of the original timeseries data [dtype: datetime64[ns]]
        - saveFigure: Absolute path with extention to save figure on disk
        - ylim: Limits of the Y axis [default the y min() and max() values]
        - rollWindow: integer with value of avarage smoothing of linear trend [default None]
        - nGS: Number of observations to predict the PhenoShape [default 46: one per week]
        - ylab: Label of the Y axis [default "NDVI"]

    """
    # get spatial point
    point = Point(X, Y)
    # get pixel value per pixel
    # first read metadata to get number of bands
    with rasterio.open(inData) as r:
        countTSS = r.count

    # save values
    valuesTSS = []
    for i in range(countTSS):
        valuesTSS.append(point_query(point, inData, band=(i+1)))
    valuesTSS = np.array(valuesTSS, dtype=np.float).squeeze()

    # load dates
    valuesTSSpd = pd.concat([pd.DataFrame(dates),
                             pd.DataFrame(dates.dt.dayofyear),
                             pd.DataFrame(dates.dt.year),
                             pd.DataFrame(valuesTSS)], axis=1).dropna()
    valuesTSSpd.columns = ['dates', 'doy', 'year', 'VI']

    # group values according to year
    groups = valuesTSSpd.groupby('year')

    # Transform numpy array to xarray
    xarray = xr.DataArray(valuesTSS)
    # DOY coordinates
    xarray.coords['doy'] = dates.dt.dayofyear
    # sort values by DOY
    xarray = xarray.sortby('doy')
    # rearrange time dimension for smoothing
    xarray['dim_0'] = xarray['doy']
    # rolling average using moving window
    if rollWindow is not None:
        xarray = xarray.rolling(dim_0=rollWindow, center=True).mean()

    # predict linear interpolation
    @jit
    def getPheno(y, x):
        """
        Apply linear interpolation in the 'time' axis
        x: DOY values
        y: ndarray with VI values
        """
        inds = np.isnan(y)  # check if array has NaN values
        if np.sum(inds) == len(x):  # check is all values are NaN
            print('ERROR: All values are NaN')
        else:
            if inds.any():  # if inds have at least one True
                x = x[~inds]
                y = y[~inds]
                xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
                ynew = np.interp(xnew, x, y)

            else:
                xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
                ynew = np.interp(xnew, x, y)

            return xnew, ynew

    # get phenology shape accross the time axis
    y = xarray.values
    x = xarray.doy.values
    xnew, phen = getPheno(y, x)

    # plot
    for name, group in groups:
        plt.plot(group.doy, group.VI, marker='o', linestyle='', ms=10, label=name)
    plt.plot(xnew, phen, '-', color='black')
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.ylabel(ylab, fontsize=14)
    plt.xlabel('Day of the year', fontsize=14)
    if saveFigure is not None:
        plt.savefig(saveFigure)
    plt.show()


def PhenoShape(inData, saveRaster, dates=None, nan_replace=None,
               rollWindow=None, nGS=46, chuckSize=256):
    """
    Process phenological shape of remote sensing data by
    folding data to day-of-the-year. Process is done in a block-by-block way
    with parallel processing.

    Args:
        - inData: Absolute path to the original timeseries data
        - dates: Dates of the original timeseries data [dtype: datetime64[ns]]
        - saveRaster: Absolute path with extention to save raster on disk
        - ylim: Limits of the Y axis [default the y min() and max() values]
        - rollWindow: integer with value of avarage smoothing of linear trend [default None]
        - nGS: Number of observations to predict the PhenoShape [default 46: one per week]
    """
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
    except TypeError:
        print('ERROR: data must be a GDAL format')

    # sort basds according to day-of-the-year
    xarray = xarray.sortby('doy')
    # rearrange time dimension for smoothing and interpolation
    xarray['time'] = xarray['doy']
    # turn a value to NaN
    if nan_replace is not None:
        xarray = xarray.where(xarray.values != nan_replace)
    # rolling average using moving window
    if rollWindow is not None:
        xarray = xarray.rolling(time=rollWindow, center=True).mean()
    # prepare inputs to getPheno
    x = xarray.doy.values
    y = xarray.values

    @jit
    def getPheno(y, x):
        """
        Apply linear interpolation in the 'time' axis
        x: DOY values
        y: ndarray with VI values
        """
        inds = np.isnan(y)  # check if array has NaN values
        y.dtype
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
    xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
    # add phenology into an xarray format according to the input xarray
    phen = xr.DataArray(phen, dims=xarray.dims, coords={
                        'time': np.linspace(np.min(x), np.max(x), nGS, dtype='int16'),
                        'y': xarray.coords['y'],
                        'x': xarray.coords['x']},
                        attrs=attrs)
        # edit metadata before save the raster
    meta.update(count=nGS, dtype=phen.values.dtype, nodata=np.nan)
    # save results
    with rasterio.open(saveRaster, "w", **meta) as dst:
        dst.write(phen)
        # save band description to metadata
        for i in range(len(xnew)):
            dst.set_band_description(i+1, 'DOY '+str(xnew[i]))


min = np.min(phen)
