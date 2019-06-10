
#from fbprophet import Prophet
import xarray as xr
import pandas as pd
#import datetime
import numpy as np
from matplotlib import pyplot as plt
import os

# in data
os.chdir('/home/javier/Documents/SF_delta/Sentinel/TSA/npphen_consulta')
days = 'dates.txt'
raster = 'TSS_toy.tif'

#%%
def Phenology(inData, dates=None, rollWindow=None, nGS = 46, nan_replace=None, saveRaster=None):

    # get variables
    if isinstance(inData, str):
        inData = xr.open_rasterio(inData).rename({'band': 'time'})
        # assing dates as time values
        inData.time.values = dates
        # attribites
        attrs = inData.attrs
        # add day of the year coordinates
        inData.coords['doy'] = inData.time.dt.dayofyear
    elif isinstance(inData, xr.DataArray):
        try:
            # add day of the year coordinates
            inData.coords['doy'] = inData.time.dt.dayofyear
            # attribites
            attrs = inData.attrs
        except:
            print('ERROR: The xarray do not have "time" dimention with dates')
    else:
        print('ERROR: inData must be either a string with the path to a raster image or a xarray with time dimension with dates.')

    # sort basds according to day-of-the-year
    inData = inData.sortby('doy')
    # turn a value to NaN
    if nan_replace != None:
        inData = inData.where(inData.values != nan_replace)
    # rolling average using moving window
    inData = inData.rolling(time=rollWindow, center=True).mean()

    # prepare inputs to getPheno
    x = inData.doy.values
    y = inData.values

    def getPheno(y,x):
        '''
        Apply linear interpolation in the 'time' axis
        x: DOY values
        y: ndarray with VI values
        '''
        inds = np.isnan(y) # check if array has NaN values
        if inds.any() == True:
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
    phen = xr.DataArray(phen, dims=inData.dims, coords={'time':xnew[:,0,0],
                                                       'y':inData.coords['y'],
                                                       'x':inData.coords['x']}, attrs=attrs)
    if saveRaster == None:
        return phen
    elif isinstance(saveRaster, str):
        import rasterio
        # save results
        with rasterio.open(outfile, "w", **inData.attrs) as dst:
            dst.write(xarray)
#%%

# import dates
dates = pd.read_csv(days, header=None)[0]
years = pd.to_datetime(dates)
# create dummy set of dates all with same year to mix according to DOY
#years2 = years.apply(lambda dt: dt.replace(year=2018))

# load imeage as xarray and remane bands
data = xr.open_rasterio(raster).rename({'band': 'time'})
# assing dates as time values
data.time.values = years

# see data
data[:, 0, 0].plot.line(marker='o'); plt.ylabel('EVI')
# drop -32767 values
data = data.where(data.values != -32767)
# check values again
data[:, 0, 0].plot.line(marker='o'); plt.ylabel('EVI')

# get phenology values
phen = Phenology(data, rollWindow=5)
phen[:,0,0].plot()


phen = Phenology(raster, dates=years, rollWindow=5, nan_replace=-32767)
phen[:,0,0].plot()
