###############################################################################
#
# PhenoPy is a Python 3.X library to process temporal data derived from
# EarthObservation data.
#
###############################################################################

from __future__ import division
from rasterstats import point_query
from shapely.geometry import Point
import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.integrate import trapz
from scipy.interpolate import Rbf, interp1d
from scipy.stats import skew
from tqdm import tqdm
import concurrent.futures
from functools import partial
import sys
from kneed import KneeLocator

# suppress numpy warnings of zero division
np.seterr(divide='ignore', invalid='ignore')

# --------------------------------------------------------------------------- #
# ------------------------------- FUNCTIONS --------------------------------- #
# --------------------------------------------------------------------------- #
def PhenoPlot(X, Y, inData, dates, type='linear', saveFigure=None, ylim=None,
              rollWindow=None, plotType=1, nGS=46, n_phen=15, ylab='NDVI'):
    """
    Plot the PhenoShape curve along with the yearly data

    Parameters
    ----------
    - X: Float
            X coordinates
    - Y: Float
            Y coordinates
    - inData: String
            Absolute path to the original timeseries data
    - dates: String
            Dates of the original timeseries data [dtype: datetime64[ns]]
    - type = String or Integer
            Interpolation type. Must be a string of ‘linear’, ‘nearest’,
            ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘RBF‘, ‘previous’,
            ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer
            to a spline interpolation of zeroth, first, second or third order;
            ‘previous’ and ‘next’ simply return the previous or next value"
            of the point) or as an integer specifying the order of the"
            spline interpolator to use. RBF uses cubic interpolation.
            Default is ‘linear’.
    - saveFigure: String
            Absolute path with extention to save figure on disk
    - ylim: List of Integers or Float
            Limits of the Y axis [default the y min() and max() values]
    - plotType: Type of plot, where 1 = plot with accumulated years and 2 = plot with
            start of the season (SOS), peak of the season (POS) and end of
            season (EOS)
            default is 1
    - rollWindow: Integers
            Value of avarage smoothing of linear trend [default None]
    - nGS: Integer
            Number of observations to predict the PhenoShape
            default is 46; one per week
    - ylab: string
            Label of the Y axis [default "NDVI"]

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
        valuesTSS.append(point_query(point, inData, band=(i + 1)))
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
    # fill NaN values if any
    xarray.values = _fillNaN(xarray.values)

    # rolling average using moving window
    if rollWindow is not None:
        xarray = xarray.rolling(dim_0=rollWindow, center=True).mean()
        xarray.values = _fillNaN(xarray.values)
    # predict linear interpolation
    # get phenology shape accross the time axis
    y = xarray.values
    x = xarray.doy.values
    phen = _getPheno(y, x, nGS, type)
    xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')

    # plot
    if plotType == 1:
        for name, group in groups:
            plt.plot(group.doy, group.VI, marker='o',
                     linestyle='', ms=10, label=name)
        plt.plot(xnew, phen, '-', color='black')

    elif plotType == 2:
        # get position of SOS, POS, and EOS
        metrics = _getLSPmetrics(phen, xnew, n_phen, len(xnew))
        isos = np.where(xnew == metrics[0])[0][0]
        ipos = np.where(xnew == metrics[1])[0][0]
        ieos = np.where(xnew == metrics[2])[0][0]

        plt.plot(xnew, phen, '-', color='black')
        plt.plot(xnew[isos], phen[isos], 'X', markersize=15,
                 label='SOS')
        plt.plot(xnew[ipos], phen[ipos], 'X', markersize=15,
                 label='POS')
        plt.plot(xnew[ieos], phen[ieos], 'X', markersize=15,
                 label='EOS')
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.ylabel(ylab, fontsize=14)
    plt.xlabel('Day of the year', fontsize=14)
    if saveFigure is not None:
        plt.savefig(saveFigure)
    plt.show()

# ---------------------------------------------------------------------------#

def PhenoShape(inData, outData, interpolType='linear', dates=None,
               nan_replace=None, rollWindow=None, nGS=46, n_jobs=4,
               chuckSize=256):
    """
    Process phenological shape of remote sensing data by
    folding data to day-of-the-year. Process is done in a block-by-block way
    with parallel processing.

    Parameters
    ----------
    - inData: String
        Absolute path to the original timeseries data
    - outData: String
        Absolute path with extention to save raster on disk
    - interpolType: String or Integer
        Type of interpolation to perform. Options include ‘linear’, ‘nearest’,
        ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘RBF‘, ‘previous’, ‘next’,
        where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
        interpolation of zeroth, first, second or third order; ‘previous’ and
        ‘next’ simply return the previous or next value of the point.
         Integer numbers specify the order of the spline interpolator to use.
         Default is ‘linear’.
    - dates: 1D pandas file
        Dates of the original timeseries data [dtype: datetime64[ns]]
    - nan_replace: Integer
        Value of the NaN data if there are any
    - rollWindow: Integer
        Value of avarage smoothing of linear trend [default None]
    - nGS: Integer
        Number of observations to predict the PhenoShape
        default is 46 [one per week]
    - n_jobs: Integer
        Number of parallel jorb to apply during modeling
    - chuckSize: Integer
        Size of raster chunks to be loaded during modeling
        Number must be multiple of 16 [GDAL specifications]
        default value is 256 [256 X 256 raster blocks]

    """

    # get names for output bands
    doy = np.linspace(1, 365, nGS, dtype='int16')
    bandNames = []
    for i in range(nGS):
        bandNames.append('DOY - ' + str(doy[i]))

    # call _getPheno2 function to loal
    do_work = partial(_getPheno2, dates=dates, nGS=nGS, type=interpolType,
                      nan_replace=nan_replace, rollWindow=rollWindow)

    # apply PhenoShape with parallel processing
    try:
        print('Processing PhenoShape of ', inData)
        _parallel_process(inData, outData, do_work, nGS, n_jobs, chuckSize,
                          bandNames)
    except AttributeError:
        print('ERROR in parallel processing...')


# ---------------------------------------------------------------------------#

def PhenoLSP(inData, outData, doy, nGS=46, n_phen=15, n_jobs=4, chuckSize=256):
    """
    Obtain land surfurface phenology metrics for a PhenoShape product

    Parameters
    ----------
    - inData: String
        Absolute path to PhenoShape raster data
    - outData: String
        Absolute path for output land surface phenology raster
    - doy: 1D array
        Numpy array of the days of the year of the time series
    - nGS: Integer
        Number of observations to predict the PhenoShape
        default is 46; one per week
    - n_phen: Integer
        Window size where to estimate SOS and EOS
    - n_jobs: Integer
        Number of parallel jorb to apply during modeling
    - chuckSize: Integer
        Size of raster chunks to be loaded during modeling
        Number must be multiple of 16 [GDAL specifications]
        default value is 256 [256 X 256 raster blocks]

    outputs
    -------
    Raster stack with the followingvariables:
        - SOS: DOY of start of season
        - POS: DOY of peak of season
        - EOS: DOY of end of season
        - vSOS: Value at start of season
        - vPOS: Value at peak of season
        - vEOS: Value at end of season
        - LOS: Length of season (DOY)
        - MGS: Mean growing season
        - MSP: Mean senescence season
        - MAU: Mean autum
        - AOS: Amplitude of season (in value units)
        - IOS: Integral of season (SOS-EOS)
        - ROG: Rate of greening [slope SOS-POS]
        - ROS: Rate of senescence [slope POS-EOS]
        - SW: Skewness of growing season
    """

    # name of output bands
    bandNames = ['SOS - DOY of Start of season',
                 'POS - DOY of Peak of season',
                 'EOS - DOY of End of season',
                 'vSOS - Vaues at start os season',
                 'vPOS - Values at peak of season',
                 'vEOS - Values at end of season',
                 'LOS - Length of season',
                 'MGS - Mean growing season',
                 'MSP - Mean senescence season',
                 'MAU - Mean autum',
                 'AOS - Amplitude of season',
                 'IOS - Integral of season [SOS-EOS]',
                 'ROG - Rate of greening [slope SOS-POS]',
                 'ROS - Rate of senescence [slope POS-EOS]',
                 'SW - Skewness of growing season [SOS-EOS]']

    # call _getPheno2 function to loal
    do_work = partial(_cal_LSP, nGS=nGS, doy=doy, n_phen=n_phen, num=len(bandNames))
    # apply PhenoShape with parallel processing
    try:
        print('Processing LSP of ', inData)
        _parallel_process(inData, outData, do_work, len(bandNames), n_jobs,
                          chuckSize, bandNames)
    except AttributeError:
        print('ERROR in parallel processing...')



###############################################################################
# Utility functions
###############################################################################

def _parallel_process(inData, outData, do_work, count,  n_jobs, chuckSize, bandNames):
    """
    Process infile block-by-block with parallel processing
    and write to a new file.
    Input function must be call _main() and need only one input (ndarray)
    """
    # apply parallel processing with rasterio
    with rasterio.Env():
        with rasterio.open(inData) as src:
            # Create a destination dataset based on source params. The
            # destination will be tiled, and we'll process the tiles
            # concurrently.
            profile = src.profile
            profile.update(blockxsize=chuckSize, blockysize=chuckSize,
                           count=count, dtype='float64', tiled=True)

            with rasterio.open(outData, "w", **profile) as dst:
                # Materialize a list of destination block windows
                # that we will use in several statements below.
                windows = [window for ij, window in dst.block_windows()]
                # This generator comprehension gives us raster data
                # arrays for each window. Later we will zip a mapping
                # of it with the windows list to get (window, result)
                # pairs.
                data_gen = (src.read(window=window) for window in windows)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_jobs
                ) as executor:
                    # Map the a function over the raster
                    # data generator, zip the resulting iterator with
                    # the windows list, and as pairs come back we
                    # write data to the destination dataset.
                    for window, result in zip(
                        tqdm(windows), executor.map(do_work, data_gen)
                    ):
                        dst.write(result, window=window)
                # save band description to metadata
                for i in range(profile['count']):
                    dst.set_band_description(i + 1, bandNames[i])

# ---------------------------------------------------------------------------#

def _getPheno(y, x, nGS, type):
    """
    Apply linear interpolation in the 'time' axis
    x: DOY values
    y: ndarray with VI values
    """
    inds = np.isnan(y)  # check if array has NaN values
    if np.sum(inds) == len(y):  # check is all values are NaN
        return y[0:nGS]
    else:
        try:
            if inds.any():  # if inds have at least one True
                x = x[~inds]
                y = y[~inds]
                _replaceElements(x, len(x))
                xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
                if type == 'linear':
                    ynew = np.interp(xnew, x, y)
                elif type == 'RBF':
                    f = Rbf(x, y, funciton='cubic')
                    ynew = f(xnew)
                else:
                    f = interp1d(x, y, kind=type)
                    ynew = f(xnew)
            else:
                xnew = np.linspace(1, 365, nGS, dtype='int16')
                ynew = np.interp(xnew, x, y)
        except NotImplementedError:
            print("ERROR: Interpolation type must be ‘linear’, ‘nearest’,"
                  "‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘RBF‘, ‘previous’,"
                  "‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer"
                  "to a spline interpolation of zeroth, first, second or third order;" "‘previous’ and ‘next’ simply return the previous or next value"
                  "of the point) or as an integer specifying the order of the"
                  "spline interpolator to use. Default is ‘linear’.")

    return ynew

# ---------------------------------------------------------------------------#

def _getPheno2(dstack, dates, nGS, type, nan_replace, rollWindow):
    """
    Obtain shape of phenological responses

    Parameters
    ----------
    - dstack: 3D arrays

    """
    # load raster as a xarray
    xarray = xr.DataArray(dstack)
    xarray.coords['dim_0'] = dates
    xarray.coords['doy'] = xarray.dim_0.dt.dayofyear

    # sort basds according to day-of-the-year
    xarray = xarray.sortby('doy')
    # rearrange time dimension for smoothing and interpolation
    xarray['dim_0'].values = xarray['doy']
    # turn a value to NaN
    if nan_replace is not None:
        xarray = xarray.where(xarray.values != nan_replace)
    # rolling average using moving window
    if rollWindow is not None:
        xarray = xarray.rolling(dim_0=rollWindow, center=True).mean()
    # prepare inputs to getPheno
    x = xarray.doy.values
    y = xarray.values

    # get phenology shape accross the time axis
    return np.apply_along_axis(_getPheno, 0, y, x, nGS, type)

# ---------------------------------------------------------------------------#

def _getLSPmetrics(phen, xnew, n_phen, num):
    """
    Obtain land surfurface phenology metrics

    Parameters
    ----------
    - phen: 1D array
        PhenoShape data
    - xnew: 1D array
        DOY values for PhenoShape data
    - n_phen: Integer
        Window size where to estimate SOS and EOS
    - num: Integer
        Number of output variables

    Outputs
    -------
    - 2D array with the following variables:

        SOS = DOY of start of season
        POS = DOY of peak of season
        EOS = DOY of end of season
        vSOS = Value at start of season
        vPOS = Value at peak of season
        vEOS = Value at end of season
        LOS = Length of season (DOY)
        MGS = Mean growing season
        MSP = Mean senescence season
        MAU = Mean autum
        AOS = Amplitude of season (in value units)
        IOS = Integral of season (SOS-EOS)
        ROG = Rate of greening [slope SOS-POS]
        ROS = Rate of senescence [slope POS-EOS]
        SW = Skewness of growing season
    """
    inds = np.isnan(phen)  # check if array has NaN values
    if np.sum(inds) > 0:  # check is all values are NaN
        return np.repeat(np.nan, num)
    else:
        try:
            # basic variables
            peak = np.max(phen)
            ipos = np.where(phen == peak)[0]
            pos = xnew[ipos]
            trough = np.min(phen)
            ampl = peak - trough

            # get position of seasonal peak and trough
            ipos = np.where(phen == peak)[0]

            # scale annual time series to 0-1
            ratio = (phen - trough) / ampl

            # select time where SOS and EOS are located (around trs value)
            # KneeLocator looks for the inflection index in the curve
            try:
                knee = KneeLocator(xnew[0:n_phen], phen[0:n_phen], curve='convex',
                                   direction='increasing')
                sos = knee.knee
                isos = np.where(xnew == knee.knee)[0]

                knee = KneeLocator(xnew[-n_phen:], phen[-n_phen:], curve='convex',
                                   direction='decreasing')
                eos = knee.knee
                ieos = np.where(xnew == knee.knee)[0]
                if sos == None:
                    isos = 0
                    sos = xnew[isos]
                if eos == None:
                    ieos = len(xnew)-1
                    eos = xnew[ieos]
            except ValueError:
                sos = np.nan
                isos = np.nan
                eos = np.nan
                ieos = np.nan
            except TypeError:
                sos = np.nan
                isos = np.nan
                eos = np.nan
                ieos = np.nan

            # los: length of season
            try:
                los = eos - sos
                if los < 0:
                    los[los < 0] = len(phen) + (eos[los < 0] - sos[los < 0])
            except ValueError:
                los = np.nan
            except TypeError:
                los = np.nan

            # get MGS, MSP, MAU
            try:
                mgs = np.mean(phen[ratio > 0.5])  # mean growing season
                msp = mau = np.nan  # mean senescence season
                if ~np.isnan(sos):
                    id = np.arange((isos-10), (isos+10))
                    id = id[(id > 0) & (id < len(phen))]
                    msp = []
                    for i in range(len(id)):
                        msp.append(phen[id[i]])
                msp = np.mean(msp)
                if ~np.isnan(eos):
                    id = np.arange((ieos-10), (ieos+10))
                    id = id[(id > 0) & (id < len(phen))]
                    mau = []
                    for i in range(len(id)):
                        mau.append(phen[id[i]])
                    mau = np.mean(mau)

            except ValueError:
                mgs = np.nan
                msp = np.nan
                mau = np.nan
            except TypeError:
                mgs = np.nan
                msp = np.nan
                mau = np.nan

            # doy of growing season
            try:
                green = xnew[(xnew > sos) & (xnew < eos)]
                id = []
                for i in range(len(green)):
                    id.append((xnew == green[i]).nonzero()[0])
                # index of growing season
                id = np.array([item for sublist in id for item in sublist])
            except ValueError:
                id = np.nan
            except TypeError:
                id = np.nan

            # get intergral of green season
            try:
                ios = trapz(phen[id], xnew[id])
            except ValueError:
                ios = np.nan
            except TypeError:
                ios = np.nan

            # rate of greening [slope SOS-POS]
            try:
                rog = (peak - phen[isos])/(pos - sos)
            except ValueError:
                rog = np.nan
            except TypeError:
                rog = np.nan

            # rate of senescence [slope POS-EOS]
            try:
                ros = (phen[ieos] - peak)/(eos - pos)
            except ValueError:
                ros = np.nan
            except TypeError:
                ros = np.nan

            # skewness of growing season
            try:
                sw = skew(phen[id])
            except ValueError:
                sw = np.nan
            except TypeError:
                sw = np.nan

            metrics = np.array((sos, pos, eos, phen[isos], peak, phen[ieos],
                                los, mgs, msp, mau, ampl, ios, rog, ros, sw))

            return metrics

        except IndexError:
            return np.repeat(np.nan, num)
        except ValueError:
            return np.repeat(np.nan, num)
        except TypeError:
            return np.repeat(np.nan, num)

# ---------------------------------------------------------------------------#

def _LSP(y, x, min_sep):
    """
    Obtain land surfurface phenology metrics

    Parameters
    ----------
    - y: 1D array
        PhenoShape data
    - x: 1D array
        DOY vaues for PhenoShape data

    Outputs
    -------
    - 2D array with the following variables:

        SOS = DOY of start of season
        POS = DOY of peak of season
        EOS = DOY of end of season
        vSOS = Value at start of season
        vPOS = Value at peak of season
        vEOS = Value at end of season
        LOS = Length of season (DOY)
        AOS = Amplitude of season (in value units)
        IOS = Integral of season (SOS-EOS)
        ROG = Rate of greening [slope SOS-POS]
        ROS = Rate of senescence [slope POS-EOS]
        SW = Skewness of the PhenoShape distribution
    """
    # set general data
    num = 12  # number of variables to output

    inds = np.isnan(y)  # check if array has NaN values
    if np.sum(inds) > 0:  # check is all values are NaN
        return np.repeat(np.nan, num)
    else:
        try:
            # obtain peaks
            peak_max = _detect_peaks(y, min_sep)
            peak_min = _detect_peaks(y, min_sep, valley=True)
            if len(peak_max) == 0:
                peak_max = x[np.where(y == np.max(y))[0]]
            elif len(peak_max) > 1:
                peak_max = peak_max[0]
            else:
                pass
            if len(peak_min) == 0 or len(peak_min) == 1:
                # Find the location of where the values change according to the neightbors
                # This is in case the interpolation creates a straoght line when no values
                # are found at the extreams
                start = [i for i in range(1, len(y)) if y[i] != y[i - 1]][0]
                end = [i for i in range(1, len(y)) if y[i] != y[i - 1]][-1]
                peak_min = np.array([start, end])
            elif len(peak_min) > 2:
                peak_min = np.array([peak_min[0], peak_min[-1]])
            else:
                pass

            # Get LSP metrics
            try:
                SOS = x[peak_min[0]]  # start of season
            except ValueError:
                SOS = np.nan
            try:
                POS = x[peak_max]  # peak of season
            except ValueError:
                POS = np.nan
            try:
                EOS = x[peak_min[1]]  # end of season
            except ValueError:
                EOS = np.nan
            try:
                vSOS = y[peak_min[0]]  # value at start of season
            except ValueError:
                vSOS = np.nan
            try:
                vPOS = y[peak_max]  # value at peak of season
            except ValueError:
                vPOS = np.nan
            try:
                vEOS = y[peak_min[1]]  # value at end of season
            except ValueError:
                vEOS = np.nan
            try:
                LOS = EOS - SOS  # length of season
            except ValueError:
                LOS = np.nan
            try:
                AOS = y[peak_max] - np.min(y[peak_min])  # amplitude of season
            except ValueError:
                AOS = np.nan
            green = x[(x > SOS) & (x < EOS)]  # doy of growing season
            id = []
            for i in range(len(green)):
                id.append((x == green[i]).nonzero()[0])
            id = np.array([item for sublist in id for item in sublist])
            try:
                # get intergral of green season
                id = []
                for i in range(len(green)):
                    id.append((x == green[i]).nonzero()[0])
                id = np.array([item for sublist in id for item in sublist])
                IOS = trapz(y[id], x[id])
            except ValueError:
                IOS = np.nan
            # rate of greening [slope SOS-POS]
            try:
                ROG = (vPOS - vSOS)/(POS - SOS)
            except ValueError:
                ROG = np.nan
            # rate of senescence [slope POS-EOS]
            try:
                ROS = (vEOS - vPOS)/(EOS - POS)
            except ValueError:
                ROS = np.nan
            # skewness of growing season
            try:
                SW = skew(y[id])
            except ValueError:
                SW = np.nan
            return np.array((SOS, POS, EOS, vSOS, vPOS, vEOS, LOS, AOS, IOS, ROG, ROS, SW))
        except IndexError:
            return np.repeat(np.nan, num)
        except ValueError:
            return np.repeat(np.nan, num)
# ---------------------------------------------------------------------------#

def _cal_LSP(dstack, nGS, doy, n_phen, num):
    """
    Process the _LSP funciton into an 3D arrays

    Parameters
    ----------
    - dstack: 3D array
        PhenoShape data.
    - min_sep: integer
         Distance to consider betweem peaks and bottoms.
    - nGS: integer
        Number of DOY values
    - num: Integer
        Number of output variables

    Output
    ------
    - 3D arrays
        Stack with the LSP metrics descibes in _LSP
    """
    # prepare input data for _LSP
    xnew = np.linspace(np.min(doy), np.max(doy), nGS, dtype='int16')

    # estimate LSP metrics along the 0 axis
    return np.apply_along_axis(_getLSPmetrics, 0, dstack, xnew, n_phen, num)

# ---------------------------------------------------------------------------#

def _replaceElements(arr, n):
    '''
    Replace monotonic vector values to avoid
    interpolation errors
    '''
    s = []
    for i in range (n):
        # check whether the element
        # is repeated or not
        if arr[i] not in s:
            s.append(arr[i])
        else:
            # find the next greatest element
            for j in range(arr[i] + 1, sys.maxsize):
                if j not in s:
                    arr[i] = j
                    s.append(j)
                    break

# ---------------------------------------------------------------------------#

def _fillNaN(x):
    # Fill NaN data by linear interpolation
    mask = np.isnan(x)
    x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    return x
