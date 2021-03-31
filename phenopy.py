###############################################################################
#
# PhenoPy is a Python 3.X library to process phenology indices derived from
# EarthObservation data.
#
###############################################################################

# libraries included in Python 3.X
from __future__ import division
import concurrent.futures
from functools import partial
import sys
import warnings

# common dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import Rbf, interp1d
from scipy.stats import skew
from sklearn.metrics import mean_squared_error

# speciel dependencies
import xarray as xr                  # manipulate 3D time-series rasters
import shapely.geometry as geom      # create geographical points
import rasterio                      # manipulate GeoTIFF
from rasterstats import point_query  # extract raster values
from tqdm import tqdm                # progress bar
from kneed import KneeLocator        # find inflection point on a curve
from KDEpy import FFTKDE             # perform fast 2D kernel density estimations


# --------------------------------------------------------------------------- #
# ------------------------------- FUNCTIONS --------------------------------- #
# --------------------------------------------------------------------------- #
def PhenoPlot(X, Y, inData, dates, type='KDE', saveFigure=None, ylim=None, rollWindow=None,
              nan_replace=None, correctionValue=None, plotType=1, phentype=1, nGS=46, n_phen=15,
              fontsize=14, titlesize=15, legendsize=15, labelsize=13, threshold=300, ylab='NDVI'):
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
    - dates: Series
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
    - plotType: Type of plot, where 1 = plot with accumulated years; 2 = plot with
            start of the season (SOS), peak of the season (POS) and end of
            season (EOS);
            default is 1
    - phenType: Type os estimation of SOS and EOS. 1 = median value between POS and start and end of season. 2 = using the knee inflexion method.
            default 1
    - rollWindow: Integers
            Value of avarage smoothing of linear trend [default None]
    - nGS: Integer
            Number of observations to predict the PhenoShape
            default is 46; one per week
    - ylab: string
            Label of the Y axis [default "NDVI"]

    """
    # get spatial point
    point = geom.Point(X, Y)
    # get pixel value per pixel
    # first read metadata to get number of bands
    with rasterio.open(inData) as r:
        countTSS = r.count

    # save values
    valuesTSS = []
    for i in range(countTSS):
        valuesTSS.append(point_query(point, inData, band=(i + 1)))
    valuesTSS = np.array(valuesTSS, dtype=np.float).squeeze()

    if nan_replace is not None:
        valuesTSS = np.where(valuesTSS == nan_replace, np.nan, valuesTSS)
        
    if correctionValue is not None:
        valuesTSS = valuesTSS/correctionValue
    
    # add dates
    valuesTSSpd = pd.concat([pd.DataFrame(dates),
                             pd.DataFrame(dates.dt.dayofyear),
                             pd.DataFrame(dates.dt.year),
                             pd.DataFrame(valuesTSS)], axis=1)
    valuesTSSpd.columns = ['dates', 'doy', 'year', 'VI']
    valuesTSSpd = valuesTSSpd.sort_values('doy')

    # group values according to year
    groups = valuesTSSpd.groupby('year')

    # get phenological shape
    phen = _getPheno0(y=valuesTSS, doy=dates.dt.dayofyear, interpolType=type, 
                       nan_replace=None, rollWindow=rollWindow, nGS=nGS)

    # doy of the predicted phenological shape
    xnew = np.linspace(np.min(valuesTSSpd.doy), np.max(valuesTSSpd.doy), nGS,
                       dtype='int16')

    # plot
    if plotType == 1:
        # get %RMSE
        rmse = _RMSE(valuesTSSpd.doy.values, valuesTSSpd.VI.values, 
                     xnew, phen)#.round(2)
        minn = np.nanmin(valuesTSS)
        maxx = np.nanmax(valuesTSS)
        nRMSE = ((rmse/(maxx-minn))*100).round(2)
        
        for name, group in groups:
            plt.plot(group.doy, group.VI, marker='o',
                     linestyle='', ms=10, label=name)
        plt.plot(xnew, phen, '-', color='black')
        plt.legend(prop={'size': legendsize})
        plt.tick_params(labelsize=labelsize)
        plt.title('%RMSE = ' + str(nRMSE), loc='left', size=titlesize)
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

        plt.ylabel(ylab, fontsize=fontsize)
        plt.xlabel('Day of the year', fontsize=fontsize)

    elif plotType == 2:
        # get position of SOS, POS, and EOS
        metrics = _getLSPmetrics(phen, xnew, nGS, len(xnew), phentype)
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

        plt.legend(prop={'size': 12})
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        plt.ylabel(ylab, fontsize=fontsize)
        plt.xlabel('Day of the year', fontsize=fontsize)

    if saveFigure is not None:
        plt.savefig(saveFigure)
    plt.show()

# ---------------------------------------------------------------------------#


def PhenoShape(inData, outData, doy, interpolType='KDE', nan_replace=None,
               rollWindow=None, nGS=46, n_jobs=4, chuckSize=256):
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
    - doy: 1D vector with day of the year data
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
    xnew = np.linspace(np.min(doy), np.max(doy), nGS, dtype='int16')
    bandNames = []
    for i in range(nGS):
        bandNames.append('DOY - ' + str(xnew[i]))

    # call _getPheno2 function to lcoal
    do_work = partial(_getPheno2, doy=doy, interpolType=interpolType, 
                      nan_replace=nan_replace, rollWindow=rollWindow, nGS=nGS)

    # apply PhenoShape with parallel processing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _parallel_process(inData, outData, do_work, nGS, n_jobs, chuckSize,
                              bandNames)
    except AttributeError:
        print('ERROR in parallel processing...')


# ---------------------------------------------------------------------------#

def PhenoLSP(inData, outData, doy, nGS=46, phentype=1, n_phen=10, n_jobs=4,
             chuckSize=256):
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
     - phenType: Type os estimation of SOS and EOS. 1 = median value between POS and start and end of season. 2 = using the knee inflexion method.
            default 1
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
        - SOS - DOY of Start of season
        - POS - DOY of Peak of season
        - POS - DOY of End of season
        - vSOS - Vaues at start os season
        - vPOS - Values at peak of season
        - vEOS - Values at end of season
        - LOS - Length of season
        - MSP - Mid spring (DOY)
        - MAU - Mid autum (DOY)
        - vMSP - Value at mid spring
        - vMAU - Value at mid autum
        - AOS - Amplitude of season
        - IOS - Integral of season [SOS-EOS]
        - ROG - Rate of greening [slope SOS-POS]
        - ROS - Rate of senescence [slope POS-EOS]
        - SW - Skewness of growing season [SOS-EOS]
    """

    # name of output bands
    bandNames = ['SOS - DOY of Start of season',
                 'POS - DOY of Peak of season',
                 'EOS - DOY of End of season',
                 'vSOS - Vaues at start os season',
                 'vPOS - Values at peak of season',
                 'vEOS - Values at end of season',
                 'LOS - Length of season',
                 'MSP - Mid spring (DOY)',
                 'MAU - Mid autum (DOY)',
                 'vMSP - Value at mean spring',
                 'vMAU - Value at mean autum',
                 'AOS - Amplitude of season',
                 'IOS - Integral of season [SOS-EOS]',
                 'ROG - Rate of greening [slope SOS-POS]',
                 'ROS - Rate of senescence [slope POS-EOS]',
                 'SW - Skewness of growing season [SOS-EOS]']

    # call _cal_LSP function to local
    do_work = partial(_cal_LSP, nGS=nGS, phentype=phentype, doy=doy,
                      n_phen=n_phen, num=len(bandNames))
    # apply PhenoLSP with parallel processing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _parallel_process(inData, outData, do_work, len(bandNames), n_jobs,
                              chuckSize, bandNames)
    except AttributeError:
        print('ERROR in parallel processing...')

# ---------------------------------------------------------------------------#


def RMSE(inData, inShape, outData, dates, normalized=False, nan_replace=None, nGS=46):
    """
    Obtain Root Mean Square Error (RMSE) values between the fitted PhenoShape
    and the real distribution of values

    Parameters
    ----------
    - inData: String
        Absolute path to original timeseries data
    - inShape: String
        Absolute path to PhenoShape data
    - outData: String
        Absolute path for output RMSE raster
    - dates: Series
        Dates of the original timeseries data [dtype: datetime64[ns]]
    - normalized: Boolean
        Perform a normalization of the RMSE values into percentages units
        Default if False
    - nan_replace: Integer
        Value of the NaN data if there are any
    - nGS: Integer
        Number of observations to predict the PhenoShape
        default is 46; one per week

    outputs
    -------
    Single-band raster with RMSE values (in same raw units as inData)
    """
 
    # read rasters
    with rasterio.open(inShape) as r:
        meta = r.profile
        phen = r.read()
    with rasterio.open(inData) as r:
        dstack = r.read()
    
    # repalce NaN if needed
    t1 = np.isin(dstack, nan_replace)
    if np.any(t1):
        dstack = dstack.astype('Float64')
        dstack[dstack == nan_replace] = np.nan
    
    # process RMSE
    rmse = _RMSE2(phen, dstack, dates, nan_replace, nGS)
    
    # normalization
    if normalized==True:
        minn = np.nanmin(dstack, axis=0)
        maxx = np.nanmax(dstack, axis=0)
        rmse = ((rmse/(maxx-minn))*100)

    # edit metadata to save raster
    meta.update(count=1, dtype='float64')

    # save results
    with rasterio.open(outData, "w", **meta) as dst:
        dst.write(rmse)

        """

    # call _cal_LSP function to local
    do_work = partial(_RMSE2, dates=dates, nan_replace=nan_replace, nGS=nGS)

    if chuckSize % 16 == 0:
        # apply parallel processing with rasterio
        with rasterio.Env():
            # open first raster
            with rasterio.open(inData) as src:
                profile = src.profile
                profile.update(blockxsize=chuckSize, blockysize=chuckSize,
                               count=1, dtype='float64', tiled=True)
                # create raster to save chuncks
                with rasterio.open(outData, "w", **profile) as dst:
                    windows = [window for ij, window in dst.block_windows()]
                    data_gen = (src.read(window=window) for window in windows)
                    # OPEN SECOND RASTER!
                    with rasterio.open(inShape) as r:
                        data_gen2 = (r.read(window=window) for window in windows)
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=n_jobs
                        ) as executor:
                            for window, result in zip(
                                tqdm(windows), executor.map(lambda x, y: do_work(x, y), data_gen, data_gen2)
                            ):
                                dst.write(result, window=window)



                # save band description to metadata
                        for i in range(profile['count']):
                            dst.set_band_description(i + 1, bandNames[i])
    else:
        print('ERROR! chuckSize needs to be divisible by 16')

    """

###############################################################################
# Utility functions
###############################################################################


def _parallel_process(inData, outData, do_work, count,  n_jobs, chuckSize,
                      bandNames):
    """
    Process infile block-by-block with parallel processing
    and write to a new file.
    chunckSize needs t be divisible by 16

    """
    if chuckSize % 16 == 0:
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
    else:
        print('ERROR! chuckSize needs to be divisible by 16')
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
            xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
            if inds.any():  # if inds have at least one True
                # x = x[~inds]
                # y = y[~inds]
                y = _fillNaN(y)
            if type == 'linear':
                ynew = np.interp(xnew, x, y)
            elif type == 'RBF':
                _replaceElements(x)  # replace doy values when are the same
                f = Rbf(x, y, funciton='cubic')
                ynew = f(xnew)
            elif type == 'KDE':
                ynew = _KDE(x, y, nGS)
            else:
                _replaceElements(x)  # replace doy values when are the same
                f = interp1d(x, y, kind=type)
                ynew = f(xnew)
            """
            plt.plot(x, y, 'o')
            plt.plot(xnew, ynew)
            plt.plot(x[inds], y[inds], 'X')
            """

            return ynew

        except NotImplementedError:
            print("ERROR: Interpolation type must be ‘KDE’ ‘linear’, ‘nearest’,"
                  "‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘RBF‘, ‘previous’,"
                  "or ‘next’. Here, 'KSE' correspond to a non-parametric linear"
                  "regression using Kernel Density Estimators with Gaussian kernel."
                  "‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline "
                  "interpolation of zeroth, first, second or third order;"
                  "‘previous’ and ‘next’ simply return the previous or next value"
                  "of the point) or as an integer specifying the order of the"
                  "spline interpolator to use. Default is KDE.")

# ---------------------------------------------------------------------------#

def _getPheno0(y, doy, interpolType, nan_replace, rollWindow, nGS):
 
    # replace nan_relace values by NaN
    if nan_replace is not None:
        y = np.where(y == nan_replace, np.nan, y)
  
    # sort values by DOY  
    idx = doy.argsort()
    y = y[idx]     
    
    # prepare tails for interpolation
    '''
    minn = np.nanmin(y)
    start = y[0:3]
    end = y[-3:]
    if np.all( np.isnan(start) ):
        y[0:3] = minn
    if np.all( np.isnan(end) ):
        y[-3:] = minn
    '''
    # get phenological shape
    phen = _getPheno(y, doy[idx], nGS, interpolType)
    
    # rolling average using moving window
    if rollWindow is not None:
        phen = _moving_average(phen, rollWindow)
    
    return phen


# ---------------------------------------------------------------------------#

def _getPheno2(dstack, doy, interpolType, nan_replace, rollWindow, nGS):
    """
    Obtain shape of phenological responses

    Parameters
    ----------
    - dstack: 3D arrays

    """    

    # get phenology shape accross the time axis
    return np.apply_along_axis(_getPheno0, 0, dstack, doy, interpolType, 
                               nan_replace, rollWindow, nGS)


# ---------------------------------------------------------------------------#


def _getLSPmetrics(phen, xnew, nGS, num, phentype):
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
        MSP = Mean spring (DOY)
        MAU = Mean autum (DOY)
        vMSP = Mean spring value
        vMAU = Mean autum value
        AOS = Amplitude of season (in value units)
        IOS = Integral of season (SOS-EOS)
        ROG = Rate of greening [slope SOS-POS]
        ROS = Rate of senescence [slope POS-EOS]
        SW = Skewness of growing season
    """
    inds = np.isnan(phen)  # check if array has NaN values
    if inds.any():  # check is all values are NaN
        return np.repeat(np.nan, num)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # basic variables
                vpos = np.max(phen)
                ipos = np.where(phen == vpos)[0]
                pos = xnew[ipos]
                trough = np.min(phen)
                ampl = vpos - trough

                # get position of seasonal peak and trough
                ipos = np.where(phen == vpos)[0]

                # scale annual time series to 0-1
                ratio = (phen - trough) / ampl

                # separate greening from senesence values
                dev = np.gradient(ratio)  # first derivative
                greenup = np.zeros([ratio.shape[0]],  dtype=bool)
                greenup[dev > 0] = True

                # select time where SOS and EOS are located (around trs value)
                # KneeLocator looks for the inflection index in the curve
                try:
                    with warnings.catch_warnings():
                        if phentype == 1:  # estimate SOS and EOS as median of the season
                            i = np.median(xnew[:ipos[0]][greenup[:ipos[0]]])
                            ii = np.median(xnew[ipos[0]:][~greenup[ipos[0]:]])
                            sos = xnew[(np.abs(xnew - i)).argmin()]
                            eos = xnew[(np.abs(xnew - ii)).argmin()]
                            isos = np.where(xnew == int(sos))[0]
                            ieos = np.where(xnew == eos)[0]
                        elif phentype == 2:  # estimate SOS and EOS by inflection curves
                            warnings.simplefilter("ignore")
                            # consider only observation before POS for SOS
                            knee1 = KneeLocator(xnew[0:ipos[0]], ratio[0:ipos[0]], S=2,
                                                curve='convex', direction='increasing')
                            sos = knee1.knee
                            isos = np.where(xnew == knee1.knee)[0]

                            # consider only observation after POS for EOS
                            x = xnew[-(nGS - ipos[0] - 1):]
                            y = ratio[-(nGS - ipos[0] - 1):]
                            knee2 = KneeLocator(range(len(x)), np.flip(y), S=2,
                                                curve='convex', direction='increasing')
                            eos = x[np.where(
                                np.flip(range(len(x))) == knee2.knee)[0]][0]
                            ieos = np.where(xnew == eos)[0]
                        else:
                            print('phentype must be either 1 or 2')
                    if sos is None:
                        isos = 0
                        sos = xnew[isos]
                    if eos is None:
                        ieos = len(xnew) - 1
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
                        los[los < 0] = len(phen) + \
                            (eos[los < 0] - sos[los < 0])
                except ValueError:
                    los = np.nan
                except TypeError:
                    los = np.nan

                # get MSP, MAU (independent from SOS and EOS)
                # mean spring
                try:
                    idx = np.mean(xnew[(xnew > sos) & (xnew < pos[0])])
                    idx = (np.abs(xnew - idx)).argmin()  # indexing value
                    msp = xnew[idx]  # DOY of MGS
                    vmsp = phen[idx]  # mgs value

                except ValueError:
                    msp = np.nan
                    vmsp = np.nan
                except TypeError:
                    msp = np.nan
                    vmsp = np.nan
                # mean autum
                try:
                    idx = np.mean(xnew[(xnew < eos) & (xnew > pos[0])])
                    idx = (np.abs(xnew - idx)).argmin()  # indexing value
                    mau = xnew[idx]  # DOY of MGS
                    vmau = phen[idx]  # mgs value

                except ValueError:
                    mau = np.nan
                    vmau = np.nan
                except TypeError:
                    mau = np.nan
                    vmau = np.nan

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
                    rog = (vpos - phen[isos]) / (pos - sos)
                except ValueError:
                    rog = np.nan
                except TypeError:
                    rog = np.nan

                # rate of senescence [slope POS-EOS]
                try:
                    ros = (phen[ieos] - vpos) / (eos - pos)
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

                metrics = np.array((sos, pos[0], eos, phen[isos][0], vpos,
                                    phen[ieos][0], los, msp, mau, vmsp, vmau, ampl, ios, rog[0],
                                    ros[0], sw))

                return metrics

        except IndexError:
            return np.repeat(np.nan, num)
        except ValueError:
            return np.repeat(np.nan, num)
        except TypeError:
            return np.repeat(np.nan, num)

# ---------------------------------------------------------------------------#


def _cal_LSP(dstack, nGS, doy, n_phen, num, phentype):
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
    return np.apply_along_axis(_getLSPmetrics, 0, dstack, xnew, nGS, num, phentype)

# ---------------------------------------------------------------------------#


def _RMSE(x, y, xnew, ynew):
    """
    Obtain RMSE values form 1D data inputs

    Parameters
    ----------
    - x, y: 1D array
        Values for DOY and time series data from the original dataset
    - xnew, ynew: 1D array
        Values of DOY and PhenoShape obtained by PhenoShape function
    """

    inds = np.isnan(ynew)  # check if array has NaN values
    inds2 = np.isnan(y)
    if inds.any():  # check is all values are NaN
        return np.nan
    else:
        if inds2.any():
            y = _fillNaN(y)
        ypred2 = np.interp(x, xnew, ynew)

        return np.sqrt(mean_squared_error(ypred2, y))

# ---------------------------------------------------------------------------#


def _RMSE2(phen, dstack, dates, nan_replace, nGS):
    """
    Apply _RMSE funciton to a spatial 3D arrays
    """
    # original data - dstack
    xarray = xr.DataArray(dstack)
    xarray.coords['dim_0'] = dates.dt.dayofyear
    # sort basds according to day-of-the-year
    xarray = xarray.sortby('dim_0')
    if nan_replace is not None:
        xarray = xarray.where(xarray.values != nan_replace)
    # xarray.values =  np.apply_along_axis(_fillNaN, 0, xarray.values)
    x = xarray.dim_0.values
    y = xarray.values

    xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
    # change shape from 3D to 2D matrix
    y2 = y.reshape(y.shape[0], (y.shape[1] * y.shape[2]))
    ynew = phen.reshape(phen.shape[0], (y.shape[1] * y.shape[2]))

    rmse = np.zeros((y.shape[1] * y.shape[2]))
    for i in tqdm(range(y.shape[1] * y.shape[2])):
        # print(i)
        rmse[i] = _RMSE(x, y2[:, i], xnew, ynew[:, i])

    # reshape from 2D to 3D
    return rmse.reshape(1, phen.shape[1], y.shape[2])

# ---------------------------------------------------------------------------#


def _replaceElements(arr):
    '''
    Replace monotonic vector values to avoid
    interpolation errors
    '''
    s = []
    for i in range(len(arr)):
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

# ---------------------------------------------------------------------------#


def _KDE(x, y, nGS):
    """Compute a bivariate kde using KDEpy."""

    # Grid points in the x and y direction
    grid_points_x, grid_points_y = nGS + 6, 2**8

    # Stack the data for 2D input, compute the KDE
    data = np.vstack((x, y)).T
    kde = FFTKDE(bw=0.025).fit(data)
    grid, points = kde.evaluate((grid_points_x, grid_points_y))

    # Retrieve grid values, reshape output and plot boundaries
    x2, y2 = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points_x, grid_points_y)

    # Compute y_pred = E[y | x] = sum_y p(y | x) * y
    y_pred = np.sum((z.T / np.sum(z, axis=1)).T * y2, axis=1)
    id = np.where(x2 < np.min(x))
    x2 = np.delete(x2, id)
    y_pred = np.delete(y_pred, id)
    id = np.where(x2 > np.max(x))
    y_pred = np.delete(y_pred, id)

    return y_pred

# ---------------------------------------------------------------------------#
    
def _moving_average(a, n=3) :
    out = np.convolve(a, np.ones(n), 'valid') / n    
    return np.concatenate([ a[:np.int(n/2)], out, a[-np.int(n/2):] ]) # add values of tail

# ---------------------------------------------------------------------------#
