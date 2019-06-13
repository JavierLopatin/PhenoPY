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
from scipy.integrate import simps
from tqdm import tqdm
import concurrent.futures
from functools import partial

# ---------------------------------------------------------------------------#
def PhenoPlot(X, Y, inData, dates, saveFigure=None, ylim=None, rollWindow=None,
    type=1, nGS=46, ylab='NDVI'):
    """
    Plot the PhenoShape curve along with the yearly data

    Parameters
    ----------
    - X: X coordinates
    - Y: Y coordinates
    - inData: Absolute path to the original timeseries data
    - dates: Dates of the original timeseries data [dtype: datetime64[ns]]
    - saveFigure: Absolute path with extention to save figure on disk
    - ylim: Limits of the Y axis [default the y min() and max() values]
    - type: Type of plot, where 1 = plot with accumulated years and 2 = plot with
            start of the season (SOS), peak of the season (POS) and end of
            season (EOS) [default 1]
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
    # rolling average using moving window
    if rollWindow is not None:
        xarray = xarray.rolling(dim_0=rollWindow, center=True).mean()

    # predict linear interpolation
    # get phenology shape accross the time axis
    y = xarray.values
    x = xarray.doy.values
    phen = _getPheno(y, x, nGS)
    xnew = np.linspace(1, 365, nGS, dtype='int16')
    plt.plot(xnew, phen, '-', x, y, 'o')

    # plot
    if type == 1:
        for name, group in groups:
            plt.plot(group.doy, group.VI, marker='o',
                     linestyle='', ms=10, label=name)
        plt.plot(xnew, phen, '-', color='black')

    elif type == 2:
        # detect peaks
        peak_max = _detect_peaks(phen, nGS / 2)
        peak_min = _detect_peaks(phen, nGS / 2, valley=True)
        if len(peak_min) == 0:
            # Find the location of where the values change according to the neightbors
            # This is in case the interpolation creates a straoght line when no values
            # are found at the extreams
            start = [i for i in range(1, len(phen))
                     if phen[i] != phen[i - 1]][0]
            end = [i for i in range(1, len(phen))
                   if phen[i] != phen[i - 1]][-1]
            peak_min = np.array([start, end])

        plt.plot(xnew, phen, '-', color='black')
        plt.plot(xnew[peak_min[0]], phen[peak_min[0]], 'X', markersize=15,
                 label='SOS')
        plt.plot(xnew[peak_max], phen[peak_max], 'X', markersize=15,
                 label='POS')
        plt.plot(xnew[peak_min[1]], phen[peak_min[1]], 'X', markersize=15,
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

def PhenoShape(inData, outData, dates=None, nan_replace=None, rollWindow=None,
    nGS=46, chuckSize=256, n_jobs=4):
    """
    Process phenological shape of remote sensing data by
    folding data to day-of-the-year. Process is done in a block-by-block way
    with parallel processing.

    Parameters
    ----------
    - inData: Absolute path to the original timeseries data
    - dates: Dates of the original timeseries data [dtype: datetime64[ns]]
    - outData: Absolute path with extention to save raster on disk
    - rollWindow: integer with value of avarage smoothing of linear trend [default None]
    - nGS: Number of observations to predict the PhenoShape [default 46: one per week]
    - chuckSize: Size of the raster chunks that would be load to memory each
                time. Needs to be multiple of 16.
    """
    # call _getPheno2 function to loal
    do_work = partial(_getPheno2, dates=dates, nGS=nGS, nan_replace=nan_replace,
        rollWindow=rollWindow)
    # apply PhenoShape with parallel processing
    try:
        _parallel_process(inData, outData, do_work, nGS, n_jobs, chuckSize)
    except AttributeError:
        print('ERROR in parallel processin...')

# ---------------------------------------------------------------------------#

def PhenoLSP(inData, outData, nGS=46, min_sep=23, n_jobs=4, chuckSize=256):
    """
    Obtain land surfurface phenology metrics for a PhenoShape product

    outputs
    -------
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
    """

    outnames = ['SOS - DOY of Start of season',
                'POS - DOY of Peak of season',
                'EOS - DOY of End of season',
                'vSOS - Vaues of start os season',
                'vPOS - Values of peak of season',
                'vEOS - Values of end of season',
                'LOS - Length of season',
                'AOS - Amplitude od season',
                'IOS - Integral of season',
                'ROG - Rate of greening',
                'ROS - Rate of senescence']

    # call _getPheno2 function to loal
    do_work = partial(_cal_LSP, min_sep=min_sep, nGS=nGS)
    # apply PhenoShape with parallel processing
    try:
        _parallel_process(inData, outData, do_work, nGS, n_jobs, chuckSize)
    except AttributeError:
        print('ERROR in parallel processin...')



###############################################################################
# Utility functions
###############################################################################

def _parallel_process(inData, outData, do_work, count,  n_jobs, chuckSize):
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


def _getPheno(y, x, nGS):
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
            xnew = np.linspace(1, 365, nGS, dtype='int16')
            ynew = np.interp(xnew, x, y)
        else:
            xnew = np.linspace(1, 365, nGS, dtype='int16')
            ynew = np.interp(xnew, x, y)

    return ynew

def _getPheno2(dstack, dates, nGS, nan_replace, rollWindow):
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
    return np.apply_along_axis(_getPheno, 0, y, x, nGS)

def _detect_peaks(spec, min_sep, valley=False):
    """
    Detects peaks.

    Parameters
    ----------
    spec : 1D array
        The specrum to analyze.

    valley : bool
        Whether to search for peaks (positive) or valleys (negative).
        Default: False

    Returns
    -------
    1D array_like
        indeces of the peaks in `spec`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
        ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    """
    # set default values
    thresh = 0.0
    edge = 'rising'
    kpsh = False
    # can't do any work if there are less than 3 points to work with
    if spec.size < 3:
        return np.array([], dtype=int)
    # if we are looking for valleys, then invert the spectra
    if valley:
        spec = -spec
    # find indexes of all peaks
    dx = spec[1:] - spec[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(spec))[0]
    if indnan.size:
        spec[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.invert(np.in1d(ind, np.unique(
            np.hstack((indnan, indnan - 1, indnan + 1)))))]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == spec.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and thresh is not None:
        ind = ind[spec[ind] >= thresh]
    # detect small peaks closer than minimum peak distance
    if ind.size and min_sep > 1:
        ind = ind[np.argsort(spec[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - min_sep) & (ind <= ind[i] + min_sep) \
                    & (spec[ind[i]] > spec[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _LSP(y, x, min_sep):
    """
    Obtain land surfurface phenology metrics

    outputs
    -------
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
    """
    # set general data
    num = 11  # number of variables to output

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
            SOS = x[peak_min[0]]  # start of season
            POS = x[peak_max]  # peak of season
            EOS = x[peak_min[1]]  # end of season
            vSOS = y[peak_min[0]]  # value at start of season
            vPOS = y[peak_max]  # value at peak of season
            vEOS = y[peak_min[1]]  # value at end of season
            LOS = EOS - SOS  # length of season
            AOS = y[peak_max] - np.min(y[peak_min])  # amplitude of season
            green = x[(x > SOS) & (x < EOS)]  # get intergral of green season
            if vPOS < vSOS or vPOS < vEOS:
                IOS = np.nan
            else:
                id = []
                for i in range(len(green)):
                    id.append((x == green[i]).nonzero()[0])
                id = np.array([item for sublist in id for item in sublist])
                IOS = simps(y[id], x[id])  # integral of season
            # rate of greening [slope SOS-POS]
            if vPOS == np.nan or vSOS == np.nan or EOS == np.nan or POS == np.nan:
                ROG = np.nan
            else:
                ROG = (vPOS - vSOS) / (POS - SOS)
            # rate of senescence [slope POS-EOS]
            if vEOS == np.nan or vPOS == np.nan or EOS == np.nan or POS == np.nan:
                ROG = np.nan
            else:
                ROS = (vEOS - vPOS) / (EOS - POS)

            return np.array((SOS, POS, EOS, vSOS, vPOS, vEOS, LOS, AOS, IOS, ROG, ROS))
        except IndexError:
            return np.repeat(np.nan, num)

def _cal_LSP(dstack, min_sep, nGS):

    # prepare input data for _LSP
    x = np.linspace(1, 365, nGS, dtype='int16')
    # estimate LSP metrics along the 0 axis
    return np.apply_along_axis(_LSP, 0, dstack, x, nGS)
