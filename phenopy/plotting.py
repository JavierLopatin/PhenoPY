import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.integrate import trapz
from scipy.interpolate import Rbf, interp1d
from scipy.stats import skew
from sklearn.metrics import mean_squared_error

#import all function from utils.py
from utils import _getPheno0, _getPheno2D, _parseLSP, _getLSPmetrics2, _rmse

def PhenoPlot(stack, X, Y, interpolType='linear', saveFigure=None, ylim=None, rollWindow=None,
              nan_replace=None, correctionValue=None, plotType=1, phentype=1, nGS=52, 
              fontsize=14, titlesize=15, legendsize=15, labelsize=13, threshold=300, ylab='NDVI', ax=None):
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
    - interpolType = String or Integer
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
    
    # Get the values for the specified X and Y
    doy=stack.doy.values#.where(img.year == 2017, drop=True).doy.values
    dates=stack.time#.where(img.year == 2017, drop=True).time
    #sorted_indices = np.argsort(doy)
    # Reorder the time dimension using the sorted indices
    #stack = stack.isel(time=sorted_indices)
    # Get the values for the specified X and Y   
    valuesTSS = stack.sel(x=X, y=Y, method="nearest").values

    # create a DataFrame for further processing
    valuesTSSpd = pd.DataFrame({
        'dates': dates,
        'doy': dates.dt.dayofyear,
        'year': dates.dt.year,
        'VI': valuesTSS
    }).sort_values('doy')

    # group values according to year
    groups = valuesTSSpd.groupby('year')
    # get phenological shape
    phen = _getPheno0(y=valuesTSS, doy=doy, interpolType=interpolType, 
                        nan_replace=None, rollWindow=rollWindow, nGS=nGS)
    # doy of the predicted phenological shape
    xnew = np.linspace(np.min(valuesTSSpd.doy), np.max(valuesTSSpd.doy), nGS,
                       dtype='int16')
   
    # Check if ax is provided, if not create one
    if ax is None:
        fig, ax = plt.subplots()

    # plot
    if plotType == 1:
        for name, group in groups:
            ax.plot(group.doy, group.VI, marker='o', linestyle='', ms=10, label=name)
        ax.plot(xnew, phen, '-', color='black')
        ax.legend(prop={'size': legendsize})
        ax.tick_params(labelsize=labelsize)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        ax.set_ylabel(ylab, fontsize=fontsize)
        ax.set_xlabel('Day of the year', fontsize=fontsize)
 
    elif plotType == 2:
        # get position of SOS, POS, and EOS
        metrics = _getLSPmetrics2(phen, xnew, nGS, len(xnew), phentype)
        isos = np.where(xnew == metrics[0])[0][0]
        ipos = np.where(xnew == metrics[1])[0][0]
        ieos = np.where(xnew == metrics[2])[0][0]
        ax.plot(xnew, phen, '-', color='black')
        ax.plot(xnew[isos], phen[isos], 'X', markersize=15, label='SOS')
        ax.plot(xnew[ipos], phen[ipos], 'X', markersize=15, label='POS')
        ax.plot(xnew[ieos], phen[ieos], 'X', markersize=15, label='EOS')
        
        ax.legend(prop={'size': 12})
        ax.tick_params(labelsize=labelsize)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(ylab, fontsize=fontsize)
        ax.set_xlabel('Day of the year', fontsize=fontsize)
        
   
    return ax     
        
        
        
def plot_with_southern_doy(shape, coordinates, ylabel='NDVI', title=None):
    """
    Plot the data from a specified shape with x-ticks reordered to represent real 
    southern hemispherical day-of-the-year (DOY) values.
    
    Parameters:
    - shape: xarray.DataArray 
        The data shape to plot.
    - coordinates: tuple
        The coordinates of the pixel to plot.
    - ylabel: str, optional
        vegetation index used in the analysis. Default is'NDVI
    - title: str, optional
        The title for the plot. Default is None, meaning no title.
    
    Returns:
    - ax: matplotlib.axes._subplots.AxesSubplot
        The plotted axis.
    """

    # Plot the data with real southern hemispherical doys
    X=coordinates[0]
    Y=coordinates[1]
    shape.sel(x=X, y=Y, method="nearest").plot()

    # Fetch the current active axis using Matplotlib's gca
    ax = plt.gca()

    # Get the original x-ticks
    original_ticks = ax.get_xticks()

    # Generate reordered labels based on the original ticks
    doy1 = np.linspace(183, 365, len(original_ticks) // 2, dtype=int)
    doy2 = np.linspace(1, 182, len(original_ticks) // 2, dtype=int)
    doy3 = np.concatenate((doy1, doy2))

    # Set the new x-tick labels
    ax.set_xticks(original_ticks)
    ax.set_xticklabels(doy3)

    # If a title is provided, set it
    if title:
        plt.title(title)
    # add y label
    plt.ylabel(ylabel)

    return ax
