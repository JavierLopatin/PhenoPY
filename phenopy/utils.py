import sys
import numpy as np
import warnings
# from kneed import KneeLocator # (for phenotype == 2, deprecated
from scipy.integrate import trapz
from scipy.interpolate import Rbf, interp1d
from scipy.stats import skew
from sklearn.metrics import mean_squared_error
import xarray as xr


def reorder_southern_hemisphere(img: xr.Dataset):
    """
    Reorder the DOY of the img for the Southern Hemisphere to ensure
    peak summer is in the middle. This reordering is critical for some applications
    like phenological studies.
    
    :param img: xarray Dataset with `time` and `doy` dimensions/coordinates.
    :return: The input xarray Dataset reordered by DOY.
    """
    doy = img.doy.values
    time_values = img.time.values

    arr1 = np.linspace(183, 365, int(365-180)).astype(int)
    arr2 = np.linspace(1, 185, 180).astype(int)
    result = np.concatenate((arr1, arr2))
    positions = [np.where(result == value)[0][0] for value in doy]

    # Create a mapping of DOY value to its position in 'result'
    position_map = {value: np.where(result == value)[0][0] for value in doy}

    # Reorder the time values based on the mapping
    reordered_times = sorted(time_values, key=lambda x: position_map[img.sel(time=x).doy.values.item()])

    # Re-index the img using the reordered time values
    da = img.sel(time=reordered_times)
    da.doy.values = np.linspace(1, 365, len(da.time.values))#np.sort(doy)#

    return positions, da


def _getPheno(y, x, nGS, interpolType):
    """
    Apply linear interpolation in the 'time' axis
    x: DOY values
    y: ndarray with VI values
    """
    inds = np.isnan(y)  # check if array has NaN values
    if np.sum(inds) == len(y):  # check if all values are NaN
        return y[0:nGS]
    else:
        try:
            xnew = np.linspace(np.min(x), np.max(x), nGS, dtype=np.int32)
            if inds.any():  # if inds have at least one True
                y = _fillNaN(y)
                _replaceElements(x)  # replace doy values when they are the same
            if interpolType == 'linear':
                ynew = np.interp(xnew, x, y)
            elif interpolType == 'RBF':
                f = Rbf(x, y, function='cubic')  # you had a typo here 'funciton' instead of 'function'
                ynew = f(xnew)
            elif interpolType == 'KDE':
                ynew = _KDE(x, y, nGS)
            else:
                f = interp1d(x, y, kind=interpolType)
                ynew = f(xnew)
            return ynew  # Moved this line to be outside the if-elif-else chain
        except Exception as e:  # It's good to handle exceptions in the try block
            print(f"An error occurred: {e}")
            return None
            
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
    phen = _getPheno(y, 
                     doy[idx],
                     #doy, 
                     nGS, 
                     interpolType)
    
    # rolling average using moving window
    if rollWindow is not None:
        phen = _moving_average(phen, rollWindow)
    
    return phen

def _getLSPmetrics2(phen, xnew, nGS, bands, phentype):
    """
    Obtain land surfurface phenology metrics

    Parameters
    ----------
    - phen: 1D array
        PhenoShape data
    - xnew: 1D array
        DOY values for PhenoShape data	
    - bands: string list
        Name of the output bands (soft requirement)

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
        return np.repeat(np.nan, len(bands))
    else:
        # basic variables
        vpos = np.max(phen)
        trough = np.min(phen)
        ampl = vpos - trough

        # get position of seasonal peak and trough
        ipos = np.where(phen == vpos)[0]
        pos = xnew[ipos]

        # scale annual time series to 0-1
        ratio = (phen - trough) / ampl

        # separate greening from senesence values
        dev = np.gradient(ratio)  # first derivative
        greenup = np.zeros([ratio.shape[0]],  dtype=bool)
        greenup[dev > 0] = True

        # select time where SOS and EOS are located (around trs value)
        # KneeLocator looks for the inflection index in the curve
        if phentype in [1, 2]:  # estimate SOS and EOS as median of the season
            if phentype == 2:
                warnings.warn("Type 2 is currently not implemented", DeprecationWarning)
            i = np.median(xnew[:ipos[0]][greenup[:ipos[0]]])
            ii = np.median(xnew[ipos[0]:][~greenup[ipos[0]:]])
            sos = xnew[(np.abs(xnew - i)).argmin()]
            eos = xnew[(np.abs(xnew - ii)).argmin()]
            isos = np.where(xnew == int(sos))[0]
            ieos = np.where(xnew == eos)[0]
#         elif phentype == 2:  # estimate SOS and EOS by inflection curves
#             #-- consider only observation before POS for SOS
#             knee1 = KneeLocator(xnew[0:ipos[0]], ratio[0:ipos[0]], S=2,
#                                 curve='convex', direction='increasing')
#             sos = knee1.knee
#             isos = np.where(xnew == knee1.knee)[0]

#             #-- consider only observation after POS for EOS
#             x = xnew[-(nGS - ipos[0] - 1):]
#             y = ratio[-(nGS - ipos[0] - 1):]
#             knee2 = KneeLocator(range(len(x)), np.flip(y), S=2,
#                                 curve='convex', direction='increasing')
#             eos = x[np.where(
#                 np.flip(range(len(x))) == knee2.knee)[0]][0]
#             ieos = np.where(xnew == eos)[0]
        else:
            print('phentype must be either 1 or 2')
        if sos is None:
            isos = 0
            sos = xnew[isos]
        if eos is None:
            ieos = len(xnew) - 1
            eos = xnew[ieos]

        # los: length of season
        los = eos - sos
        if los < 0:
            los = np.nan

        # get MSP, MAU (independent from SOS and EOS)
        
        # mean spring
        idx = np.mean(xnew[(xnew > sos) & (xnew < pos[0])])
        idx = (np.abs(xnew - idx)).argmin()  # indexing value
        msp = xnew[idx]  # DOY of MGS
        vmsp = phen[idx]  # mgs value

        # mean autum
        idx = np.mean(xnew[(xnew < eos) & (xnew > pos[0])])
        idx = (np.abs(xnew - idx)).argmin()  # indexing value
        mau = xnew[idx]  # DOY of MGS
        vmau = phen[idx]  # mgs value

        # doy of growing season
        green = xnew[(xnew > sos) & (xnew < eos)]
        id_ = []
        for i in range(len(green)):
            id_.append((xnew == green[i]).nonzero()[0])   
        # TODO: move id_ generation to a list comprehension -> id_ = [(xnew == green[i]).nonzero()[0] for i in range(len(green))]
        
        # index of growing season
        id = np.array([item for sublist in id_ for item in sublist])

        # get intergral of green season
        ios = trapz(phen[id], xnew[id]) if len(id) > 0 else np.nan
        
        # skewness of growing season
        sw = skew(phen[id]) if len(id) > 0 else np.nan

        # rate of greening [slope SOS-POS]
        rog = (vpos - phen[isos]) / (pos - sos)

        # rate of senescence [slope POS-EOS]
        ros = (phen[ieos] - vpos) / (eos - pos)

        metrics = np.array((sos, pos[0], eos, phen[isos][0], vpos,
                            phen[ieos][0], los, msp, mau, vmsp, vmau, ampl, ios, rog[0],
                            ros[0], sw))

        return metrics
    

def _getPheno2D(dstack, doy, interpolType, nan_replace, rollWindow, nGS, xnew=None):
    # dstack.doy
    ans = np.apply_along_axis(_getPheno0, 0, dstack, doy, interpolType, nan_replace, rollWindow, nGS)
    
    # TODO: ¿_getPheno0 cambia el orden del arreglo? si es así, debo corregir - DONE?
    # TODO: retornar día del año modificado, eliminar time/year, usar xnew (nuevo doy) - DONE!
    # Esto se llama PhenoShape
    
    if xnew is None:
        xnew = range(1, nGS + 1)
        
    return _assemble(ans, dstack, {'time': xnew}, True)
    

def _parseLSP(dstack, xnew, nGS, bands, phentype):
    # num=len(bandNames) = 16
    ans = np.apply_along_axis(_getLSPmetrics2, 0, dstack, xnew, nGS, bands, phentype)
    
    return _assemble(ans, dstack, {'doy': bands}, True)


def _assemble(computed_data, original_stack, z_values, asDataArray=True):
    coords_ = z_values
    coords_['y'] = original_stack['y']
    coords_['x'] = original_stack['x']
    
    if asDataArray:
        out = xr.DataArray(computed_data, 
                           coords=coords_, 
                           dims=original_stack.dims)
    else:
        pass
    
    return out


def _rmse(computed_stack, original_stack, normalized=False):
    # Compute RMSE
    N = len(computed_stack['doy'])
    squared_difference = (original_stack - computed_stack)**2
    mean_squared_error = squared_difference.sum('doy', keep_attrs=True, skipna=True) / N
    rmse = mean_squared_error**0.5

    if normalized:
        minn = original_stack.min(dim='doy', skipna=True)
        maxx = original_stack.max(dim='doy', skipna=True)
        return rmse/(maxx-minn)
    else:
        return rmse

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
    
def computeChunkSize(arr, sizeMB=100, Z='time'):
    """
    TODO: PENDING!
    :param sizeMB: aprox desired chunk size in MB.
    :param Z: name of the Z axis. By default, 'time'.
    """
    bmod = arr.dtype.itemsize
    ds = arr.shape
    if len(ds) != 3:
        raise(f'DataArray dimensions should be 3, not {ds}')
    total_sizeMB = reduce(lambda x, y: x*y, ds) / 1000**2 * bmod
    if total_sizeMB >= sizeMB:
        pass
    else:
        chunk = dict(zip(arr.dims, ds))
    return chunk

    
def _moving_average(a, n=3) :
    out = np.convolve(a, np.ones(n), 'valid') / n    
    return np.concatenate([ a[:np.int(n/2)], out, a[-np.int(n/2):] ]) # add values of tail

def _fillNaN(x):
    # Fill NaN data by linear interpolation
    mask = np.isnan(x) 
    x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    return x

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