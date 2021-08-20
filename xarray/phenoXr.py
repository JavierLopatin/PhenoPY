import numpy as np
import xarray as xr
from functools import reduce
from ..phenopy import _getPheno0


def _getPheno2D(dstack, doy, interpolType, nan_replace, rollWindow, nGS, xnew=None):
    # dstack.doy
    ans = np.apply_along_axis(_getPheno0, 0, dstack, doy, interpolType, nan_replace, rollWindow, nGS)
    
    # TODO: ¿_getPheno0 cambia el orden del arreglo? si es así, debo corregir - DONE?
    # TODO: retornar día del año modificado, eliminar time/year, usar xnew (nuevo doy) - DONE!
    # Esto se llama PhenoShape
    
    if xnew is None:
        xnew = range(1, nGS + 1)
        
    coords_ = {'time': xnew,
              'y': dstack['y'],
              'x': dstack['x']}
    ans_ = xr.DataArray(ans, 
                        coords=coords_, 
                        dims=dstack.dims)
    return ans_
    

@xr.register_dataarray_accessor("pheno")
class Pheno:
    def __init__(self, xr_obj):
        self._obj = xr_obj
        self.kwargs = {}
    
    def prepare_data(self, dimensions=['x', 'y', 'time'], *args, **kwargs):
        """
        :param stack: must be a DataArray objetc (one variable only)
        :param dimensions: a list indicating the name of the X, Y, and Z dimension. Z being usually time.
        :param args: extra arguments passed to _getPheno2D
        :param kwargs: extra arguments passed to _getPheno2D
        """
        return data, template
    
    def computeChunkSize(self, sizeMB=100, Z='time'):
        """
        PENDING!
        :param sizeMB: aprox desired chunk size in MB.
        :param Z: name of the Z axis. By default, 'time'.
        """
        bmod = self._obj.dtype.itemsize
        shape = self._obj.shape
        if len(shape) != 3:
            raise(f'DataArray dimensions should be 3, not {shape}')
        total_sizeMB = reduce(lambda x, y: x*y, shape) / 1000**2 * bmod
        if total_sizeMB >= sizeMB:
            pass
        else:
            chunk = dict(zip(self._obj.dims, shape))
        return chunk

    def computePheno(self, doy=None, interpolType='linear', nan_replace=None,
                    rollWindow=5, nGS=52):
        """
        Apply the _getPheno2D/_getPheno2 function to a xarray.DataArray object. It calculates all the necessary auxiliary objects in order to use Dask functionality (trough map_blocks).
        
        :param doy: day of the year. If not present, it will attempt to extract the doy from the time dimension.
        :param interpolType: interpolation type
        :param nan_replace: what to do with NaNs
        :param rollWindow: rolling window size
        :param nGS: number of periods per year, usually 52 (number of weeks in a year)
        
        :returns: computed xarray.DataArray
        """
        stack = self._obj
        if doy is None:
            doy = stack.time.dt.dayofyear.values
        # replicating what happens in _getPheno xnew definition
        xnew = np.linspace(np.min(doy), np.max(doy), nGS, dtype='int16')
        
        # TODO: define a function to auto calculate next chunk (to ~100MB each chunk)
        time_chunk = {'x': 10, 'y': 10, 'time': len(stack.time)}
        
        coords_ = {'time': xnew,
                  'y': stack.coords['y'],
                  'x': stack.coords['x']}
        template_ = xr.DataArray(np.zeros((nGS, len(stack.y), len(stack.x))), 
                                 coords=coords_,
                                 dims = ['time', 'y', 'x']).chunk(time_chunk)
        
        stack = stack.chunk(time_chunk)
        kwargs_ = {'doy': doy, 'interpolType': interpolType,  
                   'nan_replace': nan_replace, 'rollWindow': rollWindow, 
                   'nGS': nGS, 'xnew': xnew}
        
        stackP = stack.map_blocks(_getPheno2D, kwargs=kwargs_, template=template_)
        stackP.pheno.kwargs['computePheno'] = kwargs_
        
        return stackP
    
    def computePhenoLSP(self, nGS=None, phentype=1):
        # TODO: modificar PhenoLSP para trabajar con esta salida y generar una salida total (sin tiempo), de diferentes bandas
#         def PhenoLSP(inData, outData, doy, nGS=46, phentype=1, n_phen=10, n_jobs=4,
#              chuckSize=256):
#       it seems n_phen is not used
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
        - phenType: Type os estimation of SOS and EOS. 1 = median value between POS and start and end of season. 2 = using the knee inflexion method. default 1
        - n_phen: Integer
            Window size where to estimate SOS and EOS
        """
        if 'computePheno' not in self.kwargs:
            raise('No Pheno computed')  # TODO: replace with auto-compute
        
        stack = self._obj
        xnew = self.kwargs['computePheno']['xnew']
        time_chunk = [i for i in stack.chunks]
        time_chunk[0] = 16
        if nGS is None:
            nGS = self.kwargs['computePheno']['nGS']
        
        kwargs_ = {'xnew': xnew, 'nGS': nGS, 'num': 16, 'phentype': phentype}
        
        coords_ = {'time': range(1, 17), # TODO: transform to bands??
                   'y': stack.coords['y'],
                   'x': stack.coords['x']}
        template_ = xr.DataArray(np.zeros((16, len(stack.y), len(stack.x))), 
                                 coords=coords_,
                                 dims = ['time', 'y', 'x']).chunk(time_chunk)
            
        stackP = stack.map_blocks(_parseLSP, kwargs=kwargs_, template=template_)
        stackP.pheno.kwargs['computePhenoLSP'] = kwargs_

        return stackP
    

def _parseLSP(dstack, xnew, nGS, num, phentype):
    # estimate LSP metrics along the 0 axis
    # num=len(bandNames) = 16
    ans = np.apply_along_axis(_getLSPmetrics2, 0, dstack, xnew, nGS, num, phentype)
            
    coords_ = {'time': range(1, 17), # TODO: transform to bands??
              'y': dstack['y'],
              'x': dstack['x']}
    ans_ = xr.DataArray(ans, 
                        coords=coords_, 
                        dims=dstack.dims)
    print(ans.shape)
    return ans


from scipy.integrate import trapz
from scipy.interpolate import Rbf, interp1d
from scipy.stats import skew
from sklearn.metrics import mean_squared_error

def _getLSPmetrics2(phen, xnew, nGS, num, phentype):
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
        # basic variables
        vpos = np.max(phen)
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

        # los: length of season
        los = eos - sos
        if los < 0:
            los[los < 0] = len(phen) + \
                (eos[los < 0] - sos[los < 0])

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
        id = []
        for i in range(len(green)):
            id.append((xnew == green[i]).nonzero()[0])
        # index of growing season
        id = np.array([item for sublist in id for item in sublist])

        # get intergral of green season
        ios = trapz(phen[id], xnew[id])

        # rate of greening [slope SOS-POS]
        rog = (vpos - phen[isos]) / (pos - sos)

        # rate of senescence [slope POS-EOS]
        ros = (phen[ieos] - vpos) / (eos - pos)

        # skewness of growing season
        sw = skew(phen[id])

        metrics = np.array((sos, pos[0], eos, phen[isos][0], vpos,
                            phen[ieos][0], los, msp, mau, vmsp, vmau, ampl, ios, rog[0],
                            ros[0], sw))

        return metrics