import numpy as np
import xarray as xr
from functools import reduce
from phenoxr.pheno import _getLSPmetrics2
from phenoxr.utils import _getPheno2D, _parseLSP
from phenopy import _getPheno0


@xr.register_dataarray_accessor("pheno")
class Pheno:
    def __init__(self, xr_obj):
        self._obj = xr_obj
        self.kwargs = {}
    
    def prepare_data(self, dimensions=['x', 'y', 'time'], *args, **kwargs):
        """
        TODO: PENDING!
        :param stack: must be a DataArray objetc (one variable only)
        :param dimensions: a list indicating the name of the X, Y, and Z dimension. Z being usually time.
        :param args: extra arguments passed to _getPheno2D
        :param kwargs: extra arguments passed to _getPheno2D
        """
        pass  # return data, template
    
    def computeChunkSize(self, sizeMB=100, Z='time'):
        """
        TODO: PENDING!
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

    def PhenoShape(self, doy=None, interpolType='linear', nan_replace=None,
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
        # TODO: change hemisfere, start doy at the desired day (1 north, 182 south) and keep record about the original doy
        
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
    
    def PhenoLSP(self, nGS=None, phentype=1):
        # it seems n_phen is not used
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
        
        n_ = 16  # TODO: replace with auto-compute or user input
        stack = self._obj
        xnew = self.kwargs['computePheno']['xnew']
        time_chunk = [i for i in stack.chunks]
        time_chunk[0] = n_
        if nGS is None:
            nGS = self.kwargs['computePheno']['nGS']
        
        kwargs_ = {'xnew': xnew, 'nGS': nGS, 'num': n_, 'phentype': phentype}
        
        coords_ = {'time': range(1, n_+ 1),
                   'y': stack.coords['y'],
                   'x': stack.coords['x']}
        template_ = xr.DataArray(np.zeros((n_, len(stack.y), len(stack.x))), 
                                 coords=coords_,
                                 dims = ['time', 'y', 'x']).chunk(time_chunk)
        
        # TODO: return a Dataset instead of DataArray
        stackP = stack.map_blocks(_parseLSP, kwargs=kwargs_, template=template_)
        stackP.pheno.kwargs['computePhenoLSP'] = kwargs_

        return stackP
    
    def RMSE(self):
        # TODO: implement. Works with original_data and PhenoShape output
        pass
    
    def PhenoPlot(self):
        # TODO: original data vs PhenoShape, coordinates or position as input to plot [option to use ipyleaflet to select a point or another kind of interaction]
        pass
