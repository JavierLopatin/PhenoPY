import numpy as np
import xarray as xr
from functools import reduce
from phenoxr.pheno import _getLSPmetrics2
from phenoxr.utils import _getPheno2D, _parseLSP, _rmse, computeChunkSize
from phenopy import _getPheno0


@xr.register_dataarray_accessor("pheno")
class Pheno:
    def __init__(self, xr_obj):
        self._obj = xr_obj
        self.kwargs = {}
        self.LSP_bands = ['sos', 'pos', 'eos', 'vsos', 'vpos', 
                          'veos', 'los', 'msp', 'mau', 'vmsp', 
                          'vmau', 'ampl', 'ios', 'rog', 'ros', 'sw']
       
    def PhenoShape(self, doy: list=None, interpolType: str='linear', nan_replace: np.float64=None,
                   rollWindow: int=5, nGS: int=52, chunk_size: dict = None):
        """
        Apply the _getPheno2D/_getPheno2 function to a xarray.DataArray object. It calculates all the necessary auxiliary objects in order to use Dask functionality (trough map_blocks).
        
        :param doy: day of the year. If not present, it will attempt to extract the doy from the time dimension.
        :param interpolType: interpolation type
        :param nan_replace: numeric value to be replaced by np.nan over the computations.
        :param rollWindow: rolling window size
        :param nGS: number of periods per year, usually 52 (number of weeks in a year)
        :param chunk_size: a dictionary with the chunk size for Dask, in the form {'x': a, 'y': b, 'z': c}
        
        :returns: computed xarray.DataArray
        """
        stack = self._obj
        if doy is None:
            doy = stack.time.dt.dayofyear.values
        # replicating what happens in _getPheno xnew definition
        xnew = np.linspace(np.min(doy), np.max(doy), nGS, dtype='int16')
        # TODO: change hemisfere, start doy at the desired day (1 north, 182 south) and keep record about the original doy -> dos (day of season)
        
        
        if chunk_size is None:
            chunk_size = {'x': 200, 'y': 200, 'time': len(stack.time)} # TODO: define a function to auto calculate next chunk (to ~100MB each chunk)
        
        coords_ = {'time': xnew,
                  'y': stack.coords['y'],
                  'x': stack.coords['x']}
        template_ = xr.DataArray(np.zeros((nGS, len(stack.y), len(stack.x))), 
                                 coords=coords_,
                                 dims = ['time', 'y', 'x']).chunk(chunk_size)
        
        stack = stack.chunk(chunk_size)
        kwargs_ = {'doy': doy, 'interpolType': interpolType,  
                   'nan_replace': nan_replace, 'rollWindow': rollWindow, 
                   'nGS': nGS, 'xnew': xnew}
        
        stackP = stack.map_blocks(_getPheno2D, kwargs=kwargs_, template=template_).rename({'time': 'doy'})
        stackP.pheno.kwargs['computePheno'] = kwargs_
        
        return stackP
    
    def PhenoLSP(self, nGS=None, phentype=1):
        # it seems n_phen is not used
        """
        Obtain land surface phenology metrics for a PhenoShape product

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
            raise('No Pheno computed')  # TODO: replace with auto-compute?
        
        n_ = len(self.LSP_bands)
        stack = self._obj
        xnew = self.kwargs['computePheno']['xnew']
        chunk_size = [i for i in stack.chunks]
        chunk_size[0] = n_
        if nGS is None:
            nGS = self.kwargs['computePheno']['nGS']
        
        kwargs_ = {'xnew': xnew, 'nGS': nGS, 'bands': self.LSP_bands, 'phentype': phentype}
        
        coords_ = {'doy': self.LSP_bands,
                   'y': stack.coords['y'],
                   'x': stack.coords['x']}
        template_ = xr.DataArray(np.zeros((n_, len(stack.y), len(stack.x))), 
                                 coords=coords_,
                                 dims = ['doy', 'y', 'x']).chunk(chunk_size)
        
        stackP = stack.map_blocks(_parseLSP, kwargs=kwargs_, template=template_).rename({'doy': 'LSP_bands'})
        stackP.pheno.kwargs['computePhenoLSP'] = kwargs_  # this doesn't work, is not saved, pheno objet its lost in datadaset transformation

        return stackP.to_dataset('LSP_bands')
    
    def RMSE(self, original_stack, LSP_stack=None, normalized=False, nan_replace=None, interpolate_nans=False):
        """
        Calculate the RMSE of the PhenoShape estimation; it can also do it by section, using the sos, pos and eos from the LSP computation (if provided).
        
        :param original_stack: initial image stack, from which the phenoShape was calculated.
        :param LSP_stack: LSP computed stack.
        :param normalized: boolean, if True RMSE will be scaled to [0, 1].
        :param nan_replace: values to be converted to NaN.
        :param interpolate_nans: boolean, should NaNs values be interpolated?
        
        :returns: computed xarray.DataArray with the RMSE
        """
        # phen = ans.copy(); original_stack=ndvi.copy(); LSP_stack = ans2.copy()
        phen = self._obj # inShape, phen  || # original_stack = inData = dstack
        
        # 1. Check if I'm PhenoShape data
        if 'computePheno' not in self.kwargs:
            raise('It seems computePheno has not yet been computed...')
        
        if nan_replace is not None:
            original_stack = original_stack.where(original_stack.values != nan_replace)
        
        # 2. Get day of the year of original_stack and reorder by that
        doys = original_stack.time.dt.dayofyear.values
        sdoys = sorted(doys)
        original_stack = original_stack.assign_coords(time=doys)
        original_stack = original_stack.rename({'time': 'doy'}).sortby('doy')
        
        # 3. Linear interpolation for pheno, to match original_stack doys
        if interpolate_nans:
            phen = phen.interpolate_na('doy')
            
        phen = phen.interp(doy=sdoys, method='linear')
        
        # 4. RMSE
        rmse = _rmse(phen, original_stack, 'doy', normalized)
        if LSP_stack is None:
            return rmse
        else:
            sos = LSP_stack['sos'] # .expand_dims({'doy': sdoys})
            pos = LSP_stack['pos']
            eos = LSP_stack['eos']
            
            x_ = len(original_stack.coords['x'])
            y_ = len(original_stack.coords['y'])
            temp_ = xr.DataArray(data = np.repeat(sdoys, x_*y_).reshape(len(sdoys), y_, x_),
                                 dims = original_stack.dims, 
                                 coords = original_stack.coords,
                                 attrs = original_stack.attrs).chunk(phen.chunks)
            
            sosm = phen.where(temp_ >= sos)
            posm = phen.where((temp_ < sos) & (temp_ > eos))
            eosm = phen.where(temp_ <= eos)
            
            # TODO: take into account the option of a persist option (to persist here).
            return {'rmse': rmse, 
                    'rmse_sos': _rmse(sosm, original_stack, 'doy', normalized),
                    'rmse_pos': _rmse(posm, original_stack, 'doy', normalized),
                    'rmse_eos': _rmse(eosm, original_stack, 'doy', normalized)}
    
    def PhenoPlot(self):
        # TODO: original data vs PhenoShape, coordinates or position as input to plot [option to use ipyleaflet to select a point or another kind of interaction]
        pass
