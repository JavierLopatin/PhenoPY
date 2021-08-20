import numpy as np
import xarray as xr
from ..phenopy import _getPheno0
from phenoXr import _getLSPmetrics2

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
    

def _parseLSP(dstack, xnew, nGS, num, phentype):
    # estimate LSP metrics along the 0 axis
    # num=len(bandNames) = 16
    ans = np.apply_along_axis(_getLSPmetrics2, 0, dstack, xnew, nGS, num, phentype)
            
    coords_ = {'time': range(1, 17), # TODO: transform to bands?? remove hardcoded
              'y': dstack['y'],
              'x': dstack['x']}
    ans_ = xr.DataArray(ans, 
                        coords=coords_, 
                        dims=dstack.dims)
    print(ans.shape)
    return ans