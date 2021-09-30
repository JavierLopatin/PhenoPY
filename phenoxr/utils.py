import numpy as np
import xarray as xr
from phenopy import _getPheno0
from phenoxr.pheno import _getLSPmetrics2

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


def _rmse(computed_stack, original_stack, axis='doy', normalized=False):
    # TODO: N should be the number of valid data or the length? should be the first one I think
    N = len(computed_stack.coords[axis])
    rmse = (((original_stack - computed_stack)**2).sum(axis, keep_attrs=True, skipna=True) / N) ** 1/2
    if normalized:
        minn = original_stack.min(axis, skipna=True)
        maxx = original_stack.max(axis, skipna=True)
        return rmse/(maxx-minn)
    else:
        return rmse


def computeChunkSize(arr, sizeMB=100, Z='time'):
    """
    TODO: PENDING!
    :param sizeMB: aprox desired chunk size in MB.
    :param Z: name of the Z axis. By default, 'time'.
    """
    bmod = arr.dtype.itemsize
    shape = arr.shape
    if len(shape) != 3:
        raise(f'DataArray dimensions should be 3, not {shape}')
    total_sizeMB = reduce(lambda x, y: x*y, shape) / 1000**2 * bmod
    if total_sizeMB >= sizeMB:
        pass
    else:
        chunk = dict(zip(arr.dims, shape))
    return chunk