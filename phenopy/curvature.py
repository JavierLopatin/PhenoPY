import numpy as np
import xarray as xr

def classify_vector_numeric(v: np.ndarray) -> float:
    """
    Calculate and return the sum of the second difference of a vector.
    
    Parameters:
    - v : numpy ndarray
        Input vector.

    Returns:
    - float
        Sum of the second difference.
    """
    return np.sum(np.diff(v, n=2))

def get_curvature(phenoshape: xr.DataArray) -> xr.DataArray:
    """
    Calculate the curvature for each point in a phenoshape xarray.
    
    Parameters:
    - phenoshape : xarray DataArray
        Input data array which should have 'doy' as a coordinate or a dimension.

    Returns:
    - xarray DataArray
        Curvature values.
    """
    # Check if 'doy' is a dimension
    if 'doy' not in phenoshape.dims:
        if 'doy' in phenoshape.coords and 'time' in phenoshape.dims:
            # Temporarily swap 'time' dimension with 'doy' coordinate for the operation
            phenoshape = phenoshape.swap_dims({'time': 'doy'})
        else:
            raise ValueError("The provided xarray DataArray must have 'doy' as a coordinate or a dimension.")
    
    curvature = xr.apply_ufunc(
        classify_vector_numeric,
        phenoshape,
        input_core_dims=[['doy']],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[phenoshape.dtype]
    )

    return curvature