"""PhenoPY — land surface phenology metrics from satellite image time series.

Importing this package registers the ``pheno`` accessor on
``xarray.DataArray`` objects, so that ``da.pheno.PhenoShape(...)`` and the
other phenology methods become available.
"""

# Importing the accessor module runs the ``@xr.register_dataarray_accessor``
# decorator as a side effect, registering the ``pheno`` namespace.
from .phenopy import Pheno  # noqa: F401
from .curvature import get_curvature, classify_vector_numeric  # noqa: F401
from .utils import reorder_southern_hemisphere  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "Pheno",
    "get_curvature",
    "classify_vector_numeric",
    "reorder_southern_hemisphere",
    "__version__",
]
