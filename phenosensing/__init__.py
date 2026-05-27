"""PhenoSensing — land surface phenology metrics from satellite image time series.

Importing this package registers the ``pheno`` accessor on
``xarray.DataArray`` objects, so that ``da.pheno.PhenoShape(...)`` and the
other phenology methods become available.
"""

# Importing the accessor module runs the ``@xr.register_dataarray_accessor``
# decorator as a side effect, registering the ``pheno`` namespace.
from .accessor import Pheno  # noqa: F401
from .anomaly import anomaly  # noqa: F401
from .curvature import classify_vector_numeric, get_curvature  # noqa: F401
from .extraction import list_extractors  # noqa: F401
from .io import list_samples, load_sample  # noqa: F401
from .phase import season_phase  # noqa: F401
from .qa import list_qa_specs, qa_to_weight  # noqa: F401
from .reconstruction import list_reconstructors  # noqa: F401
from .season import n_seasons  # noqa: F401
from .trends import trend  # noqa: F401
from .uncertainty import uncertainty  # noqa: F401
from .utils import reorder_southern_hemisphere  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "Pheno",
    "get_curvature",
    "classify_vector_numeric",
    "reorder_southern_hemisphere",
    "load_sample",
    "list_samples",
    "list_reconstructors",
    "list_extractors",
    "trend",
    "anomaly",
    "n_seasons",
    "season_phase",
    "uncertainty",
    "qa_to_weight",
    "list_qa_specs",
    "__version__",
]
