"""
ag_cube_cm — Spatial NetCDF Datacube Crop Model Orchestrator
=============================================================

Public package API. Imports are intentionally minimal at the top level;
heavy sub-packages (models, spatial) are loaded on first access to keep
cold-start import time low.

Example
-------
>>> import ag_cube_cm
>>> print(ag_cube_cm.__version__)
'0.1.0'
"""

from __future__ import annotations

__version__: str = "0.1.0"
__author__: str = "Andres Aguilar"
__email__: str = "andres.aguilar@cgiar.org"
__license__: str = "MIT"

# Expose the two most commonly used entry points at the package level so that
# users can do `from ag_cube_cm import SpatialData, SpatialCM` without having
# to know the internal module path.
#
# These are imported lazily via __getattr__ to avoid pulling in heavy
# transitive dependencies (xarray, dask, geopandas …) at import time.

_LAZY_IMPORTS: dict[str, str] = {
    "SpatialData": "ag_cube_cm.spatial.data",
    "SpatialCM": "ag_cube_cm.spatial.spatial_cm",
    "SimulationResult": "ag_cube_cm.spatial.spatial_cm",
    "SimulationConfig": "ag_cube_cm.config.schemas",
    "load_config": "ag_cube_cm.config.loader",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        obj = getattr(module, name)
        # Cache on the package so subsequent accesses skip __getattr__
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'ag_cube_cm' has no attribute {name!r}")


__all__: list[str] = [
    "__version__",
    "SpatialData",
    "SpatialCM",
    "SimulationResult",
    "SimulationConfig",
    "load_config",
]
