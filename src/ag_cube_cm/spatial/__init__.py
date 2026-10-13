"""ag_cube_cm.spatial — Lazy spatial data access, raster ops, orchestration."""
from .data import SpatialData, _open_dataset
from .raster_ops import (
    clip_to_bbox,
    reproject_dataset,
    mask_with_geometry,
    get_roi_data,
    set_encoding,
    check_crs_in_dataset,
    get_boundaries_from_shapefile,
)
from .reporter import SpatialReporter
from .spatial_cm import SimulationResult, SpatialCM

__all__ = [
    # orchestration
    "SpatialCM",
    "SimulationResult",
    "SpatialReporter",
    # data access
    "SpatialData",
    "_open_dataset",
    # raster operations
    "clip_to_bbox",
    "reproject_dataset",
    "mask_with_geometry",
    "get_roi_data",
    "set_encoding",
    "check_crs_in_dataset",
    "get_boundaries_from_shapefile",
]
