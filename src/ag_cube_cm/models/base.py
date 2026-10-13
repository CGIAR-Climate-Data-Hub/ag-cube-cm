"""
ag_cube_cm.models.base
=======================

Abstract Base Class for Crop Models.
"""

import abc
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import xarray as xr

from ag_cube_cm.config.schemas import SimulationConfig
from ag_cube_cm.models._fileops import rmtree_retry


class CropModel(abc.ABC):
    """
    Abstract Base Class for all crop models (DSSAT, CAF2021, SIMPLE).
    
    This class defines the mandatory interface that the Spatial Orchestrator
    uses to run simulations across pixels.
    """

    #: Key in :meth:`collect_outputs`' return dict holding the primary result
    #: that :class:`~ag_cube_cm.spatial.spatial_cm.SpatialCM` grids into the
    #: output cube.  DSSAT reports grain yield at maturity as ``HWAM``; other
    #: models override this with their own headline variable.
    YIELD_VARIABLE: str = "HWAM"

    #: Units of :attr:`YIELD_VARIABLE`, written to the output cube.
    YIELD_UNITS: str = "kg/ha"

    #: Long name of :attr:`YIELD_VARIABLE`, written to the output cube.
    YIELD_LONG_NAME: str = "Mean grain yield at maturity"

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.working_dir: Path | None = None
        self.pixel_id: str | None = None

    def setup_working_directory(self, pixel_id: str) -> Path:
        """
        Create a unique, isolated temporary directory for this pixel's run.
        
        Using UUIDs prevents race conditions during parallel execution where
        multiple threads might try to read/write the same intermediate files.
        """
        self.pixel_id = str(pixel_id)
        base_tmp = Path(getattr(self.config.GENERAL_INFO, 'working_path', './tmp')) / "runs"
        base_tmp.mkdir(parents=True, exist_ok=True)
        self.working_dir = base_tmp / self.pixel_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        return self.working_dir

    def cleanup_working_directory(self) -> None:
        """Remove the isolated working directory and all its contents.

        Uses a retrying remove: on Windows the model binary may still hold a
        handle for a moment after the subprocess returns, and DSSAT writes some
        outputs read-only. Never raises — a directory left on disk is a
        disk-space problem, not a reason to fail a good simulation.
        """
        if self.working_dir and self.working_dir.exists():
            rmtree_retry(self.working_dir)
            self.working_dir = None

    @abc.abstractmethod
    def prepare_inputs(self, weather_slice: xr.Dataset, soil_slice: xr.Dataset) -> None:
        """
        Convert Xarray data slices into the format required by the model.
        
        For binary models (like DSSAT), this method writes specific text files
        (.WTH, .SOL, etc.) to `self.working_dir`. For pure-Python models, 
        this might just store the data arrays in memory.
        """
        pass

    @abc.abstractmethod
    def run_simulation(self) -> None:
        """
        Trigger the model execution.
        
        For binary models, this executes a subprocess in `self.working_dir`.
        For Python models, this calls the model functions directly.
        """
        pass

    @abc.abstractmethod
    def collect_outputs(self) -> pd.DataFrame | Dict[str, Any]:
        """
        Collect results after a successful simulation.
        
        Returns a DataFrame or Dictionary containing the yield and date variables
        which the orchestrator will assemble into the final datacube.
        """
        pass
