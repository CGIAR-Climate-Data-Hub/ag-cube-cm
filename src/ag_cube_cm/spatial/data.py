"""
ag_cube_cm.spatial.data
========================

Spatial Data Manager — lazy-loading wrapper around on-disk raster datasets.

**CRITICAL MEMORY CONTRACT**
Every dataset opened by :meth:`SpatialData._open_dataset` is backed by a
**Dask graph** and occupies virtually zero RAM until ``.compute()`` or an
equivalent trigger is called.  The legacy ``xr.open_dataset(...).copy()``
anti-pattern is explicitly forbidden here; see ``_open_dataset`` for the
correct approach.

The :class:`SpatialData` class provides three lazy dataset properties:

* :attr:`~SpatialData.climate`  — multi-temporal weather NetCDF cube.
* :attr:`~SpatialData.soil`    — multi-depth SoilGrids NetCDF cube.
* :attr:`~SpatialData.dem`     — optional Digital Elevation Model raster.

All three are loaded lazily and cached as instance attributes on first
access so subsequent reads are zero-cost.
"""

from __future__ import annotations

import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _open_dataset(
    filepath: str,
    engine: str = "netcdf4",
    chunks: str | dict | None = "auto",
) -> xr.Dataset:
    """Open a raster or NetCDF file as a **lazy Dask-backed** dataset.

    This is the single canonical place where files are opened.  It enforces
    the memory contract: NO ``.copy()``, NO ``.load()``, NO ``.compute()``
    in this function.  The returned object is a Dask-backed ``xr.Dataset``
    whose arrays are only materialised when the caller explicitly triggers
    computation (e.g. via ``.compute()``, ``.values``, or saving to disk).

    Supported extensions
    --------------------
    * ``.nc``    → :func:`xarray.open_dataset` with ``chunks='auto'``
    * ``.tif``   → :func:`rioxarray.open_rasterio` with ``chunks='auto'``
    * ``.pickle`` → standard pickle (not lazy — small metadata objects only)

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the dataset file.
    engine : str
        NetCDF engine passed to :func:`xarray.open_dataset`.  Default:
        ``"netcdf4"``.  Use ``"h5netcdf"`` if netCDF4 is unavailable.
    chunks : str | dict | None
        Dask chunking.  ``"auto"`` (default) keeps the lazy memory contract.
        Pass ``None`` for a plain, un-chunked open — :class:`SpatialCM` does
        this because its pixel loop issues hundreds of thousands of tiny
        ``.sel()`` calls from a thread pool, where the Dask scheduler costs
        far more than it saves.

    Returns
    -------
    xr.Dataset
        A **lazy** (Dask-backed) dataset.  RAM usage is O(metadata) until
        ``.compute()`` is called.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist on disk.
    ValueError
        If the file extension is not supported.

    Notes
    -----
    The ``chunks='auto'`` argument instructs Dask to choose chunk sizes that
    balance parallelism and memory usage automatically.  For crop-model
    pixel-level extraction, the bottleneck is spatial — so the transform
    layer (:mod:`ag_cube_cm.transform`) chunks along ``lat``/``lon`` rather
    than ``time`` when building cubes.  However, at the *open* stage we let
    Dask decide to remain general.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: '{filepath}'")

    suffix = path.suffix.lower()

    if suffix == ".nc":
        # True lazy open — returns a Dask-backed Dataset.
        # chunks='auto' → Dask picks sizes; no data is read from disk.
        logger.debug("Opening NetCDF (chunks=%r): %s", chunks, filepath)
        return xr.open_dataset(str(path), engine=engine, chunks=chunks)

    if suffix in {".tif", ".tiff"}:
        try:
            import rioxarray  # noqa: F401 — registers .rio accessor
        except ImportError as exc:
            raise ImportError(
                "rioxarray is required to open GeoTIFF files. "
                "Install with: pip install rioxarray"
            ) from exc

        logger.debug("Opening GeoTIFF (lazy): %s", filepath)
        # open_rasterio with chunks='auto' returns a Dask-backed DataArray.
        # Wrap in Dataset for a uniform return type.
        da = xr.open_rasterio(str(path), chunks="auto")  # type: ignore[attr-defined]
        return da.to_dataset(name=path.stem)

    if suffix in {".pkl", ".pickle"}:
        logger.warning(
            "Opening pickle '%s' — this IS eager (not lazy). "
            "Use .nc or .tif for large spatial datasets.",
            filepath,
        )
        with open(str(path), "rb") as fh:
            return pickle.load(fh)  # noqa: S301

    raise ValueError(
        f"Unsupported file extension '{suffix}'. "
        "Supported: .nc, .tif, .tiff, .pkl, .pickle"
    )


# ---------------------------------------------------------------------------
# SpatialData class
# ---------------------------------------------------------------------------


class SpatialData:
    """Lazy accessor for the three spatial data cubes used in simulations.

    :class:`SpatialData` does **not** hold any arrays in memory.  Each
    dataset property opens the file on first access (via
    :func:`_open_dataset`) and caches the resulting lazy ``xr.Dataset`` so
    subsequent accesses are instantaneous.

    Parameters
    ----------
    weather_path : str | None
        Path to the multi-temporal weather NetCDF cube (AgEra5 / CHIRPS).
    soil_path : str | None
        Path to the multi-depth SoilGrids NetCDF cube.
    dem_path : str | None
        Path to the Digital Elevation Model raster.  Optional; only required
        by CAF2021 and SIMPLE models.
    engine : str
        NetCDF engine used by :func:`_open_dataset`.  Default: ``"netcdf4"``.

    Attributes
    ----------
    climate : xr.Dataset
        Lazy multi-temporal weather dataset.  Populated on first access.
    soil : xr.Dataset
        Lazy multi-depth soil dataset.  Populated on first access.
    dem : xr.Dataset | None
        Lazy DEM dataset, or ``None`` if ``dem_path`` was not provided.

    Examples
    --------
    >>> sd = SpatialData(
    ...     weather_path="data/weather_mwi_2000_2019.nc",
    ...     soil_path="data/soil_mwi.nc",
    ... )
    >>> # Dataset is NOT in memory yet:
    >>> climate = sd.climate   # triggers the lazy open, but still no RAM spike
    >>> # Dask graph, not a numpy array:
    >>> print(type(climate["tmax"].data))
    <class 'dask.array.core.Array'>
    >>> # Only NOW does data materialise in RAM (single pixel, tiny):
    >>> tmax_pixel = climate["tmax"].sel(lat=13.5, lon=-90.0, method="nearest").compute()
    """

    def __init__(
        self,
        weather_path: str | None = None,
        soil_path: str | None = None,
        dem_path: str | None = None,
        engine: str = "netcdf4",
    ) -> None:
        self.weather_path = weather_path
        self.soil_path = soil_path
        self.dem_path = dem_path
        self.engine = engine

        # Private lazy-cache slots
        self._climate: xr.Dataset | None = None
        self._soil: xr.Dataset | None = None
        self._dem: xr.Dataset | None = None

    # ------------------------------------------------------------------
    # Lazy dataset properties
    # ------------------------------------------------------------------

    @property
    def climate(self) -> xr.Dataset:
        """Lazy multi-temporal weather dataset (opens on first access)."""
        if self._climate is None:
            if not self.weather_path:
                raise ValueError("weather_path was not set on SpatialData.")
            self._climate = _open_dataset(self.weather_path, self.engine)
            logger.info("Opened weather cube (lazy): %s", self.weather_path)
        return self._climate

    @property
    def soil(self) -> xr.Dataset:
        """Lazy multi-depth soil dataset (opens on first access)."""
        if self._soil is None:
            if not self.soil_path:
                raise ValueError("soil_path was not set on SpatialData.")
            self._soil = _open_dataset(self.soil_path, self.engine)
            logger.info("Opened soil cube (lazy): %s", self.soil_path)
        return self._soil

    @property
    def dem(self) -> xr.Dataset | None:
        """Lazy DEM dataset, or ``None`` if no path was provided."""
        if self._dem is None and self.dem_path:
            self._dem = _open_dataset(self.dem_path, self.engine)
            logger.info("Opened DEM (lazy): %s", self.dem_path)
        return self._dem

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def open(
        filepath: str,
        engine: str = "netcdf4",
        chunks: str | dict | None = "auto",
    ) -> xr.Dataset:
        """Thin public alias for :func:`_open_dataset`.

        Use this when you need a one-off open outside of a full
        :class:`SpatialData` instance.  This is the single supported entry
        point for reading a cube from disk — go through it rather than
        calling :func:`xarray.open_dataset` directly, so format support stays
        in one place.

        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        engine : str
            NetCDF engine.  Default: ``"netcdf4"``.
        chunks : str | dict | None
            Dask chunking; ``None`` for an un-chunked open.

        Returns
        -------
        xr.Dataset
            Dataset, Dask-backed unless ``chunks=None``.
        """
        return _open_dataset(filepath, engine, chunks)

    def close(self) -> None:
        """Release all open file handles and clear cached datasets.

        Call this explicitly when the :class:`SpatialData` object goes out
        of scope in a long-running process to avoid file-handle leaks.
        """
        for attr in ("_climate", "_soil", "_dem"):
            ds: xr.Dataset | None = getattr(self, attr)
            if ds is not None:
                try:
                    ds.close()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, attr, None)
        logger.debug("SpatialData: all datasets closed.")

    def __repr__(self) -> str:
        loaded = []
        if self._climate is not None:
            loaded.append("climate")
        if self._soil is not None:
            loaded.append("soil")
        if self._dem is not None:
            loaded.append("dem")
        status = ", ".join(loaded) if loaded else "none loaded yet"
        return (
            f"SpatialData("
            f"weather='{self.weather_path}', "
            f"soil='{self.soil_path}', "
            f"dem='{self.dem_path}', "
            f"loaded=[{status}])"
        )
