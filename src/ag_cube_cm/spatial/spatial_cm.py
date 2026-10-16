"""
ag_cube_cm.spatial.spatial_cm
==============================

Spatial orchestrator for crop models — the single implementation.

:class:`SpatialCM` owns the whole pixel pipeline: open the climate and soil
cubes, clip them to a region of interest, expand the request into one job per
``pixel x planting window x season``, run those jobs in parallel through the
configured :class:`~ag_cube_cm.models.base.CropModel`, and assemble the results
into a ``planting_window x year x y x x`` datacube.

Both front ends are thin adapters over this class:

* :func:`ag_cube_cm.mcp_server.run_simulation` — serialises
  :class:`SimulationResult` to JSON for MCP clients.
* :func:`ag_cube_cm.cli._run_with_cubes` — calls :meth:`SpatialCM.run`
  directly.

Neither holds orchestration logic of its own, so behaviour cannot drift
between them.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from ag_cube_cm.config.schemas import SimulationConfig
from ag_cube_cm.models.factory import model_factory
from ag_cube_cm.spatial.data import SpatialData
from ag_cube_cm.spatial.raster_ops import get_roi_data
from ag_cube_cm.spatial.reporter import SpatialReporter

logger = logging.getLogger(__name__)

__all__ = ["SpatialCM", "SimulationResult"]

#: Days after planting beyond which a season is assumed finished.  Used to
#: decide which seasons the weather record can actually support.
_SEASON_SPAN_DAYS = 200

#: Wider span used when slicing a pixel's weather to one season, so a late
#: maturing cultivar still has weather past its harvest date.
_SEASON_SLICE_DAYS = 250


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Outcome of a spatial simulation.

    Attributes
    ----------
    cube : xr.Dataset
        ``planting_window x year x y x x`` cube with the model's yield
        variable and a ``flag`` array (0 = ok, 1 = model error, 2 = no data).
    output_path : str | None
        Where the cube was written, or None if it was not saved.
    yield_variable : str
        Name of the yield variable inside *cube*.
    pixels_ok, pixels_failed, pixels_skipped : int
        Job counts by flag.  These count ``pixel x window x year`` jobs, not
        distinct pixels.
    n_planting_windows : int
        Number of planting windows simulated.
    years : list[int]
        Seasons simulated.
    mean_yield : float | None
        Mean of the yield variable over successful jobs, or None if none.
    """

    cube: xr.Dataset
    output_path: str | None
    yield_variable: str = "HWAM"
    yield_units: str = "kg/ha"
    pixels_ok: int = 0
    pixels_failed: int = 0
    pixels_skipped: int = 0
    n_planting_windows: int = 0
    years: list[int] = field(default_factory=list)
    mean_yield: float | None = None
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the run."""
        payload: dict[str, Any] = {
            "output_path": self.output_path,
            "pixels_ok": self.pixels_ok,
            "pixels_skipped": self.pixels_skipped,
            "pixels_failed": self.pixels_failed,
            "n_planting_windows": self.n_planting_windows,
            "n_years": len(self.years),
            "years": self.years,
            "yield_variable": self.yield_variable,
            "yield_units": self.yield_units,
        }
        # Keep the historical key so existing MCP clients keep working.
        mean = round(self.mean_yield, 1) if self.mean_yield is not None else None
        payload["mean_hwam_kg_ha"] = mean
        payload[f"mean_{self.yield_variable.lower()}"] = mean
        if self.errors:
            payload["sample_errors"] = self.errors[:5]
        return payload


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SpatialCM:
    """Run a crop model across every pixel of a climate/soil datacube pair.

    Parameters
    ----------
    config : SimulationConfig
        Validated simulation configuration.  Cube paths come from
        ``config.SPATIAL_INFO.weather_path`` / ``.soil_path``.

    Examples
    --------
    >>> from ag_cube_cm.config.loader import load_config
    >>> from ag_cube_cm.spatial import SpatialCM
    >>> cm = SpatialCM(load_config("maize_hnd.yaml"))
    >>> result = cm.run(feature="Comayagua", max_pixels=50)
    >>> result.summary()["mean_hwam_kg_ha"]
    4210.0
    >>> cm.close()
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.ncores: int = getattr(config.GENERAL_INFO, "ncores", 4) or 4
        self.target_crs: str = getattr(
            config.GENERAL_INFO, "projected_crs", "EPSG:4326"
        ) or "EPSG:4326"

        # Un-chunked: the pixel loop issues one tiny .sel() per job from a
        # thread pool, where Dask's scheduler overhead dominates any benefit.
        self.spatial_data = SpatialData(
            weather_path=config.SPATIAL_INFO.weather_path,
            soil_path=config.SPATIAL_INFO.soil_path,
            dem_path=getattr(config.SPATIAL_INFO, "dem_path", None),
            engine="netcdf4",
        )
        self._chunks: str | dict | None = None

        # A model instance is built per job (state must not be shared across
        # threads); this one exists only to read class-level metadata.
        self._template_model = model_factory(config)
        self.yield_variable: str = getattr(self._template_model, "YIELD_VARIABLE", "HWAM")
        self.yield_units: str = getattr(self._template_model, "YIELD_UNITS", "kg/ha")
        self.yield_long_name: str = getattr(
            self._template_model, "YIELD_LONG_NAME", "Mean grain yield at maturity"
        )

        self.reporter = SpatialReporter(crs=self.target_crs)

        self._weather: xr.Dataset | None = None
        self._soil: xr.Dataset | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        feature: str | None = None,
        adm_level: int | None = None,
        max_pixels: int | None = None,
        save: bool = True,
    ) -> SimulationResult:
        """Simulate every pixel x planting window x season and grid the result.

        Parameters
        ----------
        feature : str | None
            Admin unit to clip to (e.g. ``"Zomba"``).  Overrides
            ``config.SPATIAL_INFO.feature`` when given.
        adm_level : int | None
            Administrative level for the boundary lookup.  Defaults to
            ``config.SPATIAL_INFO.adm_level``.
        max_pixels : int | None
            Cap on pixels simulated — useful for smoke tests.
        save : bool
            Write the cube to ``config.SPATIAL_INFO.output_path``.  Default: True.

        Returns
        -------
        SimulationResult
        """
        weather_ds, soil_ds = self._open_cubes()
        weather_ds, soil_ds = self._apply_roi(weather_ds, soil_ds, feature, adm_level)

        time_dim = next((d for d in weather_ds.dims if d in {"time", "date"}), None)
        planting_dates = self._planting_dates()
        sim_years = self._simulation_years(weather_ds, time_dim)
        jobs = self._build_jobs(weather_ds, planting_dates, sim_years, max_pixels)

        if not jobs:
            raise ValueError(
                "No pixels to simulate — check that the weather cube overlaps "
                "the requested region."
            )

        logger.info(
            "Simulating %d job(s): %d pixel(s) x %d window(s) x %d season(s)",
            len(jobs), len(jobs) // max(len(planting_dates) * len(sim_years), 1),
            len(planting_dates), len(sim_years),
        )

        records = self._execute(jobs, weather_ds, soil_ds, time_dim)
        return self._assemble(records, planting_dates, sim_years, save)

    def close(self) -> None:
        """Release the open cube file handles."""
        for ds in (self._weather, self._soil):
            if ds is not None:
                try:
                    ds.close()
                except Exception:  # noqa: BLE001
                    pass
        self._weather = self._soil = None
        self.spatial_data.close()

    def __enter__(self) -> SpatialCM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _open_cubes(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Open both cubes, normalise their dims and assert a CRS."""
        import rioxarray  # noqa: F401 — registers the .rio accessor

        weather_ds = SpatialData.open(
            self.config.SPATIAL_INFO.weather_path, chunks=self._chunks
        )
        soil_ds = SpatialData.open(
            self.config.SPATIAL_INFO.soil_path, chunks=self._chunks
        )

        for ds in (weather_ds, soil_ds):
            # A stale grid_mapping pointer makes rioxarray raise on write_crs.
            for name in list(ds.data_vars) + list(ds.coords):
                ds.variables[name].encoding.pop("grid_mapping", None)

        weather_ds = _normalise_dims(weather_ds).rio.write_crs("EPSG:4326")
        soil_ds = _normalise_dims(soil_ds).rio.write_crs("EPSG:4326")

        self._weather, self._soil = weather_ds, soil_ds
        return weather_ds, soil_ds

    def _apply_roi(
        self,
        weather_ds: xr.Dataset,
        soil_ds: xr.Dataset,
        feature: str | None,
        adm_level: int | None,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Clip both cubes to an admin boundary, when one is requested."""
        effective_feature = feature or self.config.SPATIAL_INFO.feature
        if not effective_feature:
            return weather_ds, soil_ds

        from ag_cube_cm.ingestion.boundaries import get_admin_boundary

        effective_adm = (
            adm_level if adm_level is not None else self.config.SPATIAL_INFO.adm_level
        )
        logger.info("Clipping to '%s' (adm level %s)", effective_feature, effective_adm)
        boundary: gpd.GeoDataFrame = get_admin_boundary(
            self.config.GENERAL_INFO.country_code,
            effective_feature,
            adm_level=effective_adm,
        )
        return (
            get_roi_data(weather_ds, boundary),
            get_roi_data(soil_ds, boundary),
        )

    def _planting_dates(self) -> list[date]:
        """Return the base planting date for each staggered window."""
        base = self.config.MANAGEMENT.planting_date
        n_windows = self.config.MANAGEMENT.n_planting_windows or 1
        step = self.config.MANAGEMENT.planting_window_days
        return [base + timedelta(days=w * step) for w in range(n_windows)]

    def _simulation_years(
        self, weather_ds: xr.Dataset, time_dim: str | None
    ) -> list[int]:
        """Return the seasons the weather record can actually support.

        A season is kept only when its harvest still falls inside the weather
        record — otherwise DSSAT would force-harvest on the first missing day
        and report a misleadingly low yield.
        """
        base_pdate = self.config.MANAGEMENT.planting_date
        if time_dim is None:
            return [base_pdate.year]

        try:
            raw_times = weather_ds[time_dim].values
            all_years = sorted({int(pd.Timestamp(t).year) for t in raw_times})
        except Exception:  # noqa: BLE001
            return [base_pdate.year]

        if not all_years:
            return [base_pdate.year]

        last_yr = max(all_years)
        sim_years = [
            yr for yr in all_years
            if (_shift_year(base_pdate, yr) + timedelta(days=_SEASON_SPAN_DAYS)).year
            <= last_yr
        ]
        if not sim_years:
            logger.warning(
                "No season fits entirely inside the weather record "
                "(%s..%s); simulating all years anyway.", all_years[0], last_yr,
            )
            return all_years
        return sim_years

    def _build_jobs(
        self,
        weather_ds: xr.Dataset,
        planting_dates: list[date],
        sim_years: list[int],
        max_pixels: int | None,
    ) -> list[tuple]:
        """Expand the grid into one job per pixel x window x season."""
        pixels = [
            (float(y), float(x))
            for y in weather_ds.y.values
            for x in weather_ds.x.values
        ]
        if max_pixels is not None:
            # Callers that arrive through a JSON or protobuf boundary hand over
            # 1.0 rather than 1 — Gemini's Struct has only one number type, and
            # it is a double. A float is not a valid slice index, so without this
            # the run dies here with "slice indices must be integers", after the
            # weather and soil cubes have already been built.
            pixels = pixels[:int(max_pixels)]

        jobs: list[tuple] = []
        for idx, (py, px) in enumerate(pixels):
            for w_idx, base_pd in enumerate(planting_dates):
                for yr in sim_years:
                    jobs.append((idx, w_idx, yr, _shift_year(base_pd, yr), py, px))
        return jobs

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        jobs: list[tuple],
        weather_ds: xr.Dataset,
        soil_ds: xr.Dataset,
        time_dim: str | None,
    ) -> list[dict]:
        """Run every job, in parallel, collecting one record each.

        A thread pool (not a process pool) is used throughout: binary models
        like DSSAT spend their time blocked on a subprocess, and the closure
        below captures open Datasets, which are not picklable.
        """
        from tqdm import tqdm

        def _worker(args: tuple) -> dict:
            return self._run_job(args, weather_ds, soil_ds, time_dim)

        records: list[dict] = []
        with tqdm(total=len(jobs), desc="Simulating", unit="job") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.ncores) as pool:
                for record in pool.map(_worker, jobs):
                    records.append(record)
                    pbar.update(1)
        return records

    def _run_job(
        self,
        args: tuple,
        weather_ds: xr.Dataset,
        soil_ds: xr.Dataset,
        time_dim: str | None,
    ) -> dict:
        """Run one pixel x window x season and return a flat record.

        Never raises: a failure is reported through ``flag`` so one bad pixel
        cannot abort the sweep.
        """
        pixel_idx, w_idx, yr, pdate_yr, y, x = args
        record: dict = {
            "pixel_idx": pixel_idx, "window_idx": w_idx, "year": yr,
            "y": y, "x": x, "value": np.nan, "flag": 2, "error": "",
        }

        try:
            w_slice = weather_ds.sel(y=y, x=x, method="nearest")
            s_slice = soil_ds.sel(y=y, x=x, method="nearest")

            if time_dim is not None:
                w_slice = _slice_season(w_slice, time_dim, yr, pdate_yr)

            if not _has_data(w_slice) or not _has_data(s_slice):
                record["error"] = "no-data pixel"
                return record

            # Each window/season runs with its own planting date.
            mgmt = self.config.MANAGEMENT.model_copy(
                update={"planting_date": pdate_yr}
            )
            cfg_job = self.config.model_copy(update={"MANAGEMENT": mgmt})

            model = model_factory(cfg_job)
            model.setup_working_directory(f"px{pixel_idx}_w{w_idx:02d}_y{yr}")
            stage = "prepare_inputs"
            try:
                _prepare_inputs(model, w_slice, s_slice)
                stage = "run_simulation"
                model.run_simulation()
                stage = "collect_outputs"
                outputs = model.collect_outputs() or {}

                raw = outputs.get(self.yield_variable) if outputs else None
                record["value"] = (
                    float(raw) if raw not in (None, "", "-99") else np.nan
                )
                record["flag"] = 0 if outputs else 1
                if not outputs:
                    record["error"] = "model produced no output"
            except Exception as exc:  # noqa: BLE001
                # Naming the stage and the exception type matters: these failures
                # are reported through `flag`, never raised, so this string is the
                # only evidence left of what went wrong.
                record["flag"] = 1
                record["error"] = f"{stage}: {type(exc).__name__}: {exc}"
            finally:
                model.cleanup_working_directory()

        except Exception as exc:  # noqa: BLE001
            record["flag"] = 1
            record["error"] = f"slice: {type(exc).__name__}: {exc}"

        return record

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(
        self,
        records: list[dict],
        planting_dates: list[date],
        sim_years: list[int],
        save: bool,
    ) -> SimulationResult:
        """Grid the flat records and optionally write them to disk."""
        df = pd.DataFrame(records)
        ok_df = df[df["flag"] == 0]
        mean_yield = (
            float(ok_df["value"].dropna().mean()) if not ok_df.empty else None
        )
        if mean_yield is not None and np.isnan(mean_yield):
            mean_yield = None

        cube = self.reporter.build_yield_cube(
            df,
            planting_dates=planting_dates,
            years=sim_years,
            value_name=self.yield_variable,
            value_units=self.yield_units,
            value_long_name=self.yield_long_name,
        )

        out_path = self.config.SPATIAL_INFO.output_path if save else None
        if out_path:
            self.reporter.export_to_raster(cube, out_path, format="netcdf")

        errors = [e for e in df.loc[df["flag"] == 1, "error"].tolist() if e]
        result = SimulationResult(
            cube=cube,
            output_path=out_path,
            yield_variable=self.yield_variable,
            yield_units=self.yield_units,
            pixels_ok=int((df["flag"] == 0).sum()),
            pixels_failed=int((df["flag"] == 1).sum()),
            pixels_skipped=int((df["flag"] == 2).sum()),
            n_planting_windows=len(planting_dates),
            years=list(sim_years),
            mean_yield=mean_yield,
            errors=errors,
        )
        logger.info(
            "Done: %d ok, %d failed, %d no-data; mean %s = %s %s",
            result.pixels_ok, result.pixels_failed, result.pixels_skipped,
            self.yield_variable,
            f"{mean_yield:.1f}" if mean_yield is not None else "n/a",
            self.yield_units,
        )
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalise_dims(ds: xr.Dataset) -> xr.Dataset:
    """Rename geographic dims to the ``y``/``x`` convention used internally."""
    rename: dict[str, str] = {}
    for src, dst in (("lat", "y"), ("latitude", "y"),
                     ("lon", "x"), ("longitude", "x")):
        if src in ds.dims and dst not in ds.dims:
            rename[src] = dst
    return ds.rename(rename) if rename else ds


def _shift_year(d: date, year: int) -> date:
    """Return *d* moved to *year*, clamping 29 Feb onto a non-leap year."""
    try:
        return d.replace(year=year)
    except ValueError:
        return d.replace(year=year, day=28)


def _slice_season(
    ds: xr.Dataset, time_dim: str, year: int, pdate: date
) -> xr.Dataset:
    """Restrict a pixel's weather to the season starting in *year*."""
    try:
        times = pd.DatetimeIndex([pd.Timestamp(t) for t in ds[time_dim].values])
        harvest_yr = (pdate + timedelta(days=_SEASON_SLICE_DAYS)).year
        mask = times.year.isin(range(year, harvest_yr + 1))
        return ds.isel({time_dim: mask.values})
    except Exception:  # noqa: BLE001 — an unsliceable axis is not fatal
        return ds


def _has_data(ds: xr.Dataset) -> bool:
    """Return True when *ds* holds at least one non-NaN value."""
    try:
        df = ds.to_dataframe()
        if df.index.nlevels > 0:
            df = df.reset_index()
        return not df.dropna().empty
    except Exception:  # noqa: BLE001
        # 0-dimensional slice (single-point soil) — inspect the values directly.
        return any(
            not np.all(np.isnan(ds[v].values))
            for v in ds.data_vars
            if np.issubdtype(ds[v].dtype, np.floating)
        )


def _prepare_inputs(model: Any, w_slice: xr.Dataset, s_slice: xr.Dataset) -> None:
    """Call ``prepare_inputs``, passing elevation only when it is accepted.

    The ABC declares ``prepare_inputs(weather, soil)``; DSSAT extends it with
    an ``elevation`` keyword for the .WTH header.  Probing the signature keeps
    the orchestrator usable with models that implement only the base contract.
    """
    try:
        params = inspect.signature(model.prepare_inputs).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins/C functions
        params = {}
    if "elevation" in params:
        model.prepare_inputs(w_slice, s_slice, elevation=0.0)
    else:
        model.prepare_inputs(w_slice, s_slice)
