"""
ag_cube_cm.pipeline
====================

``full_pipeline`` mode: acquire data with **aggeodata**, then simulate.

This module is the *only* part of ag-cube-cm that depends on aggeodata, and it
is imported lazily — solely when a run config declares ``mode: full_pipeline``.
Everything else in the package reads datacubes through
:mod:`ag_cube_cm.spatial.data`, whose contract is a file format (CF-compliant
NetCDF or Zarr), not a Python API. A cube produced by any other tool works just
as well.

That split is deliberate. It keeps the core package installable and usable on
its own, while this module provides the convenience of running acquisition and
simulation with one command:

    pip install "ag-cube-cm[pipeline]"
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import click
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ag_cube_cm.cli import (
    CropSpec,
    ManagementSpec,
    SimSpec,
    WithCubesConfig,
    _DEFAULT_SOIL_DEPTHS,
    _DEFAULT_SOIL_VARS,
    _run_with_cubes,
)

try:
    # Harmonized schema: the climate + soil sections reuse aggeodata's Pydantic
    # models so a full_pipeline YAML has the same shape as a stand-alone
    # aggeodata ingestion YAML.
    from aggeodata.config import IngestionClimateConfig, SoilConfig, VariableConfig
except ImportError as exc:  # pragma: no cover - depends on install extras
    raise ImportError(
        "mode: full_pipeline requires the aggeodata package. "
        "Install it with:  pip install 'ag-cube-cm[pipeline]'  — or use "
        "mode: with_cubes, which reads datacubes already on disk and needs no "
        "data-acquisition dependencies."
    ) from exc

logger = logging.getLogger(__name__)

#: Datacube formats aggeodata can produce, in the order they are looked for.
_CUBE_SUFFIXES = (".nc", ".zarr")


def _find_cube(folder: Path, stem: str) -> Path:
    """Return an existing cube named *stem* in *folder*, whatever its format.

    aggeodata writes either ``<stem>.nc`` or ``<stem>.zarr`` depending on its
    ``output_format``. Falls back to the NetCDF name when neither exists yet,
    so the caller can use it as the build target.
    """
    for suffix in _CUBE_SUFFIXES:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return folder / f"{stem}{_CUBE_SUFFIXES[0]}"

# ---------------------------------------------------------------------------
# Mode 2 — full_pipeline
# ---------------------------------------------------------------------------

class BboxSpec(BaseModel):
    bbox: Annotated[
        list[float],
        Field(min_length=4, max_length=4,
              description="[xmin, ymin, xmax, ymax] in WGS-84 decimal degrees"),
    ]

    @model_validator(mode="after")
    def _valid_bbox(self) -> BboxSpec:
        xmin, ymin, xmax, ymax = self.bbox
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"Invalid bbox: xmin < xmax and ymin < ymax required. Got {self.bbox}")
        return self


class DatesSpec(BaseModel):
    start: Annotated[str, Field(description="Start date YYYY-MM-DD")]
    end: Annotated[str, Field(description="End date YYYY-MM-DD")]

    @model_validator(mode="after")
    def _start_before_end(self) -> DatesSpec:
        if self.start >= self.end:
            raise ValueError(f"dates.start ({self.start}) must be before dates.end ({self.end})")
        return self


class ClimateSpec(BaseModel):
    """Climate section.

    The canonical shape mirrors aggeodata's CLIMATE block: a ``variables``
    dict mapping each CF variable name to a ``VariableConfig``
    (``source`` plus per-variable knobs like ``gee_project`` /
    ``gee_dataset_id``).  The legacy ``sources: {pr: chirps, ...}`` form is
    still accepted via a ``mode='before'`` shim that rewrites it into the
    nested shape and emits a ``DeprecationWarning``.

    Build-time options (``ncores``, ``agera5_version``,
    ``reference_variable``) stay at the section level because they are
    fed to aggeodata's ``GeneralConfig`` when the internal pipeline YAML
    is generated.
    """

    model_config = ConfigDict(populate_by_name=True)

    variables: Annotated[
        dict[str, VariableConfig],
        Field(description=(
            "CF variable name -> source config. "
            "Each value is a VariableConfig: {source, gee_project?, "
            "gee_dataset_id?, chirts_source?, agera5_key?, nasa_power_param?}. "
            "Sources: chirps, chirts, agera5, nasa_power, gee."
        )),
    ]
    ncores: Annotated[int, Field(default=2, ge=1)]
    agera5_version: Annotated[str, Field(default="2_0")]
    reference_variable: Annotated[str, Field(default="pr",
        description="CF variable whose grid defines the output resolution")]

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_form(cls, data: Any) -> Any:
        """Accept the deprecated `sources: {var: src}` + flat `gee_project` form.

        Rewrites it into the canonical `variables: {var: VariableConfig}` shape
        and emits a single DeprecationWarning so existing YAMLs keep working
        while the user is nudged toward the harmonized form.
        """
        if not isinstance(data, dict):
            return data

        # Already canonical or partially canonical — leave as is
        if "variables" in data and "sources" not in data:
            return data

        if "sources" in data:
            sources = data.pop("sources")
            top_gee_project = data.pop("gee_project", None)

            if not isinstance(sources, dict):
                raise TypeError(
                    "climate.sources must be a mapping of CF variable -> source string"
                )

            variables: dict[str, dict] = {}
            for cf_var, value in sources.items():
                if isinstance(value, str):
                    var_cfg: dict[str, Any] = {"source": value}
                    if value == "gee" and top_gee_project:
                        var_cfg["gee_project"] = top_gee_project
                    variables[cf_var] = var_cfg
                elif isinstance(value, dict):
                    # Already nested — accept as is
                    variables[cf_var] = value
                else:
                    raise TypeError(
                        f"climate.sources[{cf_var!r}] must be a string or mapping; got {type(value).__name__}"
                    )

            data["variables"] = variables

            warnings.warn(
                "The flat `climate.sources: {var: source}` form (with optional "
                "top-level `gee_project`) is deprecated. Use the harmonized "
                "`climate.variables: {var: {source: ..., gee_project: ...}}` form, "
                "which matches the aggeodata climate-data-download YAML schema.",
                DeprecationWarning,
                stacklevel=3,
            )
        return data


# SoilSpec is now an alias for aggeodata.config.SoilConfig — same schema is used
# by the climate-data-download skill (when soil.enabled is true) and by
# spatial-crop-modeler. Defaults are overridden here so ag-cube-cm gets the full
# crop-modeling soil profile out of the box without explicit listing.
class SoilSpec(SoilConfig):
    variables: Annotated[
        list[str],
        Field(
            default_factory=lambda: list(_DEFAULT_SOIL_VARS),
            alias="layers",
            description="SoilGrids variable names",
        ),
    ]
    depths: Annotated[
        list[str],
        Field(
            default_factory=lambda: list(_DEFAULT_SOIL_DEPTHS),
            description="Depth intervals",
        ),
    ]
    reference_variable: Annotated[str, Field(default="wv1500")]


class FullPipelineConfig(BaseModel):
    """Config for the complete download + build + simulate pipeline."""

    mode: Literal["full_pipeline"]
    output_dir: Annotated[str, Field(description="Root directory for all outputs")]
    suffix: Annotated[str, Field(description="Short label appended to all output filenames")]
    spatial: BboxSpec
    dates: DatesSpec
    climate: ClimateSpec
    soil: SoilSpec
    simulation: SimSpec
    crop: CropSpec
    management: ManagementSpec


def _run_full_pipeline(cfg: FullPipelineConfig, dry_run: bool = False) -> dict:
    """Execute mode: full_pipeline (download + build cubes + simulate)."""
    import os

    base = Path(cfg.output_dir)
    climate_dir = base / "climate_raw"
    soil_dir = base / "soil_raw"
    base.mkdir(parents=True, exist_ok=True)
    climate_dir.mkdir(exist_ok=True)
    soil_dir.mkdir(exist_ok=True)

    sy, ey = cfg.dates.start[:4], cfg.dates.end[:4]
    # aggeodata can emit the climate cube as NetCDF or as a Zarr store, and the
    # raw downloads likewise. Resolve by globbing rather than assuming '.nc',
    # or a Zarr run re-downloads the whole archive on every invocation.
    weather_nc = _find_cube(climate_dir, f"climate_{cfg.suffix}_{sy}_{ey}")
    soil_nc = _find_cube(base, f"soil_{cfg.suffix}")
    yield_nc = base / f"yield_{cfg.suffix}.nc"
    ageo_cfg_path = base / f"aggeodata_{cfg.suffix}.yaml"

    # -- aggeodata config (canonical lowercase schema) --
    ageo_cfg: dict = {
        "dates": {"starting_date": cfg.dates.start, "ending_date": cfg.dates.end},
        "spatial_info": {"extent": cfg.spatial.bbox},
        "climate": {
            # Pydantic round-trip: dump each VariableConfig and drop unset/
            # default keys so the on-disk YAML stays compact (avoids
            # `gee_project: null` and `chirts_source: era5` noise on variables
            # where those fields aren't relevant).
            "variables": {
                cf_var: var_cfg.model_dump(exclude_none=True, exclude_defaults=True)
                for cf_var, var_cfg in cfg.climate.variables.items()
            }
        },
        "soil": {"enabled": False},
        "general": {
            "suffix":             cfg.suffix,
            "ncores":             cfg.climate.ncores,
            "task":               "download",
            "reference_variable": cfg.climate.reference_variable,
            "agera5_version":     cfg.climate.agera5_version,
            "target_crs":         "EPSG:4326",
        },
        "paths": {"output_path": str(climate_dir)},
    }
    with open(ageo_cfg_path, "w") as fh:
        yaml.dump(ageo_cfg, fh, default_flow_style=False, sort_keys=False)

    if dry_run:
        click.echo(f"[dry-run] aggeodata config -> {ageo_cfg_path}")
        click.echo(f"[dry-run] Would download: {list(cfg.climate.variables.keys())}")
        click.echo(f"[dry-run] Would build:    {weather_nc}")
        click.echo(f"[dry-run] Would download soil + build {soil_nc}")
        click.echo(f"[dry-run] Would simulate -> {yield_nc}")
        return {"status": "dry_run"}

    # -- [1] download climate --
    ref_var = cfg.climate.reference_variable
    raw_dir = climate_dir / f"{ref_var}_{cfg.suffix}_raw"
    # download_format: "zarr" leaves per-year stores and no .nc at all.
    if raw_dir.exists() and (any(raw_dir.rglob("*.nc")) or any(raw_dir.rglob("*.zarr"))):
        click.echo(f"  [skip] Climate already downloaded in {climate_dir}")
    else:
        click.echo(f"  Downloading climate ({list(cfg.climate.variables.keys())}) ...")
        from aggeodata.pipelines.download import run_download
        run_download(str(ageo_cfg_path))

    # -- [2] build climate datacube --
    if weather_nc.exists():
        click.echo(f"  [skip] Climate datacube exists: {weather_nc}")
    else:
        click.echo("  Building climate datacube ...")
        from aggeodata.pipelines.datacube import run_datacube
        nc_path = run_datacube(str(ageo_cfg_path))
        weather_nc = Path(nc_path)
        click.echo(f"  -> {weather_nc}")

    # -- [3] download soil --
    tifs = list(soil_dir.glob("*.tif"))
    if tifs:
        click.echo(f"  [skip] Soil already downloaded ({len(tifs)} files)")
    else:
        click.echo("  Downloading soil (SoilGrids) ...")
        from aggeodata.ingestion.soil import SoilGridsDownloader
        dl = SoilGridsDownloader(
            soil_layers=cfg.soil.variables,
            depths=cfg.soil.depths,
            output_folder=str(soil_dir),
        )
        downloaded = dl.download(boundaries=cfg.spatial.bbox)
        click.echo(f"  -> {len(downloaded)} GeoTIFFs")

    # -- [4] build soil datacube --
    if soil_nc.exists():
        click.echo(f"  [skip] Soil datacube exists: {soil_nc}")
    else:
        click.echo("  Building soil datacube ...")
        from aggeodata.transform.soil_cube import SoilDataCubeBuilder
        builder = SoilDataCubeBuilder(
            data_folder=str(soil_dir),
            variables=cfg.soil.variables,
            reference_variable=cfg.soil.reference_variable,
            target_crs="EPSG:4326",
        )
        builder.build_and_save(output_path=str(base), filename=soil_nc.name)
        click.echo(f"  -> {soil_nc}")

    # -- [5] simulate --
    with_cubes_cfg = WithCubesConfig(
        mode="with_cubes",
        weather_path=str(weather_nc),
        soil_path=str(soil_nc),
        output_path=str(yield_nc),
        simulation=cfg.simulation,
        crop=cfg.crop,
        management=cfg.management,
    )
    return _run_with_cubes(with_cubes_cfg, dry_run=False)
