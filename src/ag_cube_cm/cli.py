"""
ag-cube-cm CLI
==============

Two execution modes, selected by the ``mode`` key in the YAML config:

  with_cubes    Weather + soil NetCDF datacubes already exist on disk.
                Skips all data acquisition; runs DSSAT directly.

  full_pipeline Downloads climate (CHIRPS / CHIRTS / AgERA5) and soil
                (SoilGrids) data via aggeodata, builds both datacubes,
                then runs the crop model simulation.

Usage
-----
    ag-cube-cm run       CONFIG.yaml [--dry-run]
    ag-cube-cm validate  CONFIG.yaml
    ag-cube-cm template  with_cubes | full_pipeline

Entry point
-----------
    Registered as ``ag-cube-cm`` in pyproject.toml [project.scripts].
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import click
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Harmonized schema: the climate + soil sections share the aggeodata Pydantic
# models so an ag-cube-cm full_pipeline YAML uses the same shape as a stand-alone
# aggeodata ingestion YAML. See aggeodata.config.schemas for the definitions.
from aggeodata.config import IngestionClimateConfig, SoilConfig, VariableConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

_DEFAULT_SOIL_VARS = [
    "clay", "sand", "silt", "bdod", "cfvo",
    "soc", "phh2o", "wv0010", "wv0033", "wv1500",
]
_DEFAULT_SOIL_DEPTHS = ["0-5", "5-15", "15-30", "30-60", "60-100"]


class FertilizerEvent(BaseModel):
    """Single fertilizer application expressed as days after planting."""

    days_after_planting: Annotated[int, Field(ge=0)]
    n_kg_ha: Annotated[float, Field(ge=0.0, default=0.0)]
    p_kg_ha: Annotated[float, Field(ge=0.0, default=0.0)]


class CropSpec(BaseModel):
    name: Annotated[str, Field(description="Crop name, e.g. 'Maize'")]
    cultivar: Annotated[str | None, Field(default=None, description="DSSAT cultivar ID")]

    @field_validator("name", mode="before")
    @classmethod
    def _title(cls, v: str) -> str:
        return v.strip().title()


class ManagementSpec(BaseModel):
    planting_date: Annotated[date, Field(description="Base planting date (YYYY-MM-DD)")]
    n_planting_windows: Annotated[int, Field(default=1, ge=1)]
    planting_window_days: Annotated[int, Field(default=14, ge=1)]
    fertilizer_schedule: Annotated[list[FertilizerEvent], Field(default_factory=list)]

    @model_validator(mode="after")
    def _sort_fert(self) -> ManagementSpec:
        if self.fertilizer_schedule:
            self.fertilizer_schedule = sorted(
                self.fertilizer_schedule, key=lambda e: e.days_after_planting
            )
        return self


class SimSpec(BaseModel):
    country: str
    country_code: Annotated[str, Field(min_length=2, max_length=5)]
    working_path: Annotated[str, Field(description="Run directory — no spaces (Fortran constraint)")]
    model: Annotated[
        Literal["dssat", "banana_n", "simple_model", "caf"],
        Field(default="dssat"),
    ]
    dssat_path: Annotated[str | None, Field(default=None)]
    ncores: Annotated[int, Field(default=4, ge=1)]
    max_pixels: Annotated[int | None, Field(default=None, ge=1,
        description="Cap pixel count for quick tests; null = all pixels")]
    feature: Annotated[str | None, Field(default=None,
        description="Admin-unit name to restrict simulation (null = full bbox)")]
    adm_level: Annotated[int, Field(default=2, ge=0, le=5)]

    @field_validator("country_code", mode="before")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("working_path", mode="after")
    @classmethod
    def _warn_spaces(cls, v: str) -> str:
        if " " in v:
            logger.warning(
                "working_path '%s' contains spaces — DSSAT will silently fail (rc=99). "
                "Use a path without spaces.",
                v,
            )
        return v


# ---------------------------------------------------------------------------
# Mode 1 — with_cubes
# ---------------------------------------------------------------------------

class WithCubesConfig(BaseModel):
    """Config when weather + soil datacubes already exist on disk."""

    mode: Literal["with_cubes"]
    weather_path: Annotated[str, Field(description="Path to climate NetCDF (CF names: pr/tasmax/tasmin/rsds)")]
    soil_path: Annotated[str, Field(description="Path to soil NetCDF (depth x y x)")]
    output_path: Annotated[str, Field(description="Where to write the yield NetCDF")]
    simulation: SimSpec
    crop: CropSpec
    management: ManagementSpec

    @model_validator(mode="after")
    def _files_exist(self) -> WithCubesConfig:
        for label, p in [("weather_path", self.weather_path), ("soil_path", self.soil_path)]:
            if not Path(p).exists():
                raise ValueError(f"{label} does not exist: {p!r}")
        return self


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


# ---------------------------------------------------------------------------
# Discriminated union + loader
# ---------------------------------------------------------------------------

RunConfig = Annotated[
    WithCubesConfig | FullPipelineConfig,
    Field(discriminator="mode"),
]


def load_run_config(path: str | Path) -> WithCubesConfig | FullPipelineConfig:
    """Parse and validate a run-config YAML file.

    Raises
    ------
    pydantic.ValidationError
        If the YAML content does not match the expected schema.
    FileNotFoundError
        If the config file does not exist.
    """
    from pydantic import TypeAdapter

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        raw = yaml.safe_load(fh)

    adapter: TypeAdapter[WithCubesConfig | FullPipelineConfig] = TypeAdapter(
        Annotated[WithCubesConfig | FullPipelineConfig, Field(discriminator="mode")]
    )
    return adapter.validate_python(raw)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_with_cubes(cfg: WithCubesConfig, dry_run: bool = False) -> dict:
    """Execute mode: with_cubes.

    Builds a ``SimulationConfig`` YAML from the CLI's own schema, then hands
    it to :class:`~ag_cube_cm.spatial.spatial_cm.SpatialCM` — the same
    orchestrator the MCP server wraps, called directly so there is no JSON
    round-trip through the MCP layer.
    """
    sim = cfg.simulation
    mgmt = cfg.management

    sim_yaml: dict = {
        "GENERAL_INFO": {
            "country":      sim.country,
            "country_code": sim.country_code,
            "model":        sim.model,
            "working_path": sim.working_path,
            "dssat_path":   sim.dssat_path,
            "ncores":       sim.ncores,
        },
        "SPATIAL_INFO": {
            "feature_name": "shapeName",
            "adm_level":    sim.adm_level,
            "feature":      sim.feature,
            "weather_path": cfg.weather_path,
            "soil_path":    cfg.soil_path,
            "output_path":  cfg.output_path,
            "dem_path":     None,
        },
        "CROP": {
            "name":     cfg.crop.name,
            "cultivar": cfg.crop.cultivar,
        },
        "MANAGEMENT": {
            "planting_date":        str(mgmt.planting_date),
            "n_planting_windows":   mgmt.n_planting_windows,
            "planting_window_days": mgmt.planting_window_days,
        },
    }

    if mgmt.fertilizer_schedule:
        sim_yaml["MANAGEMENT"]["fertilizer_schedule"] = [
            {"days_after_planting": e.days_after_planting,
             "n_kg_ha": e.n_kg_ha, "p_kg_ha": e.p_kg_ha}
            for e in mgmt.fertilizer_schedule
        ]

    tmp_cfg = Path(sim.working_path) / "_run_config.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg, "w") as fh:
        yaml.dump(sim_yaml, fh, default_flow_style=False, sort_keys=False)

    if dry_run:
        click.echo(f"[dry-run] Would run SpatialCM(config_path={tmp_cfg}, "
                   f"max_pixels={sim.max_pixels}, feature={sim.feature!r})")
        return {"status": "dry_run"}

    from ag_cube_cm.config.loader import load_config
    from ag_cube_cm.spatial.spatial_cm import SpatialCM

    try:
        with SpatialCM(load_config(str(tmp_cfg))) as cm:
            result = cm.run(
                feature=sim.feature,
                adm_level=sim.adm_level,
                max_pixels=sim.max_pixels,
            )
        payload = {"status": "ok", **result.summary()}
    finally:
        tmp_cfg.unlink(missing_ok=True)
    return payload


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
    weather_nc = climate_dir / f"climate_{cfg.suffix}_{sy}_{ey}.nc"
    soil_nc = base / f"soil_{cfg.suffix}.nc"
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
    if raw_dir.exists() and any(raw_dir.rglob("*.nc")):
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


# ---------------------------------------------------------------------------
# Template YAMLs
# ---------------------------------------------------------------------------

_TEMPLATE_WITH_CUBES = """\
mode: with_cubes

# Paths to pre-built NetCDF datacubes (produced by aggeodata)
weather_path: /data/climate_hnd_2020_2022.nc
soil_path:    /data/soil_hnd.nc
output_path:  /data/yield_hnd.nc

simulation:
  country:      Honduras
  country_code: HND
  model:        dssat
  working_path: /dssat_runs   # no spaces in path
  ncores:       4
  max_pixels:   null          # null = all pixels; integer = quick test cap
  feature:      null          # null = entire bbox; e.g. "Comayagua"
  adm_level:    2

crop:
  name:      Maize
  cultivar:  IB1072

management:
  planting_date:        "2021-05-15"
  n_planting_windows:   3
  planting_window_days: 14
  fertilizer_schedule:  []
  # Uncomment to add fertilizer:
  # fertilizer_schedule:
  #   - days_after_planting: 15
  #     n_kg_ha: 60.0
  #     p_kg_ha: 30.0
  #   - days_after_planting: 45
  #     n_kg_ha: 30.0
"""

_TEMPLATE_FULL_PIPELINE = """\
mode: full_pipeline

output_dir: /ag_cube_example/hnd
suffix:     hnd_comayagua

spatial:
  bbox: [-87.8, 14.0, -86.8, 15.0]   # [xmin, ymin, xmax, ymax] WGS-84

dates:
  start: "2020-01-01"
  end:   "2022-12-31"

climate:
  # Same shape as aggeodata's climate-data-download YAML: each variable carries
  # its own source plus any per-variable options (gee_project, gee_dataset_id,
  # chirts_source, agera5_key, nasa_power_param).
  variables:
    pr:
      source: chirps     # precipitation (0.05 deg, no API key)
    tasmax:
      source: chirts     # max temperature (0.05 deg, no API key)
    tasmin:
      source: chirts     # min temperature
    rsds:
      source: agera5     # solar radiation (0.1 deg, CDS API key required)
    # rsds:
    #   source: nasa_power      # alternative: no API key, 0.5 deg
    # pr:
    #   source: gee
    #   gee_project: my-gcp-project   # required for 'gee' sources
  ncores:             2
  agera5_version:     "2_0"
  reference_variable: pr

soil:
  variables: [clay, sand, silt, bdod, cfvo, soc, phh2o, wv0010, wv0033, wv1500]
  depths:    ["0-5", "5-15", "15-30", "30-60", "60-100"]
  reference_variable: wv1500

simulation:
  country:      Honduras
  country_code: HND
  model:        dssat
  working_path: /dssat_runs
  ncores:       4
  max_pixels:   null
  feature:      null
  adm_level:    2

crop:
  name:      Maize
  cultivar:  IB1072

management:
  planting_date:        "2020-05-15"
  n_planting_windows:   3
  planting_window_days: 14
  fertilizer_schedule:  []
"""


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable INFO logging.")
def main(verbose: bool) -> None:
    """ag-cube-cm -- spatial crop model CLI (DSSAT / Banana-N)."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s  %(message)s")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True,
              help="Validate config and print planned steps without executing.")
def run(config: str, dry_run: bool) -> None:
    """Run a simulation from CONFIG (YAML).

    Mode is determined by the 'mode' key inside the file:

    \b
      with_cubes     Use pre-built weather + soil datacubes.
      full_pipeline  Download data, build cubes, then simulate.
    """
    from pydantic import ValidationError

    try:
        cfg = load_run_config(config)
    except (ValidationError, FileNotFoundError, ValueError) as exc:
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Mode : {cfg.mode}")
    click.echo(f"Crop : {cfg.crop.name}  ({cfg.crop.cultivar})")
    click.echo(f"Model: {cfg.simulation.model}  | cores={cfg.simulation.ncores}")

    if cfg.mode == "with_cubes":
        click.echo(f"Weather : {cfg.weather_path}")
        click.echo(f"Soil    : {cfg.soil_path}")
        click.echo(f"Output  : {cfg.output_path}")
        result = _run_with_cubes(cfg, dry_run=dry_run)
    else:
        click.echo(f"BBox    : {cfg.spatial.bbox}")
        click.echo(f"Period  : {cfg.dates.start} -> {cfg.dates.end}")
        click.echo(f"OutDir  : {cfg.output_dir}")
        result = _run_full_pipeline(cfg, dry_run=dry_run)

    if dry_run:
        return

    if result.get("status") == "ok":
        click.echo("\nSimulation complete.")
        click.echo(f"  Pixels ok      : {result.get('pixels_ok')}")
        click.echo(f"  Pixels skipped : {result.get('pixels_skipped')}")
        click.echo(f"  Pixels failed  : {result.get('pixels_failed')}")
        click.echo(f"  Mean HWAM      : {result.get('mean_hwam_kg_ha')} kg/ha")
        click.echo(f"  Output         : {result.get('output_path')}")
    else:
        click.echo(f"\nSimulation failed: {result.get('message', result)}", err=True)
        sys.exit(1)


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def validate(config: str) -> None:
    """Validate a CONFIG file and print a summary without running anything."""
    from pydantic import ValidationError

    try:
        cfg = load_run_config(config)
    except FileNotFoundError as exc:
        click.echo(f"File not found: {exc}", err=True)
        sys.exit(1)
    except ValidationError as exc:
        click.echo(f"Validation failed:\n{exc}", err=True)
        sys.exit(1)
    except ValueError as exc:
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Config OK  (mode={cfg.mode})")
    click.echo(f"  Crop    : {cfg.crop.name}  cultivar={cfg.crop.cultivar}")
    click.echo(f"  Model   : {cfg.simulation.model}")
    click.echo(f"  Planting: {cfg.management.planting_date}  "
               f"x{cfg.management.n_planting_windows} windows "
               f"(step={cfg.management.planting_window_days}d)")
    if cfg.management.fertilizer_schedule:
        total_n = sum(e.n_kg_ha for e in cfg.management.fertilizer_schedule)
        click.echo(f"  Fert    : {len(cfg.management.fertilizer_schedule)} events, "
                   f"total N={total_n} kg/ha")
    else:
        click.echo("  Fert    : none (rainfed baseline)")

    if cfg.mode == "with_cubes":
        click.echo(f"  Weather : {cfg.weather_path}")
        click.echo(f"  Soil    : {cfg.soil_path}")
        click.echo(f"  Output  : {cfg.output_path}")
    else:
        click.echo(f"  BBox    : {cfg.spatial.bbox}")
        click.echo(f"  Period  : {cfg.dates.start} -> {cfg.dates.end}")
        clim_summary = {
            v: cfg.climate.variables[v].source for v in cfg.climate.variables
        }
        click.echo(f"  Climate : {clim_summary}")
        click.echo(f"  OutDir  : {cfg.output_dir}/{cfg.suffix}")


@main.command()
@click.argument("template_mode",
                type=click.Choice(["with_cubes", "full_pipeline"]), metavar="MODE")
def template(template_mode: str) -> None:
    """Print a starter YAML template to stdout.

    MODE is either 'with_cubes' or 'full_pipeline'.
    Redirect to a file:  ag-cube-cm template with_cubes > run.yaml
    """
    if template_mode == "with_cubes":
        click.echo(_TEMPLATE_WITH_CUBES)
    else:
        click.echo(_TEMPLATE_FULL_PIPELINE)


if __name__ == "__main__":
    main()
