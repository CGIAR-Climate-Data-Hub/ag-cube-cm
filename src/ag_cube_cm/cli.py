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

logger = logging.getLogger(__name__)

# `full_pipeline` mode lives in ag_cube_cm.pipeline, which is the only module
# that imports aggeodata. It is loaded on demand (see _full_pipeline_config)
# so `with_cubes` runs — and the CLI itself — work without the data-acquisition
# dependencies installed.

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
# Discriminated union + loader
# ---------------------------------------------------------------------------

def _full_pipeline_config() -> type[BaseModel]:
    """Return ``FullPipelineConfig``, importing the aggeodata-backed module.

    Kept behind a function so the import cost — and the aggeodata dependency —
    is paid only by configs that actually ask for ``mode: full_pipeline``.
    """
    from ag_cube_cm.pipeline import FullPipelineConfig

    return FullPipelineConfig


def load_run_config(path: str | Path) -> BaseModel:
    """Parse and validate a run-config YAML file.

    The ``mode`` key is read first and only the matching schema is loaded, so a
    ``with_cubes`` config never pulls in the data-acquisition dependencies.

    Returns
    -------
    WithCubesConfig | FullPipelineConfig

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If ``mode`` is missing or unrecognised.
    ImportError
        If ``mode: full_pipeline`` is requested without the ``pipeline`` extra.
    pydantic.ValidationError
        If the YAML content does not match the expected schema.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")

    mode = raw.get("mode")
    if mode == "with_cubes":
        return WithCubesConfig.model_validate(raw)
    if mode == "full_pipeline":
        return _full_pipeline_config().model_validate(raw)

    raise ValueError(
        f"Unknown mode {mode!r}. Expected 'with_cubes' or 'full_pipeline'."
    )


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
        from ag_cube_cm.pipeline import _run_full_pipeline

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
