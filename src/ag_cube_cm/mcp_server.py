"""
ag-cube-cm MCP server
=====================
Exposes spatial crop-model operations as MCP tools.

Typical workflow (use the **aggeodata** MCP server first for data downloads):
  aggeodata: download_chirps / download_agera5 / download_soil
  aggeodata: build_climate_datacube / build_soil_datacube
  ag-cube-cm: generate_config  →  run_simulation

Start the server:
    python -m ag_cube_cm.mcp_server
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "ag-cube-cm",
    instructions=(
        "Tools for running DSSAT/BANANA_N crop model simulations on spatial "
        "datacubes produced by the aggeodata package. Use the aggeodata MCP "
        "server first to download climate and soil data."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(payload: Any) -> str:
    return json.dumps({"status": "ok", **payload}, default=str)


def _err(msg: str) -> str:
    return json.dumps({"status": "error", "message": msg})


# ---------------------------------------------------------------------------
# Tool 1 — list_admin_units
# ---------------------------------------------------------------------------

@mcp.tool()
def list_admin_units(
    country_code: str,
    adm_level: int = 2,
) -> str:
    """List all administrative unit names for a country at a given level.

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-3 code (e.g. 'MWI', 'HND', 'COL').
    adm_level : int
        1 = region/province, 2 = district/department (default), 3 = sub-district.

    Returns
    -------
    JSON with status, country_code, adm_level, count, and sorted list of names.
    """
    try:
        from ag_cube_cm.ingestion.boundaries import list_admin_units as _list

        names = _list(country_code, adm_level=adm_level)
        return _ok({
            "country_code": country_code.upper(),
            "adm_level": adm_level,
            "count": len(names),
            "units": names,
        })
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-600:]}")


# ---------------------------------------------------------------------------
# Tool 2 — generate_config
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_config(
    country: str,
    country_code: str,
    model: str,
    weather_path: str,
    soil_path: str,
    crop: str,
    cultivar: str,
    planting_date: str,
    output_path: str,
    working_path: str,
    dssat_path: str | None = None,
    n_planting_windows: int = 1,
    planting_window_days: int = 7,
    ncores: int = 4,
    fertilizer_n_kg_ha: float = 0.0,
    fertilizer_p_kg_ha: float = 0.0,
    feature: str | None = None,
    adm_level: int = 2,
    save_to: str | None = None,
) -> str:
    """Generate and optionally save a simulation YAML config file.

    Parameters
    ----------
    country : str           Full country name (e.g. 'Malawi').
    country_code : str      ISO 3-letter code (e.g. 'MWI').
    model : str             'dssat', 'banana_n', 'simple_model', or 'caf'.
    weather_path : str      Path to the climate NetCDF datacube from aggeodata.
    soil_path : str         Path to the soil NetCDF datacube from aggeodata.
    crop : str              Crop name (e.g. 'Maize', 'Wheat', 'Bean').
    cultivar : str          DSSAT cultivar ID (e.g. 'IB1072').
    planting_date : str     Base planting date 'YYYY-MM-DD'.
    output_path : str       Where to save the yield NetCDF output.
    working_path : str      DSSAT run working directory (NO spaces in path).
    dssat_path : str | None DSSAT installation root (None = use bundled binary).
    n_planting_windows : int  Number of planting windows to simulate.
    planting_window_days : int  Days between consecutive windows.
    ncores : int            Parallel threads.
    fertilizer_n_kg_ha : float  N applied at planting (kg/ha). 0 = no fertilizer.
    fertilizer_p_kg_ha : float  P applied at planting (kg/ha).
    feature : str | None    Admin unit to restrict the simulation to.
    adm_level : int         Admin level for the feature boundary (default 2).
    save_to : str | None    If given, writes the YAML to this file path.

    Returns
    -------
    JSON with status, config_yaml (string), and save_path.
    """
    try:
        space_warning = (
            f"WARNING: working_path '{working_path}' contains spaces. "
            "DSSAT will silently fail (rc=99). Use a path without spaces."
            if " " in working_path else None
        )

        fert_block = ""
        if fertilizer_n_kg_ha > 0 or fertilizer_p_kg_ha > 0:
            fert_block = (
                "  fertilizer_schedule:\n"
                f"    - days_after_planting: 5\n"
                f"      n_kg_ha: {fertilizer_n_kg_ha}\n"
                f"      p_kg_ha: {fertilizer_p_kg_ha}\n"
            )

        dssat_line = (
            f"  dssat_path: '{dssat_path}'\n" if dssat_path
            else "  dssat_path: null\n"
        )

        feature_line = (
            f"  feature: '{feature}'\n" if feature else "  feature: null\n"
        )

        yaml_text = (
            f"GENERAL_INFO:\n"
            f"  country: '{country}'\n"
            f"  country_code: '{country_code.upper()}'\n"
            f"  model: '{model}'\n"
            f"  working_path: '{working_path}'\n"
            f"{dssat_line}"
            f"  ncores: {ncores}\n"
            f"\n"
            f"SPATIAL_INFO:\n"
            f"  feature_name: 'shapeName'\n"
            f"  adm_level: {adm_level}\n"
            f"{feature_line}"
            f"  soil_path: '{soil_path}'\n"
            f"  weather_path: '{weather_path}'\n"
            f"  output_path: '{output_path}'\n"
            f"  dem_path: null\n"
            f"\n"
            f"CROP:\n"
            f"  name: '{crop}'\n"
            f"  cultivar: '{cultivar}'\n"
            f"\n"
            f"MANAGEMENT:\n"
            f"  planting_date: '{planting_date}'\n"
            f"  n_planting_windows: {n_planting_windows}\n"
            f"  planting_window_days: {planting_window_days}\n"
            f"{fert_block}"
        )

        save_path = None
        if save_to:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            with open(save_to, "w") as fh:
                fh.write(yaml_text)
            save_path = save_to

        payload: dict[str, Any] = {"config_yaml": yaml_text, "save_path": save_path}
        if space_warning:
            payload["warning"] = space_warning
        return _ok(payload)
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Tool 3 — list_supported_crops
# ---------------------------------------------------------------------------

@mcp.tool()
def list_supported_crops() -> str:
    """List all crops supported by the DSSAT model and their 2-letter codes."""
    crops = {
        "Maize": "MZ", "Wheat": "WH", "Rice": "RI", "Sorghum": "SG",
        "Millet": "ML", "Soybean": "SB", "Bean": "BN", "Cassava": "CS",
        "Potato": "PT", "Sugarcane": "SC", "Sugarbeet": "BS",
        "Sunflower": "SU", "Canola": "CN", "Tomato": "TM",
        "Cabbage": "CB", "Alfalfa": "AL", "Bermudagrass": "BM",
    }
    common_cultivars = {
        "Maize":   ["IB1072 (tropical)", "PC0002 (temperate)", "MEDIUM (generic)"],
        "Wheat":   ["IB1015 (spring)", "IB1487 (winter)"],
        "Bean":    ["IB0001"],
        "Soybean": ["IB0001"],
    }
    return _ok({"crops": crops, "example_cultivars": common_cultivars})


# ---------------------------------------------------------------------------
# Tool 4 — run_simulation
# ---------------------------------------------------------------------------

@mcp.tool()
def run_simulation(
    config_path: str,
    max_pixels: int | None = None,
    feature: str | None = None,
    adm_level: int = 2,
) -> str:
    """Run a spatial crop model simulation from a YAML config file.

    Prerequisites
    -------------
    Climate and soil NetCDF datacubes must already exist on disk (produced by
    the **aggeodata** MCP server tools ``build_climate_datacube`` and
    ``build_soil_datacube``).

    Parameters
    ----------
    config_path : str
        Path to the simulation YAML config (generated by generate_config or
        written manually).
    max_pixels : int | None
        Optional cap on number of pixels to simulate (useful for quick tests).
        None = run all land pixels.
    feature : str | None
        Admin unit name to restrict the simulation to (e.g. 'Zomba').
        Overrides the feature set in the config file.
    adm_level : int
        Administrative level for the feature boundary lookup (default 2).

    Returns
    -------
    JSON with status, output_path, pixel counts (ok/skip/failed), and
    mean HWAM yield when available.
    """
    try:
        from ag_cube_cm.config.loader import load_config
        from ag_cube_cm.spatial.spatial_cm import SpatialCM

        cfg = load_config(config_path)
        with SpatialCM(cfg) as cm:
            result = cm.run(
                feature=feature,
                adm_level=adm_level if feature else None,
                max_pixels=max_pixels,
            )
        return _ok(result.summary())
    except Exception as exc:
        tb = traceback.format_exc()[-800:]
        return _err(f"{type(exc).__name__}: {exc}" + chr(10) + tb)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
