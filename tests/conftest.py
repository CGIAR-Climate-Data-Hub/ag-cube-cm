"""Shared fixtures for the ag-cube-cm test suite.

Everything here is synthetic and deterministic: the suite runs with no network,
no SoilGrids/CHIRPS download, and no DSSAT binary.  That is deliberate — these
tests exist to catch changes in the *files we hand to DSSAT*, which is the layer
that soil-science work touches, and that layer is pure computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ag_cube_cm.config.schemas import SimulationConfig


def pytest_addoption(parser) -> None:
    """Add ``--update-golden`` for deliberate fixture regeneration."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite the golden DSSAT input fixtures instead of asserting "
             "against them. Use only alongside a reasoned explanation of why "
             "the files changed.",
    )


# A pixel in the Comayagua valley, Honduras — a real location, so latitude and
# day-length behave realistically, but the values below are synthetic.
TEST_LAT = 14.45
TEST_LON = -87.65

#: SoilGrids layer bottoms, in cm.
SOIL_DEPTHS = [5, 15, 30, 60, 100]


@pytest.fixture
def weather_slice() -> xr.Dataset:
    """One pixel of daily weather, in aggeodata CF names and units.

    Two full years of a smooth seasonal cycle plus a deterministic rainfall
    pattern.  No randomness: the golden files must be byte-stable.
    """
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    doy = dates.dayofyear.to_numpy().astype(float)

    # Smooth tropical seasonality; amplitude kept small, as in Central America.
    tasmax = 30.0 + 3.0 * np.sin(2 * np.pi * (doy - 100) / 365.0)
    tasmin = 18.0 + 2.5 * np.sin(2 * np.pi * (doy - 100) / 365.0)
    # Bimodal wet season, zero in the dry months — exercises the zero path.
    pr = np.where(
        (doy > 130) & (doy < 310),
        8.0 * np.abs(np.sin(2 * np.pi * doy / 45.0)),
        0.0,
    )
    # AgERA5 delivers radiation in J m-2 d-1; the writer must convert to MJ.
    rsds = (18.0 + 4.0 * np.sin(2 * np.pi * (doy - 80) / 365.0)) * 1e6

    return xr.Dataset(
        {
            "tasmax": (["time"], tasmax.round(3)),
            "tasmin": (["time"], tasmin.round(3)),
            "pr": (["time"], pr.round(3)),
            "rsds": (["time"], rsds.round(1)),
        },
        coords={"time": dates, "y": TEST_LAT, "x": TEST_LON},
    )


def _soil_profile(sand, clay, silt, soc, bdod, wv1500, wv0033, wv0010,
                  cfvo=50.0, phh2o=62.0, nitrogen=None) -> xr.Dataset:
    """Build a soil slice in raw SoilGrids units (the writer rescales them).

    When ``nitrogen`` is omitted it is derived from organic carbon at a C:N
    ratio of 10, which is typical of agricultural topsoils — so richer soils
    carry more nitrogen, as they do in the field.
    """
    n = len(SOIL_DEPTHS)
    if nitrogen is None:
        # soc (dg/kg) -> % is /100; nitrogen (cg/kg) -> % is /1000.
        # So a C:N ratio of 10 means nitrogen_cg_kg == soc_dg_kg.
        nitrogen = list(np.atleast_1d(np.asarray(soc, dtype=float)))

    def col(v):
        return (["depth"], np.asarray(v, dtype=float) if np.ndim(v) else np.full(n, float(v)))

    return xr.Dataset(
        {
            "sand": col(sand), "clay": col(clay), "silt": col(silt),
            "soc": col(soc), "bdod": col(bdod), "cfvo": col(cfvo),
            "phh2o": col(phh2o), "nitrogen": col(nitrogen),
            "wv1500": col(wv1500), "wv0033": col(wv0033), "wv0010": col(wv0010),
        },
        coords={"depth": SOIL_DEPTHS, "y": TEST_LAT, "x": TEST_LON},
    )


@pytest.fixture
def soil_slice() -> xr.Dataset:
    """A loam profile — the mid-range case.

    Units are raw SoilGrids: sand/clay/silt in g/kg, soc in dg/kg, bdod in
    cg/cm3, wv* in 10^-3 cm3/cm3, cfvo in cm3/dm3, phh2o in pH*10.
    """
    return _soil_profile(
        sand=[400, 400, 390, 380, 370],
        clay=[220, 230, 250, 270, 280],
        silt=[380, 370, 360, 350, 350],
        soc=[180, 140, 90, 50, 30],
        bdod=[130, 135, 140, 145, 150],
        wv1500=[140, 145, 155, 165, 170],
        wv0033=[280, 285, 290, 295, 300],
        wv0010=[420, 415, 410, 405, 400],
    )


@pytest.fixture
def sandy_soil_slice() -> xr.Dataset:
    """A sand profile — the free-draining extreme."""
    return _soil_profile(
        sand=[900, 895, 890, 885, 880],
        clay=[40, 45, 50, 55, 60],
        silt=[60, 60, 60, 60, 60],
        soc=[60, 40, 25, 15, 10],
        bdod=[155, 158, 160, 162, 165],
        wv1500=[50, 55, 60, 65, 70],
        wv0033=[120, 125, 130, 135, 140],
        wv0010=[380, 375, 370, 365, 360],
    )


@pytest.fixture
def clay_soil_slice() -> xr.Dataset:
    """A clay profile — the poorly-drained extreme."""
    return _soil_profile(
        sand=[150, 145, 140, 135, 130],
        clay=[550, 560, 580, 600, 610],
        silt=[300, 295, 280, 265, 260],
        soc=[220, 170, 110, 60, 35],
        bdod=[115, 118, 122, 126, 130],
        wv1500=[280, 285, 295, 305, 310],
        wv0033=[420, 425, 430, 435, 440],
        wv0010=[520, 515, 510, 505, 500],
    )


@pytest.fixture
def sim_config(tmp_path) -> SimulationConfig:
    """A minimal valid maize configuration pointing at a temp working dir."""
    return SimulationConfig.model_validate({
        "GENERAL_INFO": {
            "country": "Honduras",
            "country_code": "HND",
            "model": "dssat",
            "working_path": str(tmp_path / "dssat_runs"),
            "dssat_path": None,
            "ncores": 1,
        },
        "SPATIAL_INFO": {
            "feature_name": "shapeName",
            "adm_level": 2,
            "feature": None,
            "weather_path": str(tmp_path / "weather.nc"),
            "soil_path": str(tmp_path / "soil.nc"),
            "output_path": str(tmp_path / "yield.nc"),
            "dem_path": None,
        },
        "CROP": {"name": "Maize", "cultivar": "IB1072"},
        "MANAGEMENT": {
            "planting_date": "2020-05-01",
            "n_planting_windows": 1,
            "planting_window_days": 14,
        },
    })


@pytest.fixture
def prepared_model(sim_config, weather_slice, soil_slice):
    """A DSSATModel with inputs written into an isolated working directory."""
    from ag_cube_cm.models.dssat.base import DSSATModel

    model = DSSATModel(sim_config)
    model.setup_working_directory("test_px")
    model.prepare_inputs(weather_slice, soil_slice, elevation=600.0)
    yield model
    model.cleanup_working_directory()
