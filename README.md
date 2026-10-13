# ag-cube-cm — Spatial Crop Model Orchestration

**ag-cube-cm** is a Python package for running process-based crop models (PBMs) over spatial domains. It takes pre-built climate and soil datacubes as inputs, runs pixel-level simulations in parallel, and produces yield potential maps — gridded NetCDF outputs showing grain yield (kg/ha) across planting windows, years, and space.

The package is the **crop modeling layer** of a two-package workflow. Data acquisition (CHIRPS, CHIRTS, AgERA5, SoilGrids) is handled by the companion package [**aggeodata**](https://github.com/CGIAR-Climate-Data-Hub/aggeodata).

Supported process-based crop models:
- **[DSSAT](https://dssat.net/)** — industry-standard model covering maize, wheat, rice, soybean, and ~14 other crops.
- **[CAF2021](https://doi.org/10.1007/s10457-022-00755-6)** — agroforestry model for coffee under shade trees.
- **[SIMPLE](https://doi.org/10.1016/j.eja.2019.01.009)** — generic single-crop model, easily parameterized.
- **Banana-N** — pure-Python banana growth and nitrogen dynamics model.

---

## Two-Package Workflow

```
[aggeodata]                         [ag-cube-cm]
  download_chirps  ┐                  generate_config
  download_chirts  ├─► build_climate_datacube ─┐
  download_agera5  ┘     climate_<suffix>.nc   │
                                               ├─► run_simulation ─► yield_<suffix>.nc
  download_soil ──────► build_soil_datacube ───┘     (HWAM kg/ha, planting_window×year×y×x)
                          soil_<suffix>.nc
```

`ag-cube-cm` is purely a consumer: it reads the `.nc` datacubes and orchestrates the crop model. It performs no data downloads.

---

## Installation

```bash
# Core (spatial orchestration only)
pip install git+https://github.com/CGIAR-Climate-Data-Hub/ag-cube-cm.git

# With DSSAT and other crop models
pip install "ag-cube-cm[models] @ git+https://github.com/CGIAR-Climate-Data-Hub/ag-cube-cm.git"

# With MCP server (AI assistant integration)
pip install "ag-cube-cm[mcp] @ git+https://github.com/CGIAR-Climate-Data-Hub/ag-cube-cm.git"

# Everything
pip install "ag-cube-cm[all] @ git+https://github.com/CGIAR-Climate-Data-Hub/ag-cube-cm.git"
```

> **DSSAT binary:** bundled in `models/dssat/static/bin/` for Linux and Windows. No separate DSSAT installation needed.

---

## How to Use

### Step 1 — Build datacubes with aggeodata

Install the companion package and build climate and soil datacubes for your region of interest. See the [aggeodata repository](https://github.com/CGIAR-Climate-Data-Hub/aggeodata) for details.

```bash
pip install aggeodata
```

The two datacubes you need as inputs:

| File | Dimensions | Variables |
|------|-----------|-----------|
| `climate_<suffix>_YYYY_YYYY.nc` | `time × y × x` | `pr` (mm/d), `tasmax` (°C), `tasmin` (°C), `rsds` (J/m²/d) |
| `soil_<suffix>.nc` | `depth × y × x` | `clay`, `sand`, `bdod`, `wv0033`, `wv1500`, … |

CF variable names from aggeodata (`pr`, `tasmax`, `tasmin`, `rsds`) are automatically remapped to DSSAT-expected names inside `DSSATModel.prepare_inputs()` — no manual renaming needed.

---

### Step 2 — Write a simulation config

```yaml
# dssat_maize_hnd.yaml

GENERAL_INFO:
  country:      'Honduras'
  country_code: 'HND'
  model:        'dssat'
  working_path: 'D:/dssat_runs'   # ← NO spaces (Fortran path constraint)
  dssat_path:   null              # null = use bundled binary
  ncores:       8

SPATIAL_INFO:
  feature_name: 'shapeName'       # column used for sub-region clipping
  adm_level:    2
  feature:      null              # null = entire datacube extent
                                  # e.g. 'Comayagua' = clip to one district
  weather_path: 'data/climate_hnd_comayagua_2020_2022.nc'
  soil_path:    'data/soil_hnd_comayagua.nc'
  output_path:  'results/yield_hnd_comayagua.nc'
  dem_path:     null

CROP:
  name:     'Maize'
  cultivar: 'IB1072'    # tropical maize; see list_supported_crops for options

MANAGEMENT:
  planting_date:        '2020-05-01'
  n_planting_windows:   4           # number of staggered planting scenarios
  planting_window_days: 14          # days between consecutive windows
  # Optional fertilizer schedule (rainfed = omit):
  # fertilizer_schedule:
  #   - days_after_planting: 15
  #     n_kg_ha: 60.0
  #     p_kg_ha: 30.0
```

> **`working_path` must not contain spaces.** DSSAT's `DSSATPRO.V48` file is whitespace-delimited; a space in the path corrupts the model entry and causes a silent `rc=99` error.

---

### Step 3 — Run the simulation

```python
from ag_cube_cm import SpatialCM, load_config

with SpatialCM(load_config("dssat_maize_hnd.yaml")) as cm:
    result = cm.run()            # feature=..., max_pixels=... to narrow the run

print(f"Mean yield : {result.mean_yield} kg/ha")
print(f"Pixels OK  : {result.pixels_ok}")
print(f"Output     : {result.output_path}")
result.cube                      # the in-memory yield datacube
```

`SpatialCM` is the single orchestrator; the CLI and the MCP server are thin
wrappers over it. From an MCP client the same run is:

```python
from ag_cube_cm.mcp_server import run_simulation
import json

result = json.loads(run_simulation(config_path="dssat_maize_hnd.yaml"))
print(result["mean_hwam_kg_ha"], result["pixels_ok"])
```

Or run all three steps end-to-end with the bundled example script:

```bash
python examples/end_to_end_workflow.py
```

---

### Yield output structure

The output NetCDF (`yield_<suffix>.nc`) has four dimensions:

```
Dimensions : planting_window × year × y × x
Variables  :
  HWAM (kg/ha)  — grain yield at maturity
  flag          — 0 = ok, 1 = model error, 2 = skipped (no data)
```

Typical post-processing:

```python
import xarray as xr
import matplotlib.pyplot as plt

ds   = xr.open_dataset("results/yield_hnd_comayagua.nc")
hwam = ds["HWAM"]   # (planting_window, year, y, x)

# Mean potential yield across all windows and years
mean_yield = hwam.mean(dim=["planting_window", "year"])

# Inter-annual variability (CV %)
cv = hwam.std(dim="year").mean(dim="planting_window") / mean_yield * 100

# Best planting window (maximises mean yield)
best_window = hwam.mean(dim="year").argmax(dim="planting_window")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
mean_yield.plot(ax=axes[0], cmap="YlGn",    cbar_kwargs={"label": "Mean yield (kg/ha)"})
cv.plot(        ax=axes[1], cmap="RdYlGn_r",cbar_kwargs={"label": "CV (%)"})
best_window.plot(ax=axes[2], cmap="viridis", cbar_kwargs={"label": "Best window (index)"})
plt.tight_layout()
plt.savefig("yield_map.png", dpi=150)
```

---

## Full End-to-End Example

See [`examples/end_to_end_workflow.py`](examples/end_to_end_workflow.py) for a complete
runnable script covering:

1. aggeodata config (CHIRPS + CHIRTS + AgERA5)
2. Climate data download → datacube assembly
3. SoilGrids download → soil datacube assembly
4. DSSAT simulation across all pixels × planting windows
5. 3-panel yield distribution map (mean yield, CV%, optimal planting window)

Region: Honduras — Comayagua valley | Period: 2020–2022 | Crop: Maize IB1072

---

## AI-Assisted Workflow (MCP)

The package ships an MCP server so that an AI assistant (e.g. Claude Code) can drive
the full workflow from a single natural-language request.

### Register the server

Add to `.claude/mcp_config.json` (or `~/.claude/mcp_config.json` for global):

```json
{
  "mcpServers": {
    "ag-cube-cm": {
      "command": "python",
      "args": ["-m", "ag_cube_cm.mcp_server"],
      "env": { "PYTHONPATH": "${workspaceFolder}/src" }
    }
  }
}
```

Start the server:
```bash
python -m ag_cube_cm.mcp_server
```

### Available tools

| Tool | Key parameters | What it does |
|------|---------------|-------------|
| `list_admin_units` | `country_code`, `adm_level` | Lists district/department names — call before simulation to verify `feature` names |
| `list_supported_crops` | — | Lists all supported crops and example cultivar IDs |
| `generate_config` | `country`, `weather_path`, `soil_path`, `crop`, `cultivar`, `planting_date`, `ncores`, `save_to` | Generates and saves a simulation YAML config |
| `run_simulation` | `config_path`, `feature`, `adm_level`, `max_pixels` | Runs the crop model on all pixels × planting windows |

> **Data downloads** (`download_weather`, `download_soil`, `build_climate_datacube`, `build_soil_datacube`) are in the **aggeodata** MCP server. Use that server first, then pass the resulting `.nc` paths here.

### Output chaining

| After calling (aggeodata) | Read field | Pass to ag-cube-cm |
|---------------------------|-----------|-------------------|
| `build_climate_datacube` | `output_path` | `weather_path` in `generate_config` |
| `build_soil_datacube` | `output_path` | `soil_path` in `generate_config` |
| `generate_config` | `save_path` | `config_path` in `run_simulation` |

### Example conversation

```
You:     Simulate maize yield potential in Mwanza district, Malawi,
         2010–2012, planting 2010-11-01, 4 windows, no fertilizer, 8 cores.
         I already have weather_mwi.nc and soil_mwi.nc.

Claude:  [list_admin_units(country_code="MWI", adm_level=2)]
         → confirms "Mwanza" is a valid district name

         [generate_config(country="Malawi", country_code="MWI",
                          model="dssat", crop="Maize", cultivar="IB1072",
                          weather_path="weather_mwi.nc",
                          soil_path="soil_mwi.nc",
                          planting_date="2010-11-01",
                          n_planting_windows=4,
                          ncores=8,
                          save_to="mwanza.yaml")]

         [run_simulation(config_path="mwanza.yaml",
                         feature="Mwanza", adm_level=2)]
         → 87 pixels × 4 windows × 3 years
         → mean HWAM: 4 210 kg/ha
         → output: mwanza_yield.nc
```

---

## Package Architecture

```
src/ag_cube_cm/
├── config/
│   ├── schemas.py     Pydantic v2 SimulationConfig — validates all YAML inputs
│   └── loader.py      load_config() — parse + validate YAML
├── ingestion/
│   └── boundaries.py  get_admin_boundary() — GeoBoundaries API lookup for sub-region clipping
├── models/
│   ├── base.py        CropModel ABC — prepare_inputs / run_simulation / collect_outputs
│   ├── factory.py     @register_model decorator + model_factory()
│   ├── dssat/         DSSATModel — Fortran file writers (.WTH .SOL .X), subprocess runner, Summary.OUT parser
│   ├── banana_n/      Pure-Python banana growth + nitrogen model
│   └── caf2021/       Coffee-shade agroforestry model
├── spatial/
│   ├── spatial_cm.py  SpatialCM — THE orchestrator: pixel × window × season sweep,
│   │                  parallel execution, result assembly. Returns SimulationResult.
│   ├── data.py        SpatialData — the single file-open path for cubes
│   ├── raster_ops.py  clip_to_bbox, mask_with_geometry, get_roi_data
│   └── reporter.py    SpatialReporter — grid assembly + NetCDF / Parquet writer
├── transform/
│   └── soil_cube.py   SoilDataCubeBuilder — stacks SoilGrids GeoTIFFs into depth×y×x NetCDF
├── cli.py             `ag-cube-cm run` — calls SpatialCM directly
└── mcp_server.py      FastMCP server (4 tools) — thin JSON wrapper over SpatialCM
```

Both front ends delegate to `SpatialCM`, so orchestration behaviour cannot
drift between the CLI and the MCP server.

---

## Supported Crops (DSSAT)

| Crop | Code | | Crop | Code |
|------|----- |-|------|------|
| Maize | MZ | | Bean | BN |
| Wheat | WH | | Cassava | CS |
| Rice | RI | | Potato | PT |
| Sorghum | SG | | Sugarcane | SC |
| Millet | ML | | Sunflower | SU |
| Soybean | SB | | Canola | CN |

Run `list_supported_crops()` from the MCP server or Python API for the full list with example cultivar IDs.

---

## Requirements

- Python ≥ 3.10
- DSSAT binary — bundled in `models/dssat/static/bin/` (Linux + Windows)
- Input datacubes built with [aggeodata](https://github.com/anaguilarar/aggeodata)

---

## References

Lizaso, J.I., et al. (2011). DSSAT v4.5 crop models. *Agricultural Systems*.

Van Oijen, M., Haggar, J., et al. (2022). Ecosystem services from coffee agroforestry in Central America: estimation using the CAF2021 model. *Agroforestry Systems*. https://doi.org/10.1007/s10457-022-00755-6

Zhao, C., et al. (2019). A SIMPLE crop model. *European Journal of Agronomy*. https://doi.org/10.1016/j.eja.2019.01.009

---

## License

MIT
