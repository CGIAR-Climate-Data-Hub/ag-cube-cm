"""
ag_cube_cm.models.dssat.base
=============================

DSSAT CropModel Implementation.
"""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import xarray as xr
from pyproj import Transformer

from ag_cube_cm.models._fileops import copy_retry
from ag_cube_cm.models.base import CropModel
from ag_cube_cm.models.dssat import soil as dssat_soil
from ag_cube_cm.models.factory import register_model

logger = logging.getLogger(__name__)


@register_model("dssat")
class DSSATModel(CropModel):
    """
    DSSAT Implementation of the CropModel ABC.

    This model orchestrates the writing of strict fixed-width Fortran files
    (.WTH, .SOL, .X), dynamically generates a batch file (DSSBatch.v48),
    executes the binary via subprocess, and parses Summary.OUT.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.crop = getattr(self.config.CROP, 'name', 'Maize').lower()
        self.cultivar = getattr(self.config.CROP, 'cultivar', None)
        self.crop_code = self._get_crop_code(self.crop)
        self.exp_filename = None
        self._soil_layers: list = []  # [(depth_cm, slll, sdul), ...]
        self._nyers: int = 1          # auto-detected from weather span in prepare_inputs

    def _get_crop_code(self, crop: str) -> str:
        """Map crop names to DSSAT 2-letter codes."""
        codes = {
            "maize": "MZ", "millet": "ML", "sugarbeet": "BS", "rice": "RI",
            "sorghum": "SG", "sweetcorn": "SW", "alfalfa": "AL", "bermudagrass": "BM",
            "soybean": "SB", "canola": "CN", "sunflower": "SU", "potato": "PT",
            "tomato": "TM", "cabbage": "CB", "sugarcane": "SC", "wheat": "WH",
            "bean": "BN", "cassava": "CS"
        }
        return codes.get(crop.lower(), "MZ")

    def _get_maturity_days(self) -> int:
        """Approximate days from planting to physiological maturity.

        Used to (1) decide whether the last simulated season's harvest would
        cross into a year with no weather data — drop the season if so;
        (2) write HDATE in the X-file (DSSAT parses it even though HARVS=R
        means harvest at maturity, not on this date).
        """
        return self._MATURITY_DAYS.get(self.crop, 120)

    def prepare_inputs(self, weather_slice: xr.Dataset, soil_slice: xr.Dataset,
                       elevation: float = -99.0) -> None:
        """
        Convert Xarray slices into strict DSSAT Fortran formats inside the
        isolated working directory.

        Parameters
        ----------
        elevation : float
            Site elevation in metres (written to .WTH header).  Pass -99 when
            no DEM is available.
        """
        if not self.working_dir:
            raise RuntimeError("Working directory not setup. Call setup_working_directory() first.")

        df_wth = weather_slice.to_dataframe().reset_index().dropna()
        df_sol = soil_slice.to_dataframe().reset_index().dropna()

        # Accept CF variable names produced by aggeodata and rename to DSSAT-expected names.
        # This allows using aggeodata datacubes directly without a manual renaming step.
        _cf_aliases = {
            "tasmax": "tmax", "tasmin": "tmin",
            "pr": "precipitation",
            "rsds": "solar_radiation",
            "time": "date",   # time-index column alias
        }
        df_wth = df_wth.rename(columns={k: v for k, v in _cf_aliases.items() if k in df_wth.columns})

        # Extract coordinates for header files — reproject to WGS84 if needed
        raw_y = float(df_sol['y'].iloc[0]) if 'y' in df_sol.columns else 0.0
        raw_x = float(df_sol['x'].iloc[0]) if 'x' in df_sol.columns else 0.0
        src_crs = None
        try:
            src_crs = soil_slice.rio.crs
        except Exception:
            pass

        # Explicit CRS check: to_epsg() returns None for non-EPSG projections like
        # IGH (SoilGrids native). Treat None as "definitely not 4326" and reproject.
        needs_reproject = False
        if src_crs is not None:
            src_epsg = src_crs.to_epsg()
            needs_reproject = (src_epsg != 4326)
        if needs_reproject:
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(raw_x, raw_y)
        else:
            lat, lon = raw_y, raw_x

        # Unit conversions
        # Temperatures: Kelvin → Celsius (AgERA5 delivers in K)
        for temp_col in ['tmin', 'tmax']:
            if temp_col in df_wth.columns and df_wth[temp_col].mean() > 100:
                df_wth[temp_col] = df_wth[temp_col] - 273.15
        # Solar radiation: J m-2 d-1 → MJ m-2 d-1 (AgERA5 delivers in J/m²/d)
        for _srad_col in ('solar_radiation', 'srad'):
            if _srad_col in df_wth.columns and df_wth[_srad_col].mean() > 10000:
                df_wth[_srad_col] = df_wth[_srad_col] / 1e6
                break

        if not df_wth.empty:
            self._write_wth(df_wth, lat, lon, elevation)
            # Auto-detect NYERS: seasons that fit between planting year and last weather year.
            # Config can override via number_years; 0 or None triggers auto-detection.
            cfg_nyers = getattr(self.config.GENERAL_INFO, 'number_years', None)
            if cfg_nyers:
                self._nyers = int(cfg_nyers)
            else:
                planting_date_val = getattr(self.config.MANAGEMENT, 'planting_date', None)
                pdate_dt = datetime.strptime(
                    str(planting_date_val) if planting_date_val else '2000-01-01', '%Y-%m-%d'
                )
                planting_year = pdate_dt.year
                last_wth_year = int(pd.to_datetime(df_wth['date']).dt.year.max())

                # Sanity: weather must reach at least the planting year. Clamping to
                # NYERS=1 here would just push the failure into DSSAT (rc=99); fail
                # fast with a clear message so the caller can fix the date range.
                if last_wth_year < planting_year:
                    raise RuntimeError(
                        f"Weather data ends in {last_wth_year} but planting is in "
                        f"{planting_year}. Extend the weather datacube to cover the "
                        f"planting year, or change planting_date."
                    )

                # If harvest (maturity days after planting) crosses into the next
                # calendar year, the last simulated season would need weather past
                # last_wth_year — drop it.
                maturity_days = self._get_maturity_days()
                harvest_crosses_year = (
                    pdate_dt + timedelta(days=maturity_days)
                ).year > pdate_dt.year
                nyers_raw = last_wth_year - planting_year + 1
                self._nyers = max(1, nyers_raw - (1 if harvest_crosses_year else 0))

        if not df_sol.empty:
            self._write_sol(df_sol, lat, lon)

        self._write_mzx(lat, lon)

    def _write_wth(self, df_wth: pd.DataFrame, lat: float, lon: float,
                   elevation: float = -99.0) -> None:
        """
        Write the .WTH file with strict Fortran 0-padded spacing.
        Maintains the exact column spacing required by DSSAT.
        Groups by year to write annual weather files.
        """
        # DSSAT requires strictly chronological rows — sort by date first.
        df_wth = df_wth.copy()
        # Accept either 'date' or 'time' as the timestamp column.
        if 'date' not in df_wth.columns and 'time' in df_wth.columns:
            df_wth = df_wth.rename(columns={'time': 'date'})
        df_wth['_date_parsed'] = pd.to_datetime(df_wth['date'])
        df_wth = df_wth.sort_values('_date_parsed')

        # Fill any missing days so DSSAT never hits a gap mid-season.
        # Gaps (e.g. from sidecar files corrupting the download index) cause
        # DSSAT to force-harvest on the first missing date.
        full_range = pd.date_range(df_wth['_date_parsed'].min(),
                                   df_wth['_date_parsed'].max(), freq='D')
        n_missing = len(full_range) - len(df_wth)
        if n_missing > 0:
            df_wth = (df_wth.set_index('_date_parsed')
                        .reindex(full_range)
                        .interpolate(method='linear')
                        .reset_index()
                        .rename(columns={'index': '_date_parsed'}))
            df_wth['date'] = df_wth['_date_parsed']
            # Linear interpolation of precipitation across long gaps is unrealistic
            # (rainfall is intermittent, not smooth). Warn loud enough that data
            # quality issues surface — but don't abort, since the alternative is a
            # silent DSSAT force-harvest on the first missing day.
            if n_missing > 5:
                logger.warning(
                    "[Pixel %s] Filled %d missing weather days via linear "
                    "interpolation — gaps > 5 days may yield unrealistic "
                    "precipitation; verify the source datacube.",
                    getattr(self, 'pixel_id', '?'), n_missing,
                )
            else:
                logger.debug(
                    "[Pixel %s] Filled %d missing weather day(s) via interpolation.",
                    getattr(self, 'pixel_id', '?'), n_missing,
                )

        df_wth['_year'] = pd.to_datetime(df_wth['date']).dt.year

        # Calculate Long-Term Average Temp (TAV) and Amplitude (AMP)
        tav = 20.0
        amp = 5.0
        if 'tmax' in df_wth.columns and 'tmin' in df_wth.columns:
            tav = ((df_wth['tmax'] + df_wth['tmin']) / 2).mean()

            # Simple AMP approximation
            df_wth = df_wth.copy()
            df_wth['month'] = pd.to_datetime(df_wth['date']).dt.month
            monthly_mean = ((df_wth['tmax'] + df_wth['tmin']) / 2).groupby(df_wth['month']).mean()
            amp = (monthly_mean.max() - monthly_mean.min()) / 2

        elev_str = f"{int(elevation):5d}" if elevation != -99.0 else "  -99"

        for yr, group in df_wth.groupby('_year'):
            yy = str(yr)[2:]
            file_path = self.working_dir / f"WTHE{yy}01.WTH"
            with open(file_path, "w") as f:
                f.write(f"$WEATHER DATA : WTHE\n\n")
                f.write("@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT  CCO2\n")
                f.write(f"  WTHE {lat:8.3f} {lon:8.3f} {elev_str} {tav:5.1f} {amp:5.1f}   -99   -99      \n")
                f.write("@  DATE  TMAX  TMIN  RAIN  SRAD\n")
                for _, row in group.iterrows():
                    yyyyjjj = pd.to_datetime(row['date']).strftime("%Y%j")
                    srad = row.get('solar_radiation', row.get('srad', -99.0))
                    tmax = row.get('tmax', -99.0)
                    tmin = row.get('tmin', -99.0)
                    rain = row.get('precipitation', row.get('rain', -99.0))
                    # Write row with strict Fortran spacing and matching DSSATTools order
                    f.write(f"{yyyyjjj:>7} {tmax:5.1f} {tmin:5.1f} {rain:5.1f} {srad:5.1f}\n")

    def _write_sol(self, df_sol: pd.DataFrame, lat: float, lon: float) -> None:
        """
        Write the .SOL file using standard variables.
        """
        file_path = self.working_dir / "TR.SOL"
        country = getattr(self.config.GENERAL_INFO, 'country', 'USA')

        # --- rescale every layer from raw SoilGrids units to DSSAT units ---
        layers: list[dict] = []
        for _, row in df_sol.iterrows():
            depth = row.get('depth', -99)
            if isinstance(depth, str):
                depth = int(depth.split('-')[1]) if '-' in depth else int(depth)
            else:
                depth = int(depth)

            # SoilGrids v2 scale factors → DSSAT units
            layers.append({
                'depth': depth,
                'slll': row.get('wv1500', -99.0) / 1000,  # 10^-3 cm3/cm3 → cm3/cm3
                'sdul': row.get('wv0033', -99.0) / 1000,
                'ssat': row.get('wv0010', -99.0) / 1000,
                'sbdm': row.get('bdod',   -99.0) / 100,   # cg/cm3 → g/cm3
                'sloc': row.get('soc',    -99.0) / 100,   # dg/kg → %
                'slcl': row.get('clay',   -99.0) / 10,    # g/kg → %
                'slsi': row.get('silt',   -99.0) / 10,    # g/kg → %
                'slcf': row.get('cfvo',   -99.0) / 10,    # cm3/dm3 × 10 → %
                'slhw': row.get('phh2o',  -99.0) / 10,    # pH × 10 → pH
                # SoilGrids nitrogen is cg/kg: /100 → g/kg, /10 again → %.
                'slni': row.get('nitrogen', -99.0) / 1000
                        if 'nitrogen' in row else -99.0,
            })

        if not layers:
            raise ValueError("Soil slice produced no layers; cannot write TR.SOL")

        depths = [lyr['depth'] for lyr in layers]
        # Sand is not written to the .SOL but is needed for every texture-derived
        # property below; SoilGrids gives clay and silt, so derive it.
        for lyr in layers:
            lyr['sand'] = max(0.0, 100.0 - lyr['slcl'] - lyr['slsi'])

        # --- profile-level properties, derived from the topsoil ---
        top = layers[0]
        profile_texture = dssat_soil.dssat_texture_code(top['sand'], top['slcl'])
        texture_name = dssat_soil.texture_class(top['sand'], top['slcl'])
        salb = dssat_soil.albedo(top['sloc'])
        slu1 = dssat_soil.stage1_evaporation_limit(top['slcl'], top['sand'])
        sldr = dssat_soil.drainage_rate(top['sand'], top['slcl'])
        slro = dssat_soil.runoff_curve_number(top['sand'], top['slcl'])
        total_depth = max(depths)
        srgf_values = dssat_soil.root_growth_factors(depths)

        logger.debug(
            "[Pixel %s] Soil: %s | SALB=%.3f SLU1=%.2f SLDR=%.2f SLRO=%.1f depth=%dcm",
            self.pixel_id, texture_name, salb, slu1, sldr, slro, total_depth,
        )

        with open(file_path, "w") as f:
            f.write("*SOILS: EXPS\n\n")
            # DSSAT parses this line by column position (IPSOIL), so the field
            # widths below are not cosmetic: source occupies columns 11-35,
            # the texture code 36-40, and the depth is right-aligned to 48.
            # The texture name goes after the depth, where DSSAT treats the
            # remainder of the line as free-text description.
            f.write(
                f"*TRAN0001  {'SoilGrids':<25}{profile_texture:<5}"
                f"{total_depth:>8} {texture_name.upper()}\n"
            )
            f.write("@SITE        COUNTRY          LAT     LONG SCS FAMILY\n")
            f.write(f" EXPS        {country[:10]:<10}  {lat:8.3f} {lon:8.3f}\n")
            f.write("@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE\n")
            f.write(
                f"   -99 {salb:5.2f} {slu1:5.1f} {sldr:5.2f} {slro:5.1f}"
                f"  1.00  1.00 -99.0 -99.0 -99.0\n"
            )
            f.write("@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC\n")

            # Keep (depth, slll, sdul) for the initial-conditions block of the X-file.
            self._soil_layers = []
            self._soil_layer_detail = layers
            for lyr, srgf in zip(layers, srgf_values):
                ssks = dssat_soil.saturated_conductivity(
                    lyr['ssat'], lyr['sdul'], lyr['slll']
                )
                self._soil_layers.append((lyr['depth'], lyr['slll'], lyr['sdul']))
                f.write(
                    f"{lyr['depth']:6d}   -99 {lyr['slll']:5.3f} {lyr['sdul']:5.3f} "
                    f"{lyr['ssat']:5.3f} {srgf:5.3f} {ssks:5.2f} {lyr['sbdm']:5.2f} "
                    f"{lyr['sloc']:5.2f} {lyr['slcl']:5.1f} {lyr['slsi']:5.1f} "
                    f"{lyr['slcf']:5.1f} {lyr['slni']:5.3f} {lyr['slhw']:5.2f} "
                    f"-99.0 -99.0 -99.0\n"
                )

    def _write_mzx(self, lat: float, lon: float) -> None:
        """
        Generate the experiment (.MZX) file matching the legacy KEAG8104 template format.
        """
        self.exp_filename = f"EXPS0001.{self.crop_code}X"
        file_path = self.working_dir / self.exp_filename

        planting_date_val = getattr(self.config.MANAGEMENT, 'planting_date', None)
        planting_date = str(planting_date_val) if planting_date_val is not None else '2000-01-01'
        pdate = datetime.strptime(planting_date, '%Y-%m-%d')
        pdate_str = pdate.strftime("%y%j")
        plname = pdate.strftime("%d-%b")

        # Simulation start = 1 month before planting
        sdate = (pdate - timedelta(days=30)).strftime("%y%j")
        # Harvest reference date = planting + crop-specific maturity days.
        # HARVS=R means DSSAT harvests at physiological maturity, not on this date,
        # but the field must still be parseable.
        maturity_days = self._get_maturity_days()
        hdate = (pdate + timedelta(days=maturity_days)).strftime("%y%j")
        # Planting window ± 3 days for automatic management
        pfrst = (pdate - timedelta(days=3)).strftime("%y%j")
        plast = (pdate + timedelta(days=3)).strftime("%y%j")

        wsta_id = "WTHE" + sdate[:2] + "01"

        fert_apps = getattr(self.config.MANAGEMENT, 'fertilizer_schedule', [])
        has_fert = bool(fert_apps)
        mf_flag = 1 if has_fert else 0
        nyers = self._nyers  # set by prepare_inputs from weather-data span

        crop_module = self._CROPS_MODULES.get(self.crop, "MZCER")

        with open(file_path, "w") as f:
            f.write(f"*EXP.DETAILS: EXPS0001{self.crop_code} SIMULATION\n\n")

            f.write("*GENERAL\n")
            f.write("@PEOPLE\nCGIAR\n@ADDRESS\nCGIAR\n@SITE\nCGIAR\n")
            f.write("@ PAREA  PRNO  PLEN  PLDR  PLSP  PLAY HAREA  HRNO  HLEN  HARM.........\n")
            f.write("    450     4    30   -99   -99   -99   450     4   -99   -99\n\n")

            f.write("*TREATMENTS                        -------------FACTOR LEVELS------------\n")
            f.write("@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM\n")
            f.write(f" 1 1 1 0 Default                    1  1  0  1  1  0  {mf_flag}  0  0  0  0  1  1\n\n")

            f.write("*CULTIVARS\n")
            f.write("@C CR INGENO CNAME\n")
            f.write(f" 1 {self.crop_code} {self.cultivar if self.cultivar else '999991'} NONE\n\n")

            f.write("*FIELDS\n")
            f.write("@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME\n")
            f.write(f" 1 UFWA0001 {wsta_id:<8}   -99   -99   -99   -99   -99   -99   SL   -99  TRAN0001   -99\n")
            f.write("@L ...........XCRD ...........YCRD .....ELEV .............AREA .SLEN .FLWR .SLAS FLHST FHDUR\n")
            f.write(f" 1 {lon:15.3f} {lat:15.3f}       -99               -99   -99   -99   -99 FH301     0\n\n")

            f.write("*INITIAL CONDITIONS\n")
            f.write("@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME\n")
            f.write(f" 1    {self.crop_code} {sdate}   150   -99  1.00  1.00   -99  1512  1.10  0.10     0   -99 -99\n")
            f.write("@C  ICBL  SH2O  SNH4  SNO3\n")
            # Derive SH2O from actual soil layers: slll + 80% of plant-available water
            if self._soil_layers:
                depths = [d for (d, _, _) in self._soil_layers]
                detail = getattr(self, '_soil_layer_detail', None) or []
                total_n = [lyr.get('slni', -99.0) for lyr in detail] or None
                if total_n and not all(n > 0 for n in total_n):
                    total_n = None  # nitrogen layer absent or unusable
                mineral_n = dssat_soil.initial_mineral_nitrogen(depths, total_n)

                for (depth, slll, sdul), (nh4, no3) in zip(self._soil_layers, mineral_n):
                    sh2o = slll + 0.8 * max(sdul - slll, 0.0)
                    sh2o = max(sh2o, slll + 0.05)  # ensure above wilting point
                    f.write(f" 1 {depth:5d} {sh2o:5.3f} {nh4:5.2f} {no3:5.2f}\n")
            else:
                f.write(" 1    20 0.250  0.60  3.00\n")
                f.write(" 1    60 0.250  0.60  3.00\n")
            f.write("\n")

            f.write("*PLANTING DETAILS\n")
            f.write("@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL                        PLNAME\n")
            f.write(f" 1 {pdate_str}   -99   5.0   5.0     S     R    75     0     5   -99   -99   -99   -99     0                        {plname}\n\n")

            f.write("*FERTILIZERS (INORGANIC)\n")
            f.write("@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME\n")
            if has_fert:
                for fert in fert_apps:
                    dap   = getattr(fert, 'days_after_planting', 0)
                    fdate = (pdate + timedelta(days=dap)).strftime("%y%j")
                    n_amt = getattr(fert, 'n_kg_ha', 0.0)
                    p_amt = getattr(fert, 'p_kg_ha', 0.0)
                    k_amt = getattr(fert, 'k_kg_ha', 0.0)
                    fmcd  = getattr(fert, 'fmcd', 'FE001')
                    facd  = getattr(fert, 'facd', 'AP001')
                    fdep  = getattr(fert, 'depth_cm', 5)
                    f.write(f" 1 {fdate} {fmcd:>5} {facd:>5} {fdep:5d} {n_amt:5.1f} {p_amt:5.1f} {k_amt:5.1f}   -99   -99   -99 Application\n")
            else:
                f.write(f" 1 {pdate_str} FE006   -99     4   -99   -99   -99   -99   -99   -99 Nofertilizer\n")
            f.write("\n")

            f.write("*HARVEST DETAILS\n")
            f.write("@H HDATE  HSTG  HCOM HSIZE   HPC  HBPC HNAME\n")
            f.write(f" 1 {hdate} GS006     H     A   100     0 {self.crop.upper()[:5]}\n\n")

            f.write("*SIMULATION CONTROLS\n")
            f.write("@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL\n")
            f.write(f" 1 GE             {nyers:2d}     1     S {sdate}  2150 DEFAULT                  {crop_module}\n")
            f.write("@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2\n")
            f.write(" 1 OP              Y     Y     Y     N     N     N     N     Y     M\n")
            f.write("@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL\n")
            f.write(" 1 ME              M     M     E     R     S     C     R     1     G     S     2\n")
            f.write("@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS\n")
            ferti_flag = "R" if has_fert else "N"
            f.write(f" 1 MA              R     N     {ferti_flag}     N     R\n")
            f.write("@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE CHOUT OPOUT FMOPT\n")
            f.write(" 1 OU              N     N     Y     1     N     N     N     N     N     N     0     N     N     A\n\n")

            f.write("@  AUTOMATIC MANAGEMENT\n")
            f.write("@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN\n")
            f.write(f" 1 PL          {pfrst} {plast}    40   100    30    40    10\n")
            f.write("@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF\n")
            f.write(" 1 IR              0     0     0 GS000 IR001     0     1\n")
            f.write("@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF\n")
            f.write(" 1 NI             30    50    25 FE001 GS000\n")
            f.write("@N RESIDUES    RIPCN RTIME RIDEP\n")
            f.write(" 1 RE            100     1    20\n")
            f.write("@N HARVEST     HFRST HLAST HPCNP HPCNR\n")
            f.write(" 1 HA             -99   -99   100     0\n")

    def _generate_batch_file(self) -> None:
        """Kept for reference; not used — direct C-mode execution avoids path-space issues."""
        pass

    # ------------------------------------------------------------------
    # Static folder bundled with this package (contains binaries,
    # Genotype/, StandardData/, Soil/, *.CDE files)
    # ------------------------------------------------------------------
    _STATIC_PATH: Path = Path(__file__).parent / "static"
    _DSSAT_VERSION: str = "048"
    _bootstrap_lock: threading.Lock = threading.Lock()  # one bootstrap at a time

    # Simulation module name by crop — written to DSSATPRO M-line and X-file SMODEL.
    # Legumes share the CRGRO engine regardless of species.
    _CROPS_MODULES: dict = {
        "maize":        "MZCER",
        "millet":       "MLCER",
        "rice":         "RICER",
        "sorghum":      "SGCER",
        "wheat":        "CSCER",
        "sweetcorn":    "MZCER",
        "soybean":      "CRGRO",
        "bean":         "CRGRO",
        "tomato":       "CRGRO",
        "cabbage":      "CRGRO",
        "alfalfa":      "CRGRO",
        "bermudagrass": "CRGRO",
        "sugarbeet":    "BSCER",
        "sugarcane":    "SCCAN",
        "potato":       "PTSUB",
        "cassava":      "CSYCA",
        "sunflower":    "SUOIL",
        "canola":       "CNOIL",
    }

    # Genotype file prefix by crop — names the .CUL/.ECO/.SPE files in static/Genotype/.
    # Differs from the simulation module for legumes (CRGRO engine, crop-specific files).
    _GENOTYPE_PREFIX: dict = {
        "bean":         "BNGRO",
        "soybean":      "SBGRO",
        "peanut":       "PNGRO",
        "tomato":       "TMGRO",
        "cabbage":      "CBGRO",
        "alfalfa":      "ALGRO",
        "bermudagrass": "BMGRO",
        "wheat":        "WHCER",
    }

    # Approximate days from planting to physiological maturity. Used in two places:
    #  (1) prepare_inputs — decides whether the last NYERS season would need
    #      weather beyond the datacube's last year (drop it if so).
    #  (2) _write_mzx — writes HDATE in the X-file. DSSAT parses it for validation
    #      even though HARVS=R harvests at maturity, not on this date.
    # Perennials use 365 to indicate a full-year window. Crops absent from this
    # table fall back to 120 days (maize default).
    _MATURITY_DAYS: dict = {
        "maize":        120,
        "millet":       100,
        "rice":         140,
        "sorghum":      120,
        "sweetcorn":    100,
        "wheat":        180,
        "bean":          90,
        "soybean":      130,
        "tomato":       110,
        "cabbage":      100,
        "alfalfa":      365,
        "bermudagrass": 365,
        "sugarbeet":    180,
        "sugarcane":    365,
        "potato":       110,
        "cassava":      300,
        "sunflower":    120,
        "canola":       150,
    }

    def _get_genotype_prefix(self) -> str:
        """Return the genotype file prefix for the current crop."""
        return self._GENOTYPE_PREFIX.get(
            self.crop,
            self._CROPS_MODULES.get(self.crop, "MZCER"),
        )

    def _copy_genotype_files(self) -> None:
        """Copy crop-specific .CUL / .ECO / .SPE files from static/Genotype/
        into self.working_dir so DSSAT can find them without relying on CRD path
        resolution (which varies across DSSAT binary versions).
        """
        prefix = self._get_genotype_prefix()
        genotype_dir = self._STATIC_PATH / "Genotype"
        for ext in ("CUL", "ECO", "SPE"):
            src = genotype_dir / f"{prefix}{self._DSSAT_VERSION}.{ext}"
            if src.exists():
                copy_retry(src, self.working_dir / src.name)
            else:
                logger.warning("Genotype file not found: %s", src)

    def _write_confile(self, dssat_home: Path) -> None:
        """Write DSSATPRO.V48 (Windows) or DSSATPRO.L48 (Linux/Mac) to
        self.working_dir.

        Mirrors DSSATTools create_dssat_config_path_file:
            WED    <weather/workdir>
            M{crop_code}    <workdir> dscsm048 {module}{version}
            CRD    <Genotype/>
            PSD    <Pest/>
            SLD    <Soil/>
            STD    <StandardData/>
        """
        is_windows = platform.system().lower() == "windows"
        confile = "DSSATPRO.V48" if is_windows else "DSSATPRO.L48"

        crop_module = self._CROPS_MODULES.get(self.crop, "MZCER")

        def _abs(path: Path) -> str:
            """Absolute path string, no trailing separator. Do not resolve symlinks to avoid Fortran 80-char limit."""
            return str(path)

        workdir_abs = _abs(self.working_dir)

        # DSSAT DSSATPRO.V48 uses whitespace-delimited parsing for every line.
        # A space anywhere in the path corrupts the M-line and produces rc=99
        # ("Crop code incompatible with model specified"). Fail fast with a clear message.
        if ' ' in workdir_abs:
            raise RuntimeError(
                f"DSSAT working directory path contains spaces, which DSSAT cannot handle "
                f"(DSSATPRO whitespace-delimiter bug):\n  {workdir_abs}\n"
                f"Set GENERAL_INFO.working_path to a path without spaces, "
                f"e.g. 'D:/tmp/dssat_runs' or '/tmp/dssat_runs'."
            )

        # WED must point to the weather directory (with a trailing separator).
        wth_stem = workdir_abs + os.sep

        with open(self.working_dir / confile, "w") as f:
            f.write(f"WED    {wth_stem}\n")
            # MMZ path = run directory WITHOUT trailing separator; DSSAT appends sep+filename
            f.write(f"M{self.crop_code}    {workdir_abs} dscsm048 "
                    f"{crop_module}{self._DSSAT_VERSION}\n")
            f.write(f"CRD    {_abs(dssat_home / 'Genotype')}\n")
            f.write(f"PSD    {_abs(dssat_home / 'Pest')}\n")
            f.write(f"SLD    {_abs(dssat_home / 'Soil')}\n")
            f.write(f"STD    {_abs(dssat_home / 'StandardData')}\n")

    def _bootstrap_dssat_home(self, tmp_base: str) -> tuple:
        """Copy the bundled static/ folder to <tmp_base>/DSSAT048/ and return
        (dssat_home: Path, bin_path: Path).  Skips files that already exist so
        repeated calls in the same session are cheap.
        """
        dssat_home = Path(tmp_base) / f"DSSAT{self._DSSAT_VERSION}"

        with self._bootstrap_lock:
            dssat_home.mkdir(parents=True, exist_ok=True)
            for item in self._STATIC_PATH.iterdir():
                dest = dssat_home / item.name
                if item.is_dir():
                    if not dest.exists():
                        shutil.copytree(str(item), str(dest))
                    continue

                # Only write a file that is missing or stale. Unconditionally
                # unlinking and recopying — as this previously did — reopens a
                # race on every call: a sibling thread staging DATA.CDE /
                # MODEL.ERR / the binary out of this shared tree can land in the
                # window between the unlink and the copy and lose its pixel to a
                # PermissionError. Once the tree is populated this loop now
                # touches nothing, so the window closes entirely.
                if dest.exists() and dest.stat().st_size == item.stat().st_size:
                    continue
                try:
                    copy_retry(item, dest)
                except OSError:
                    # A sibling thread is writing the same file; its copy is as
                    # good as ours and the size check above will accept it.
                    logger.debug(
                        "Bootstrap: %s is being written by another thread, skipping",
                        item.name,
                    )

        is_windows = platform.system().lower() == "windows"
        exe_name = "dscsm048.exe" if is_windows else "dscsm048"
        bin_path = dssat_home / "bin" / exe_name

        if not bin_path.exists():
            raise FileNotFoundError(
                f"DSSAT binary not found in bundled static folder: {bin_path}"
            )

        # Ensure the binary is executable on Linux/Mac
        if not is_windows:
            bin_path.chmod(bin_path.stat().st_mode | 0o111)

        return dssat_home, bin_path

    def run_simulation(self) -> None:
        """Execute the DSSAT binary via subprocess.

        Three execution paths (in priority order):
        1. Config-provided ``bin_path`` exists on disk → use it directly with
           ``dssat_path`` as DSSAT_HOME.
        2. DSSATTools available → import its already-bootstrapped paths
           (DSSAT_HOME, BIN_PATH, CRD_PATH, SLD_PATH, STD_PATH).  DSSATTools
           handles the static-file copy at import time so nothing extra is done
           here.
        3. Fallback → copy ag_cube_cm's bundled ``static/`` folder to
           ``<dssat_path or tempdir>/DSSAT048/`` (used when DSSATTools is not
           installed).
        """
        cfg_bin   = getattr(self.config.GENERAL_INFO, 'bin_path',   None)
        cfg_home  = getattr(self.config.GENERAL_INFO, 'dssat_path', None)

        if cfg_bin and os.path.exists(cfg_bin):
            # Path 1 — user-supplied binary
            bin_path   = cfg_bin
            dssat_home = Path(cfg_home or os.path.dirname(cfg_bin))
            logger.debug("[Pixel %s] Using config-supplied DSSAT binary: %s", self.pixel_id, bin_path)
        else:
            # Always use the bundled static/ via _bootstrap_dssat_home.
            #
            # We deliberately do NOT try `DSSATTools.run` here even when it's
            # installed. Its module init does `os.remove(<temp>/DSSAT048/<f>)`
            # followed by `os.symlink(...)`. On Windows without Developer Mode
            # or admin, the symlink raises OSError 1314 — but the os.remove has
            # already deleted the file. Under ncores>1 this races with sibling
            # threads that are in the middle of running DSSAT subprocesses
            # against the same shared <temp>/DSSAT048/ tree, and a vanished
            # DATA.CDE / DETAIL.CDE / MODEL.ERR / OUTPUT.CDE / SIMULATION.CDE
            # at the wrong moment makes DSSAT exit rc=99 with the misleading
            # "Crop code incompatible / Model: ZCER 048" error.
            #
            # _bootstrap_dssat_home does the same job (copy static into
            # <temp>/DSSAT048/), guarded by self._bootstrap_lock so concurrent
            # threads can't corrupt the tree mid-run.
            tmp_base = cfg_home or tempfile.gettempdir()
            dssat_home, bin_path_obj = self._bootstrap_dssat_home(tmp_base)
            bin_path = str(bin_path_obj)
            logger.debug(
                "[Pixel %s] Using bundled DSSAT bootstrap at %s", self.pixel_id, dssat_home,
            )

        # Copy crop .CUL/.ECO/.SPE into run dir (DSSAT needs them locally).
        self._copy_genotype_files()
        # Write DSSATPRO.V48/.L48 to the pixel working directory.
        self._write_confile(dssat_home)

        # Copy CDE files and MODEL.ERR from dssat_home to working_dir (so DSSAT can find them with DSSAT_HOME pointing to working_dir)
        for fname in ["DATA.CDE", "DETAIL.CDE", "MODEL.ERR", "OUTPUT.CDE", "SIMULATION.CDE"]:
            src_file = dssat_home / fname
            if src_file.exists():
                # Reads the shared bootstrap tree, so retry past transient locks.
                copy_retry(src_file, self.working_dir / fname)

        # Use C mode (direct execution): dscsm048 C EXPS0001.MZX 1
        # This avoids the DSSBatch.v48 71-char padded-path bug in DSSAT 4.8.
        cmd = [bin_path, "C", self.exp_filename, "1"]

        try:
            # Inherit the full parent environment so PATH / LD_LIBRARY_PATH / HOME
            # are available on Linux; only override DSSAT_HOME.
            run_env = os.environ.copy()
            run_env["DSSAT_HOME"] = str(self.working_dir)

            result = subprocess.run(
                cmd,
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                env=run_env,
            )
            logger.debug("[Pixel %s] DSSAT stdout: %s", self.pixel_id, result.stdout[-500:])
            summary = self.working_dir / "Summary.OUT"
            if not summary.exists():
                # Crop failure (rc=3) or other non-fatal DSSAT result —
                # collect_outputs will return {} and the caller treats it as NaN yield.
                last_lines = result.stdout.strip().splitlines()
                diag = " | ".join(last_lines[-3:]) if last_lines else "no output"
                logger.warning(
                    "[Pixel %s] DSSAT finished without Summary.OUT (rc=%d): %s",
                    self.pixel_id, result.returncode, diag,
                )
            else:
                logger.debug("[Pixel %s] DSSAT executed successfully.", self.pixel_id)

        except subprocess.CalledProcessError as e:
            logger.error("[Pixel %s] DSSAT failed (rc=%d)", self.pixel_id, e.returncode)
            logger.error("[Pixel %s] STDOUT: %s", self.pixel_id, e.stdout)
            logger.error("[Pixel %s] STDERR: %s", self.pixel_id, e.stderr)
            raise RuntimeError(f"DSSAT execution failed for pixel {self.pixel_id}")

    def collect_outputs(self) -> Dict[str, Any]:
        """
        Parse Summary.OUT and extract relevant yield data into a dictionary.
        """
        summary_file = self.working_dir / "Summary.OUT"
        if not summary_file.exists():
            return {}
        result_dict = {}
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        # Locate the header line and all data rows (one per simulated year)
        header_line = None
        data_rows = []
        for i, line in enumerate(lines):
            if "@RUNNO" in line or ("@" in line and "HWAM" in line):
                header_line = line
                for j in range(i + 1, len(lines)):
                    stripped = lines[j].strip()
                    if stripped and not stripped.startswith('!') and not stripped.startswith('@'):
                        data_rows.append(lines[j])
                break

        if not header_line or not data_rows:
            return result_dict

        headers = header_line.replace('@', '').split()

        # Parse every row (one per NYERS season)
        all_years: list[dict] = []
        for row in data_rows:
            row_dict: dict = {}
            values = row.split()
            for h, v in zip(headers, values):
                try:
                    row_dict[h] = float(v)
                except ValueError:
                    row_dict[h] = v
            all_years.append(row_dict)

        # result_dict uses first-year values for metadata fields (PDAT, MDAT, …)
        result_dict.update(all_years[0])

        # Count successful vs failed years regardless of NYERS — downstream tools
        # often need to know the sample size behind the mean HWAM, and a failure
        # rate per pixel is useful for spatial QA.
        hwam_values = [
            r["HWAM"] for r in all_years
            if isinstance(r.get("HWAM"), float) and r["HWAM"] > 0
        ]
        result_dict["years_succeeded"] = len(hwam_values)
        result_dict["years_failed"] = len(all_years) - len(hwam_values)

        if len(all_years) == 1:
            return result_dict

        # Multi-year: add per-year HWAM list and overwrite scalar HWAM with mean
        result_dict["HWAM_yearly"] = [r.get("HWAM", float("nan")) for r in all_years]
        result_dict["HWAM"] = (
            sum(hwam_values) / len(hwam_values) if hwam_values else float("nan")
        )

        return result_dict
