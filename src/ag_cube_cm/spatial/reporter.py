"""
ag_cube_cm.spatial.reporter
===========================

Universal Spatial Reporter for converting simulated outputs back into
spatial structures (tabular or raster).
"""

import logging
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

class SpatialReporter:
    """
    Handles the aggregation and exportation of simulation results into 
    tabular formats (Parquet, CSV) and raster formats (NetCDF, GeoTIFF).
    """

    def __init__(self, crs: str = "EPSG:4326"):
        self.crs = crs

    def build_yield_cube(
        self,
        records: pd.DataFrame,
        planting_dates: List[Any],
        years: List[int],
        value_name: str = "HWAM",
        value_units: str = "kg/ha",
        value_long_name: str = "Mean grain yield at maturity",
    ) -> xr.Dataset:
        """Grid flat simulation records into a 4-D CF-compliant datacube.

        Parameters
        ----------
        records : pd.DataFrame
            One row per job, with columns ``window_idx``, ``year``, ``y``,
            ``x``, ``value`` and ``flag``.
        planting_dates : list
            Base planting date of each window; written as a coordinate.
        years : list[int]
            Seasons simulated, in output order.
        value_name, value_units, value_long_name : str
            Naming for the yield variable.

        Returns
        -------
        xr.Dataset
            ``planting_window x year x y x x`` with the yield variable and a
            ``flag`` array (0 = ok, 1 = model error, 2 = no data).
        """
        if records.empty:
            logger.warning("No records provided — returning an empty Dataset.")
            return xr.Dataset()

        n_windows = len(planting_dates)
        y_vals = sorted(records["y"].unique())
        x_vals = sorted(records["x"].unique())
        yi = {v: i for i, v in enumerate(y_vals)}
        xi = {v: i for i, v in enumerate(x_vals)}
        yri = {yr: i for i, yr in enumerate(years)}

        shape = (n_windows, len(years), len(y_vals), len(x_vals))
        value_grid = np.full(shape, np.nan, dtype=np.float32)
        flag_grid = np.full(shape, 2, dtype=np.int8)

        for row in records.itertuples(index=False):
            w = int(row.window_idx)
            yr = yri.get(int(row.year))
            if yr is None:
                continue
            value_grid[w, yr, yi[row.y], xi[row.x]] = row.value
            flag_grid[w, yr, yi[row.y], xi[row.x]] = int(row.flag)

        dims = ["planting_window", "year", "y", "x"]
        ds = xr.Dataset(
            {
                value_name: (dims, value_grid,
                             {"long_name": value_long_name, "units": value_units}),
                "flag": (dims, flag_grid,
                         {"long_name": "0=ok 1=model_error 2=no_data"}),
            },
            coords={
                "planting_window": np.arange(n_windows),
                "year": list(years),
                "planting_date": (["planting_window"],
                                  [str(p) for p in planting_dates]),
                "y": y_vals,
                "x": x_vals,
            },
        )

        # A stale grid_mapping pointer makes rioxarray raise on write_crs.
        for name in list(ds.data_vars) + list(ds.coords):
            ds.variables[name].encoding.pop("grid_mapping", None)
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs(self.crs)
        ds["x"].attrs.update({
            "standard_name": "longitude", "long_name": "longitude",
            "units": "degrees_east", "axis": "X",
        })
        ds["y"].attrs.update({
            "standard_name": "latitude", "long_name": "latitude",
            "units": "degrees_north", "axis": "Y",
        })
        ds.attrs["Conventions"] = "CF-1.8"
        return ds

    def export_to_tabular(self, dataset: xr.Dataset, filepath: Union[str, Path], format: str = "parquet") -> None:
        """
        Exports the Xarray Dataset into a tabular format, dropping NaNs to save space.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to export.
        filepath : Union[str, Path]
            Destination path.
        format : str
            'parquet' or 'csv'.
        """
        logger.info(f"Exporting to {format.upper()} at {filepath}")
        df = dataset.to_dataframe().reset_index()
        
        # Drop spatial NaNs (pixels outside ROI or missing data)
        data_cols = [c for c in df.columns if c not in ['x', 'y', 'PDAT']]
        if data_cols:
            df = df.dropna(subset=data_cols, how='all')

        if format.lower() == "parquet":
            df.to_parquet(filepath, index=False)
        elif format.lower() == "csv":
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported tabular format: {format}")

    def export_to_raster(self, dataset: xr.Dataset, filepath: Union[str, Path], format: str = "netcdf") -> None:
        """
        Exports the Xarray Dataset directly to a spatial raster file.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to export.
        filepath : Union[str, Path]
            Destination path.
        format : str
            'netcdf' or 'tif'.
        """
        logger.info(f"Exporting to {format.upper()} at {filepath}")

        if not dataset.rio.crs:
            dataset.rio.write_crs(self.crs, inplace=True)

        if 'x' in dataset.coords and 'y' in dataset.coords:
            dataset = dataset.rio.set_spatial_dims(x_dim="x", y_dim="y")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # netCDF4 refuses to overwrite a file that is still open elsewhere;
        # removing first keeps repeated runs idempotent.
        if path.exists():
            path.unlink()

        if format.lower() == "netcdf":
            dataset.to_netcdf(path)
        elif format.lower() in ["tif", "tiff"]:
            try:
                dataset.rio.to_raster(filepath)
            except Exception as e:
                logger.error(f"Failed to export full dataset to TIF. Reason: {e}. Attempting first variable only.")
                first_var = list(dataset.data_vars)[0]
                dataset[first_var].rio.to_raster(filepath)
        else:
            raise ValueError(f"Unsupported raster format: {format}")
