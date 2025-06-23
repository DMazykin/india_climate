"""
ag_temp_aggregation.py

Step‑wise implementation to derive **area‑weighted daily mean 2‑m temperature** for Indian ADM‑3 districts from ERA5‑Land.

We will build this script in *four* incremental steps:

1. **Load input datasets**  – ADM3 polygons (GADM 4.1) and daily ERA5‑Land NetCDF; verify CRS & basic alignment.
2. **Quick‑and‑dirty aggregation** for a single day using `rasterstats.zonal_stats`, to validate geometry overlap and inspect run‑time.
3. **Reusable weight matrix**  – pre‑compute the fractional overlap of each ERA5‑Land cell with every ADM3 polygon and store it as a sparse CSR matrix on disk.
4. **Loop over time (or Dask chunks)**  – apply the weight matrix to all daily 2‑m temperature fields, producing a tidy Parquet/NetCDF table indexed by `date` × `GID_3`.

Dependencies
------------
```bash
conda install -c conda-forge geopandas rioxarray rasterstats xarray dask-scikit-image scipy tqdm
```

You can run this module as a script (`python ag_temp_aggregation.py`) or import the helper functions in a notebook.
"""

import pathlib
from typing import List, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401 – side‑effect CRS helpers
from tqdm import tqdm


class ClimateAggregator:
    def __init__(
        self,
        data_dir: Union[str, pathlib.Path] = "input",
        output_dir: Union[str, pathlib.Path] = "output",
        adm3_file: str = "gadm41_IND.gpkg",
        era5_file: str = "era5_daily_2m_temperature_2024_2025.nc",
    ):
        self.DATA_DIR = pathlib.Path(data_dir)
        self.OUTPUT_DIR = pathlib.Path(output_dir)
        self.ADM3_FILE = self.DATA_DIR / adm3_file
        self.ERA5_FILE = self.DATA_DIR / era5_file
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def get_era5_daily_data(
        self,
        variable: str,
        years: List[int],
        output_file: str,
        area: List[float] = [38, 68, 6, 97],
    ):
        """Download daily ERA5 data for a user-defined variable and years using cdsapi."""
        import cdsapi

        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [variable],
                "year": [str(year) for year in years],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": area,
                "format": "netcdf",
            },
            str(self.DATA_DIR / output_file),
        )
        print(f"✅ Downloaded daily ERA5 data for {variable} to {output_file}")

    def build_and_export_weight_matrix(
        self,
        adm_level: int = 3,
        all_touched: bool = True,
        coslat_weight: bool = True,
        mapping_out: str = None,
    ) -> pd.DataFrame:
        """
        Build and export the mapping table (GID_x, i, j, weight) to output_dir as parquet for any admin level.
        """
        layer = f"ADM_ADM_{adm_level}"
        gid_col = f"GID_{adm_level}"
        # Load ADM polygons
        adm = gpd.read_file(self.ADM3_FILE, layer=layer)
        if adm.crs is None:
            adm.set_crs(epsg=4326, inplace=True)
        else:
            adm = adm.to_crs(epsg=4326)
        if gid_col not in adm.columns:
            raise KeyError(f"{gid_col} not found in ADM file.")
        # Load ERA5 grid (just for lat/lon)
        da_template = xr.open_dataset(self.ERA5_FILE)
        lat = da_template["latitude"].values
        lon = da_template["longitude"].values
        n_lon = lon.size
        cos_lat = np.cos(np.deg2rad(lat)) if coslat_weight else np.ones_like(lat)
        i_idx = []
        j_idx = []
        gids = []
        weights = []
        missed_polygons = []
        for poly_idx, (gid, geom) in tqdm(
            list(enumerate(adm[[gid_col, "geometry"]].itertuples(index=False))),
            desc=f"Rasterising polygons (ADM{adm_level})",
        ):
            minx, miny, maxx, maxy = geom.bounds
            lat_sel = np.where((lat >= miny - 0.1) & (lat <= maxy + 0.1))[0]
            lon_sel = np.where((lon >= minx - 0.1) & (lon <= maxx + 0.1))[0]
            if lat_sel.size == 0 or lon_sel.size == 0:
                missed_polygons.append(poly_idx)
                continue
            touched = []
            prep_geom = geom.buffer(0)
            for i in lat_sel:
                for j in lon_sel:
                    pt = (lon[j], lat[i])
                    if prep_geom.contains(gpd.points_from_xy([pt[0]], [pt[1]])[0]):
                        touched.append((i, j))
            if not touched:
                missed_polygons.append(poly_idx)
                continue
            local_weights = []
            for i, j in touched:
                w = cos_lat[i]
                local_weights.append(w)
                i_idx.append(i)
                j_idx.append(j)
                gids.append(gid)
            s = float(sum(local_weights))
            weights.extend([w / s for w in local_weights])
        # Fallback for missed polygons (assign to nearest grid cell by centroid)
        if missed_polygons:
            from sklearn.neighbors import BallTree

            lats2d, lons2d = np.meshgrid(lat, lon, indexing="ij")
            coords = np.column_stack([lats2d.ravel(), lons2d.ravel()])
            tree = BallTree(np.deg2rad(coords), metric="haversine")
            for poly_idx in missed_polygons:
                gid = adm.iloc[poly_idx][gid_col]
                c = adm.iloc[poly_idx].geometry.centroid
                dist, idx = tree.query(np.deg2rad([[c.y, c.x]]), k=1)
                i, j = divmod(idx[0][0], n_lon)
                i_idx.append(i)
                j_idx.append(j)
                gids.append(gid)
                weights.append(1.0)
        # Ensure integer indices for lat/lon
        i_idx_arr = np.asarray(i_idx, dtype=int)
        j_idx_arr = np.asarray(j_idx, dtype=int)
        lat_vals = lat[i_idx_arr]
        lon_vals = lon[j_idx_arr]
        df = pd.DataFrame(
            {
                gid_col: gids,
                "i": i_idx_arr,
                "j": j_idx_arr,
                "lat": lat_vals,
                "lon": lon_vals,
                "weight": weights,
            }
        )
        if mapping_out is None:
            mapping_out = f"grid_to_adm{adm_level}_mapping.parquet"
        mapping_out = pathlib.Path(mapping_out)
        if not mapping_out.is_absolute():
            out_path = self.OUTPUT_DIR / mapping_out
        else:
            out_path = mapping_out
        df.to_parquet(out_path, index=False)
        print(f"✅ Saved mapping table → {out_path}")
        return df

    def weighted_mean_indicator_timeseries(
        self,
        var: str,
        adm_level: int = 3,
        era5_file: str = None,
        mapping_file: str = None,
        output_file: str = None,
    ) -> pd.DataFrame:
        """
        Calculate area-weighted mean for any indicator (variable) for each ADM region for all dates in the ERA5 file using the mapping file.
        Supports ADM1, ADM2, ADM3 (default ADM3).
        Returns DataFrame with columns: GID, date, value, series_id. Also saves to parquet.
        """
        if era5_file is None:
            era5_file = self.ERA5_FILE
        if mapping_file is None:
            mapping_file = f"grid_to_adm{adm_level}_mapping.parquet"
        mapping_file = pathlib.Path(mapping_file)
        if not mapping_file.is_absolute():
            input_path = self.DATA_DIR / mapping_file
            output_path = self.OUTPUT_DIR / mapping_file
            if input_path.exists():
                mapping_file = input_path
            elif output_path.exists():
                mapping_file = output_path
            else:
                mapping_file = input_path
        if output_file is None:
            output_file = f"adm{adm_level}_{var}_timeseries.parquet"
        output_file = pathlib.Path(output_file)
        if not output_file.is_absolute():
            out_path = self.OUTPUT_DIR / output_file
        else:
            out_path = output_file
        mapping = pd.read_parquet(mapping_file)
        gid_col = f"GID_{adm_level}"
        if gid_col not in mapping.columns:
            raise KeyError(f"{gid_col} not found in mapping file.")
        ds = xr.open_dataset(era5_file)
        results = []
        times = pd.to_datetime(ds["valid_time"].values)
        unique_dates = np.unique(times.date)
        for date in unique_dates:
            mask = times.date == date
            arr = ds[var].isel(valid_time=mask).mean(dim="valid_time").values
            if var in ["t2m", "2m_temperature", "temperature"]:
                arr = arr - 273.15
            arr_flat = arr.reshape(-1)
            valid_mask = (
                (mapping["i"] >= 0)
                & (mapping["i"] < arr.shape[0])
                & (mapping["j"] >= 0)
                & (mapping["j"] < arr.shape[1])
            )
            df = mapping[valid_mask].copy()
            df["value"] = arr_flat[df["i"] * arr.shape[1] + df["j"]]
            df["weighted_value"] = df["value"] * df["weight"]
            agg = df.groupby(gid_col).agg(value=("weighted_value", "sum")).reset_index()
            agg["series_id"] = agg[gid_col] + f"_{var}"
            agg["date"] = pd.to_datetime(date)
            agg = agg.rename(columns={gid_col: "GID"})
            results.append(agg[["series_id", "date", "value"]])
        out_df = pd.concat(results, ignore_index=True)
        out_df.to_parquet(out_path, index=False)
        print(f"✅ Saved ADM{adm_level} {var} timeseries to {out_path}")
        return out_df

    def timeseries_metadata(
        self,
        indicator: str,
        units: str,
        adm_level: int = 3,
        mapping_file: str = None,
        adm_file: str = None,
        title_template: str = "{name} ({gid}) - {indicator_title}",
        indicator_title: str = None,
        output_file: str = None,
    ) -> pd.DataFrame:
        """
        Create a metadata DataFrame for ADM time series and save to parquet.
        Columns: GID, NAME, adm_level, indicator, units, series_id, title
        """
        if mapping_file is None:
            mapping_file = f"grid_to_adm{adm_level}_mapping.parquet"
        mapping_file = pathlib.Path(mapping_file)
        if not mapping_file.is_absolute():
            input_path = self.DATA_DIR / mapping_file
            output_path = self.OUTPUT_DIR / mapping_file
            if input_path.exists():
                mapping_file = input_path
            elif output_path.exists():
                mapping_file = output_path
            else:
                mapping_file = input_path
        mapping = pd.read_parquet(mapping_file)
        gid_col = f"GID_{adm_level}"
        name_col = f"NAME_{adm_level}"
        unique_gids = mapping[gid_col].unique()
        if adm_file is None:
            adm_file = self.ADM3_FILE
        layer = f"ADM_ADM_{adm_level}"
        adm = gpd.read_file(adm_file, layer=layer)
        if name_col not in adm.columns:
            adm[name_col] = adm[gid_col]
        meta = adm[adm[gid_col].isin(unique_gids)][
            [gid_col, name_col]
        ].drop_duplicates()
        meta = meta.rename(columns={gid_col: "GID", name_col: "NAME"})
        meta = meta.sort_values("GID").reset_index(drop=True)
        meta["adm_level"] = adm_level
        meta["indicator"] = indicator
        meta["units"] = units
        meta["series_id"] = meta["GID"] + f"_{indicator}"
        if indicator_title is None:
            indicator_title = indicator.replace("_", " ").title()
        meta["title"] = meta.apply(
            lambda row: title_template.format(
                name=row["NAME"],
                gid=row["GID"],
                indicator=indicator,
                indicator_title=indicator_title,
            ),
            axis=1,
        )
        if output_file is None:
            output_file = f"adm{adm_level}_metadata_{indicator}.parquet"
        output_file = pathlib.Path(output_file)
        if not output_file.is_absolute():
            out_path = self.OUTPUT_DIR / output_file
        else:
            out_path = output_file
        meta.to_parquet(out_path, index=False)
        print(f"✅ Saved ADM{adm_level} metadata to {out_path}")
        return meta
