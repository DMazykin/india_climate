"""
Download ERA5 monthly climate data for India using CDS API and prepare for database ingestion.
"""

import cdsapi
import xarray as xr
import pandas as pd
import duckdb
import geopandas as gpd
from shapely.geometry import Point

"""
# --- Download ERA5 monthly climate data ---
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            '2m_temperature', 'total_precipitation', 'volumetric_soil_water_layer_1',
            '10m_u_component_of_wind', '10m_v_component_of_wind',
            'surface_solar_radiation_downwards'
        ],
        'year': [str(year) for year in range(2009, 2024)],
        'month': ['%02d' % m for m in range(1, 13)],
        'time': '00:00',
        'area': [38, 68, 6, 97],
        'format': 'netcdf'
    },
    'era5_monthly_data_09_24.nc'
)
"""

# --- Convert NetCDF to DataFrame ---
ds = xr.open_dataset("era5_monthly_data_09_24.nc")
climate_df = ds.to_dataframe().reset_index()
climate_df["date"] = pd.to_datetime(climate_df["date"].astype(str), format="%Y%m%d")

# --- Setup DuckDB in-memory with spatial extension ---
duckdb_con = duckdb.connect(database=":memory:")
duckdb_con.execute("INSTALL spatial; LOAD spatial;")
duckdb_con.register("climate_df", climate_df)

# Create unique grid points view
duckdb_con.execute("DROP VIEW IF EXISTS unique_era5_points;")
duckdb_con.execute("""
    CREATE VIEW unique_era5_points AS
    SELECT DISTINCT latitude, longitude
    FROM climate_df;
""")

# --- Load ADM3 boundaries and register directly in DuckDB ---
gdf = gpd.read_file("input/gadm41_IND.gpkg", layer="ADM_ADM_3")
gdf = gdf.to_crs("EPSG:4326") if gdf.crs is None or gdf.crs != "EPSG:4326" else gdf

# Convert to WKB for DuckDB and register as pandas DataFrame
gdf_wkb = gdf.copy()
gdf_wkb["geometry"] = gdf_wkb.geometry.to_wkb()
duckdb_con.register("ADM_ADM_3_raw", gdf_wkb)

# Create proper geometry table in DuckDB - replace WKB geometry with GEOMETRY type
duckdb_con.execute("""
    CREATE TABLE ADM_ADM_3 AS
    SELECT 
        GID_0, NAME_0, GID_1, NAME_1, NL_NAME_1, GID_2, NAME_2, NL_NAME_2, 
        GID_3, NAME_3, VARNAME_3, NL_NAME_3, TYPE_3, ENGTYPE_3, CC_3,
        ST_GeomFromWKB(geometry) AS geometry
    FROM ADM_ADM_3_raw
""")

# --- Spatial join: Grid point → ADM3 region ---
duckdb_con.execute("""
    CREATE TABLE point_to_region_mapping AS
    SELECT p.latitude, p.longitude,
           t.GID_3, t.NAME_3, t.GID_2, t.NAME_2, t.GID_1, t.NAME_1
    FROM unique_era5_points AS p
    LEFT JOIN ADM_ADM_3 AS t
    ON ST_Contains(t.geometry, ST_Point(p.longitude, p.latitude));
""")

# --- Join full climate data with ADM3 ---
duckdb_con.execute("""
    CREATE VIEW climate_data_by_region AS
    SELECT c.*,
           m.GID_3, m.NAME_3, m.GID_2, m.NAME_2, m.GID_1, m.NAME_1
    FROM climate_df AS c
    LEFT JOIN point_to_region_mapping AS m
    ON c.latitude = m.latitude AND c.longitude = m.longitude
    WHERE m.GID_3 IS NOT NULL;
""")

# --- Aggregate monthly by ADM3 and date ---
agg_df = duckdb_con.execute("""
    SELECT
        m.GID_3 AS gid, m.NAME_3 AS name, c.date,
        AVG(c.t2m) - 273.15 AS avg_temperature,
        SUM(c.tp) AS total_precipitation,
        AVG(c.swvl1) AS avg_soil_moisture,
        AVG(c.u10) AS avg_wind_u,
        AVG(c.v10) AS avg_wind_v,
        AVG(c.ssrd) AS avg_solar_radiation
    FROM climate_data_by_region AS c
    LEFT JOIN point_to_region_mapping AS m
    ON c.latitude = m.latitude AND c.longitude = m.longitude
    WHERE m.GID_3 IS NOT NULL
    GROUP BY m.GID_3, m.NAME_3, c.date
    ORDER BY m.GID_3, c.date
""").fetchdf()

agg_df["date"] = pd.to_datetime(agg_df["date"]).dt.date
agg_df["primary_key"] = agg_df["gid"] + "_" + agg_df["date"].astype(str)
agg_df["title"] = agg_df["name"] + " - " + agg_df["date"].astype(str)
agg_df = agg_df[
    ["primary_key", "title"]
    + [col for col in agg_df.columns if col not in ["primary_key", "title"]]
]

agg_df.to_parquet("output/climate_agg_by_adm3.parquet")
print("Saved aggregated climate data to output/climate_agg_by_adm3.parquet")

# --- Weekly Aggregation ---
climate_df["week"] = climate_df["date"].dt.to_period("W").apply(lambda r: r.start_time)
climate_df["t2m"] -= 273.15
climate_df["tp"] *= 1000  # meters to mm

adm3_gdf = gpd.read_parquet("output/geoparquet/ADM_ADM_3.parquet")
climate_df["geometry"] = climate_df.apply(
    lambda r: Point(r["longitude"], r["latitude"]), axis=1
)
climate_gdf = gpd.GeoDataFrame(climate_df, geometry="geometry", crs=adm3_gdf.crs)
joined = gpd.sjoin(climate_gdf, adm3_gdf, how="left", predicate="within")

indicators = {
    "t2m": ("temperature", "Celsius"),
    "tp": ("precipitation", "mm"),
    "swvl1": ("soil_moisture", "m3/m3"),
    "u10": ("wind_u", "m/s"),
    "v10": ("wind_v", "m/s"),
    "ssrd": ("solar_radiation", "J/m2"),
}

# Aggregate by week
agg_rows = []
for var, (indicator, units) in indicators.items():
    grouped = (
        joined.groupby(["GID_3", "NAME_3", "week"])[var]
        .mean()
        .reset_index()
        .rename(columns={var: "value"})
    )
    grouped["indicator"] = indicator
    grouped["units"] = units
    grouped["series_id"] = grouped["GID_3"] + "_" + indicator
    grouped["title"] = (
        grouped["GID_3"]
        + " "
        + grouped["NAME_3"]
        + " "
        + indicator.replace("_", " ").title()
    )
    agg_rows.append(grouped)

weekly_df = pd.concat(agg_rows, ignore_index=True)

# Metadata table
meta_df = weekly_df[
    ["series_id", "GID_3", "NAME_3", "indicator", "units", "title"]
].drop_duplicates()
meta_df = meta_df.rename(columns={"GID_3": "gid", "NAME_3": "name"})

# Values table
values_df = weekly_df[["series_id", "week", "value"]].rename(columns={"week": "date"})

# Save outputs
meta_df.to_parquet("output/climate_adm3_metadata.parquet", index=False)
values_df.to_parquet("output/climate_adm3_timeseries.parquet", index=False)

print("Saved metadata to output/climate_adm3_metadata.parquet")
print("Saved time series values to output/climate_adm3_timeseries.parquet")
