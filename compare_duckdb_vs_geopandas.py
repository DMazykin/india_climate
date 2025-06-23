"""
Compare spatial join and aggregation performance between DuckDB and GeoPandas for ERA5 climate data and ADM3 boundaries.
Input files:
- era5_monthly_data_09_24.nc (NetCDF, ERA5 grid)
- output/geoparquet/ADM_ADM_3.parquet (ADM3 boundaries)
All operations are performed in memory.
"""

import time
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import duckdb

# --- Load data ---
print("Loading ERA5 NetCDF...")
ds = xr.open_dataset("era5_monthly_data_09_24.nc")
climate_df = ds.to_dataframe().reset_index()
climate_df["date"] = pd.to_datetime(climate_df["date"].astype(str), format="%Y%m%d")

print("Loading ADM3 boundaries from GPKG via Ibis/DuckDB metatable...")
geopackage_path = "input/gadm41_IND.gpkg"
layer_name = "ADM_ADM_3"

# Instead of using Ibis/Arrow, use pandas DataFrame to DuckDB directly
# Remove WKB conversion for geometry, just use GeoPandas to read and pass to DuckDB
adm3_gdf = gpd.read_file(geopackage_path, layer=layer_name)
if adm3_gdf.crs is None or adm3_gdf.crs != "EPSG:4326":
    adm3_gdf = adm3_gdf.to_crs("EPSG:4326")

# Write GeoDataFrame to DuckDB using pandas DataFrame (geometry as WKB)
adm3_gdf["geometry"] = adm3_gdf["geometry"].to_wkb()
con = duckdb.connect(database=":memory:")
con.execute("INSTALL spatial; LOAD spatial;")
con.execute(f"DROP TABLE IF EXISTS {layer_name}")
con.register("adm3_df", adm3_gdf)
con.execute(
    f"CREATE TABLE {layer_name} AS SELECT *, ST_GeomFromWKB(geometry) AS geometry FROM adm3_df"
)

# Convert back to GeoDataFrame for GeoPandas (for fair comparison)
adm3_gdf["geometry"] = gpd.GeoSeries.from_wkb(adm3_gdf["geometry"])

# --- Prepare climate_df for spatial join ---
climate_df["geometry"] = climate_df.apply(
    lambda row: Point(row["longitude"], row["latitude"]), axis=1
)
climate_gdf = gpd.GeoDataFrame(climate_df, geometry="geometry", crs=adm3_gdf.crs)

# --- Option 1: GeoPandas spatial join ---
print("\n--- GeoPandas spatial join and aggregation ---")
t0 = time.time()
joined = gpd.sjoin(climate_gdf, adm3_gdf, how="left", predicate="within")
agg_gpd = joined.groupby(["GID_3", "NAME_3", "date"])["t2m"].mean().reset_index()
t1 = time.time()
print(f"GeoPandas join+agg time: {t1 - t0:.2f} seconds. Rows: {len(agg_gpd)}")

# --- Option 2: DuckDB spatial join ---
print("\n--- DuckDB spatial join and aggregation ---")
t2 = time.time()
# Load data into DuckDB in memory
# Remove erroneous read_csv_auto/INSERT line and use in-memory DataFrame
climate_mem_con = duckdb.connect(database=":memory:")
climate_mem_con.execute("INSTALL spatial; LOAD spatial;")
# Register ADM3 boundaries from Parquet (geometry column is valid WKB)
climate_mem_con.execute(
    "CREATE TABLE adm3 AS SELECT * FROM read_parquet('output/geoparquet/ADM_ADM_3.parquet')"
)
# Register climate_df as a DuckDB table
climate_mem_con.register(
    "climate_mem", climate_df[["date", "latitude", "longitude", "t2m"]]
)
# Spatial join and aggregation
query = """
SELECT
    a.GID_3,
    a.NAME_3,
    c.date,
    AVG(c.t2m) AS avg_temperature
FROM climate_mem AS c
LEFT JOIN adm3 AS a
    ON ST_Contains(a.geometry, ST_Point(c.longitude, c.latitude))
GROUP BY a.GID_3, a.NAME_3, c.date
"""
agg_duckdb = climate_mem_con.execute(query).fetchdf()
t3 = time.time()
print(f"DuckDB join+agg time: {t3 - t2:.2f} seconds. Rows: {len(agg_duckdb)}")

# --- Results ---
print("\nGeoPandas result sample:")
print(agg_gpd.head())
print("\nDuckDB result sample:")
print(agg_duckdb.head())
