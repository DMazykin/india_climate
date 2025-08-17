import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import mapping


class GeoFormatConverter:
    """
    Simplified converter for GPKG and geoBoundaries geojson to Foundry-ready Parquet files.
    """

    @staticmethod
    def convert_gpkg_to_parquet(gpkg_path, adm_level, output_file):
        layer_name = f"ADM_ADM_{adm_level}"
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        gdf["geoshape"] = gdf["geometry"].apply(
            lambda geom: json.dumps(mapping(geom)) if geom is not None else None
        )
        gid_col = f"GID_{adm_level}"
        name_col = "COUNTRY" if adm_level == 0 else f"NAME_{adm_level}"
        out_df = pd.DataFrame(
            {
                "adm_level": adm_level,
                "GID": gdf[gid_col],
                "NAME": gdf[name_col] if name_col in gdf.columns else None,
                "geoshape": gdf["geoshape"],
            }
        )
        for col in ["GID", "NAME"]:
            if col in out_df.columns:
                out_df[col] = out_df[col].astype("string")
        out_df.to_parquet(output_file, index=False)
        print(f"Saved: {output_file}")
        return out_df

    @staticmethod
    def convert_geoboundaries_geojson_to_parquet(geojson_path, output_file):
        gdf = gpd.read_file(geojson_path)
        if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        gdf["geoshape"] = gdf["geometry"].apply(
            lambda geom: json.dumps(mapping(geom)) if geom is not None else None
        )

        def parse_adm_level(shape_type):
            if isinstance(shape_type, str) and shape_type.startswith("ADM"):
                try:
                    return int(shape_type[3:])
                except Exception:
                    return None
            return None

        gdf["adm_level"] = gdf["shapeType"].apply(parse_adm_level)
        out_df = pd.DataFrame(
            {
                "adm_level": gdf["adm_level"],
                "GID": gdf["shapeID"],
                "NAME": gdf["shapeName"] if "shapeName" in gdf.columns else None,
                "geoshape": gdf["geoshape"],
                "shapeISO": gdf["shapeISO"] if "shapeISO" in gdf.columns else None,
                "shapeGroup": gdf["shapeGroup"]
                if "shapeGroup" in gdf.columns
                else None,
                "shapeType": gdf["shapeType"] if "shapeType" in gdf.columns else None,
            }
        )
        for col in ["GID", "NAME", "shapeISO", "shapeGroup", "shapeType"]:
            if col in out_df.columns:
                out_df[col] = out_df[col].astype("string")
        out_df.to_parquet(output_file, index=False)
        print(f"Saved: {output_file}")
        return out_df
