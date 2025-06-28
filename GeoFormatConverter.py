import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
)
from shapely.geometry.polygon import orient
from shapely.geometry import mapping


class GeoFormatConverter:
    """
    Converts a GPKG file to a Parquet file for a given admin level, with Foundry-ready geoshape columns.
    """

    def __init__(self, geopackage_path):
        self.geopackage_path = geopackage_path

    def convert_to_parquet(self, adm_level, output_file):
        layer_name = f"ADM_ADM_{adm_level}"
        gid_col = f"GID_{adm_level}"
        name_col = ["COUNTRY", f"NAME_{adm_level}"][adm_level > 0]
        varname_col = f"VARNAME_{adm_level}" if adm_level > 0 else None
        type_col = f"TYPE_{adm_level}" if adm_level > 0 else None
        hasc_col = f"HASC_{adm_level}" if adm_level > 0 else None
        iso_col = f"ISO_{adm_level}" if adm_level == 1 else None
        allowed_types = (Polygon, MultiPolygon, LineString, MultiLineString, MultiPoint)

        def fix_geom(geom):
            if isinstance(geom, (Polygon, MultiPolygon)):
                if not geom.is_valid:
                    geom = geom.buffer(0)
                if isinstance(geom, Polygon):
                    geom = orient(geom, sign=1.0)
                elif isinstance(geom, MultiPolygon):
                    geom = MultiPolygon([orient(p, sign=1.0) for p in geom.geoms])
            return geom

        def ensure_numeric_coords(geojson_geom):
            def convert_coords(coords):
                if isinstance(coords, (list, tuple)):
                    return [convert_coords(c) for c in coords]
                try:
                    return float(coords)
                except Exception:
                    return coords

            if "coordinates" in geojson_geom:
                geojson_geom["coordinates"] = convert_coords(
                    geojson_geom["coordinates"]
                )
            return geojson_geom

        gdf = gpd.read_file(self.geopackage_path, layer=layer_name)
        if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        gdf["geoshape"] = None
        mask = gdf["geometry"].apply(
            lambda geom: isinstance(geom, allowed_types) and not isinstance(geom, Point)
        )
        gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].apply(fix_geom)
        gdf.loc[mask, "geoshape"] = gdf.loc[mask, "geometry"].apply(
            lambda geom: json.dumps(ensure_numeric_coords(mapping(geom)))
            if geom is not None
            else None
        )
        gdf.loc[~mask, "geoshape"] = None

        cols = [
            ("adm_level", adm_level),
            ("GID", gdf[gid_col]),
            ("NAME", gdf[name_col]),
            ("geoshape", gdf["geoshape"]),
        ]
        if varname_col and varname_col in gdf.columns:
            cols.append(("VARNAME", gdf[varname_col]))
        else:
            cols.append(("VARNAME", None))
        if type_col and type_col in gdf.columns:
            cols.append(("TYPE", gdf[type_col]))
        else:
            cols.append(("TYPE", None))
        if hasc_col and hasc_col in gdf.columns:
            cols.append(("HASC", gdf[hasc_col]))
        else:
            cols.append(("HASC", None))
        if iso_col and iso_col in gdf.columns:
            cols.append(("ISO", gdf[iso_col]))
        else:
            cols.append(("ISO", None))

        out_df = pd.DataFrame(
            {
                k: v if not isinstance(v, type(None)) else [None] * len(gdf)
                for k, v in cols
            }
        )
        # Ensure all string columns are dtype=str to avoid Spark schema issues
        for col in ["VARNAME", "TYPE", "HASC", "ISO"]:
            if col in out_df.columns:
                out_df[col] = out_df[col].astype('string')
        out_df.to_parquet(output_file, index=False)
        print(f"Saved: {output_file}")
        return out_df
