import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import mapping


class GeoHierarchyConverter:
    """
    GeoHierarchyConverter
    --------------------
    Converts geoBoundaries geojson files to Foundry-ready Parquet files with hierarchical linkage.

    Output columns:
        - shape_id: Unique ID for the current region (ADM1, ADM2, etc.)
        - adm_level: Administrative level (e.g., 1 for ADM1, 2 for ADM2)
        - name: Name of the current region
        - shape_iso: ISO code for the current region (if available)
        - geo_shape: Geometry as GeoJSON string
        - parent_shape_id: Unique ID of the parent region (e.g., ADM1 shape_id for ADM2)
        - parent_name: Name of the parent region
        - parent_iso: ISO code of the parent region (if available)

    This structure supports hierarchical queries and linking between administrative levels.
    Parent columns are filled if present in the geoBoundaries file.
    All string columns are explicitly cast to string dtype for compatibility with Spark/Foundry.
    """

    @staticmethod
    def convert_geoboundaries_geojson_to_parquet(geojson_path, output_file):
        gdf = gpd.read_file(geojson_path)
        if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        gdf["geo_shape"] = gdf["geometry"].apply(
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
                "shape_id": gdf["shapeID"],
                "adm_level": gdf["adm_level"],
                "name": gdf["shapeName"] if "shapeName" in gdf.columns else None,
                "shape_iso": gdf["shapeISO"] if "shapeISO" in gdf.columns else None,
                "geo_shape": gdf["geo_shape"],
                "parent_shape_id": gdf["parent"] if "parent" in gdf.columns else None,
                "parent_name": gdf["parentName"]
                if "parentName" in gdf.columns
                else None,
                "parent_iso": gdf["parentISO"] if "parentISO" in gdf.columns else None,
            }
        )
        for col in [
            "shape_id",
            "name",
            "shape_iso",
            "geo_shape",
            "parent_shape_id",
            "parent_name",
            "parent_iso",
        ]:
            if col in out_df.columns:
                out_df[col] = out_df[col].astype("string")
        out_df.to_parquet(output_file, index=False)
        print(f"Saved: {output_file}")
        return out_df

    @staticmethod
    def convert_geoboundaries_hierarchy_to_parquet(
        adm0_path, adm1_path, adm2_path, output_file
    ):
        """
        Converts ADM0, ADM1, ADM2 geoBoundaries geojsons into a unified Parquet with hierarchical linkage using geometry-based matching.
        Parent columns are inferred by spatial containment (ADM2 within ADM1, ADM1 within ADM0).
        All string columns are explicitly cast to string dtype for compatibility.
        """
        # Read all levels
        adm0 = gpd.read_file(adm0_path)
        adm1 = gpd.read_file(adm1_path)
        adm2 = gpd.read_file(adm2_path)
        for gdf in [adm0, adm1, adm2]:
            if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                gdf.to_crs("EPSG:4326", inplace=True)
            gdf["geo_shape"] = gdf["geometry"].apply(
                lambda geom: json.dumps(mapping(geom)) if geom is not None else None
            )
        # ADM1 → ADM0
        adm1["parent_shape_id"] = None
        adm1["parent_name"] = None
        adm1["parent_iso"] = None
        for idx, row in adm1.iterrows():
            parent = adm0[adm0.geometry.contains(row.geometry)]
            if not parent.empty:
                adm1.at[idx, "parent_shape_id"] = (
                    parent.iloc[0]["shapeID"] if "shapeID" in parent.columns else None
                )
                adm1.at[idx, "parent_name"] = (
                    parent.iloc[0]["shapeName"]
                    if "shapeName" in parent.columns
                    else None
                )
                adm1.at[idx, "parent_iso"] = (
                    parent.iloc[0]["shapeISO"] if "shapeISO" in parent.columns else None
                )
        # ADM2 → ADM1
        adm2["parent_shape_id"] = None
        adm2["parent_name"] = None
        adm2["parent_iso"] = None
        for idx, row in adm2.iterrows():
            parent = adm1[adm1.geometry.contains(row.geometry)]
            if not parent.empty:
                adm2.at[idx, "parent_shape_id"] = (
                    parent.iloc[0]["shapeID"] if "shapeID" in parent.columns else None
                )
                adm2.at[idx, "parent_name"] = (
                    parent.iloc[0]["shapeName"]
                    if "shapeName" in parent.columns
                    else None
                )
                adm2.at[idx, "parent_iso"] = (
                    parent.iloc[0]["shapeISO"] if "shapeISO" in parent.columns else None
                )
        # Add adm_level
        for gdf, level in zip([adm0, adm1, adm2], [0, 1, 2]):
            gdf["adm_level"] = level

        # Build output DataFrame
        def build_out_df(gdf):
            return pd.DataFrame(
                {
                    "shape_id": gdf["shapeID"],
                    "adm_level": gdf["adm_level"],
                    "name": gdf["shapeName"] if "shapeName" in gdf.columns else None,
                    "shape_iso": gdf["shapeISO"] if "shapeISO" in gdf.columns else None,
                    "geo_shape": gdf["geo_shape"],
                    "parent_shape_id": gdf["parent_shape_id"]
                    if "parent_shape_id" in gdf.columns
                    else None,
                    "parent_name": gdf["parent_name"]
                    if "parent_name" in gdf.columns
                    else None,
                    "parent_iso": gdf["parent_iso"]
                    if "parent_iso" in gdf.columns
                    else None,
                }
            )

        out_df = pd.concat(
            [build_out_df(adm0), build_out_df(adm1), build_out_df(adm2)],
            ignore_index=True,
        )
        for col in [
            "shape_id",
            "name",
            "shape_iso",
            "geo_shape",
            "parent_shape_id",
            "parent_name",
            "parent_iso",
        ]:
            if col in out_df.columns:
                out_df[col] = out_df[col].astype("string")
        out_df.to_parquet(output_file, index=False)
        print(f"Saved: {output_file}")
        return out_df

    @staticmethod
    def convert_geoboundaries_hierarchy_zips_to_parquet(
        adm0_zip, adm1_zip, adm2_zip, output_file
    ):
        """
        Reads three zip files (ADM0, ADM1, ADM2), each containing a geoBoundaries geojson, infers hierarchy, and outputs unified Parquet.
        The geojson in each zip must be named with ADM_0, ADM_1, ADM_2 in its filename, respectively.
        """
        import zipfile
        import tempfile

        def extract_geojson(zip_path, adm_level):
            with zipfile.ZipFile(zip_path, "r") as z:
                for info in z.infolist():
                    fname = info.filename
                    # Only match geojson files for the correct ADM level
                    if (
                        f"ADM_{adm_level}" in fname or f"ADM{adm_level}" in fname
                    ) and fname.endswith(".geojson"):
                        with tempfile.TemporaryDirectory() as tmpdir:
                            z.extract(fname, tmpdir)
                            path = f"{tmpdir}/{fname}"
                            gdf = gpd.read_file(path)
                            if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                                gdf = gdf.to_crs("EPSG:4326")
                            gdf["geo_shape"] = gdf["geometry"].apply(
                                lambda geom: json.dumps(mapping(geom))
                                if geom is not None
                                else None
                            )
                            return gdf
                raise ValueError(f"No geojson for ADM_{adm_level} found in {zip_path}")

        adm0 = extract_geojson(adm0_zip, 0)
        adm1 = extract_geojson(adm1_zip, 1)
        adm2 = extract_geojson(adm2_zip, 2)
        # ADM1 → ADM0
        adm1["parent_shape_id"] = None
        adm1["parent_name"] = None
        adm1["parent_iso"] = None
        for idx, row in adm1.iterrows():
            parent = adm0[adm0.geometry.contains(row.geometry)]
            if not parent.empty:
                adm1.at[idx, "parent_shape_id"] = (
                    parent.iloc[0]["shapeID"] if "shapeID" in parent.columns else None
                )
                adm1.at[idx, "parent_name"] = (
                    parent.iloc[0]["shapeName"]
                    if "shapeName" in parent.columns
                    else None
                )
                adm1.at[idx, "parent_iso"] = (
                    parent.iloc[0]["shapeISO"] if "shapeISO" in parent.columns else None
                )
        # ADM2 → ADM1
        adm2["parent_shape_id"] = None
        adm2["parent_name"] = None
        adm2["parent_iso"] = None
        for idx, row in adm2.iterrows():
            parent = adm1[adm1.geometry.contains(row.geometry)]
            if not parent.empty:
                adm2.at[idx, "parent_shape_id"] = (
                    parent.iloc[0]["shapeID"] if "shapeID" in parent.columns else None
                )
                adm2.at[idx, "parent_name"] = (
                    parent.iloc[0]["shapeName"]
                    if "shapeName" in parent.columns
                    else None
                )
                adm2.at[idx, "parent_iso"] = (
                    parent.iloc[0]["shapeISO"] if "shapeISO" in parent.columns else None
                )
        # Add adm_level
        for gdf, level in zip([adm0, adm1, adm2], [0, 1, 2]):
            gdf["adm_level"] = level

        def build_out_df(gdf):
            return pd.DataFrame(
                {
                    "shape_id": gdf["shapeID"],
                    "adm_level": gdf["adm_level"],
                    "name": gdf["shapeName"] if "shapeName" in gdf.columns else None,
                    "shape_iso": gdf["shapeISO"] if "shapeISO" in gdf.columns else None,
                    "geo_shape": gdf["geo_shape"],
                    "parent_shape_id": gdf["parent_shape_id"]
                    if "parent_shape_id" in gdf.columns
                    else None,
                    "parent_name": gdf["parent_name"]
                    if "parent_name" in gdf.columns
                    else None,
                    "parent_iso": gdf["parent_iso"]
                    if "parent_iso" in gdf.columns
                    else None,
                }
            )

        out_df = pd.concat(
            [build_out_df(adm0), build_out_df(adm1), build_out_df(adm2)],
            ignore_index=True,
        )
        for col in [
            "shape_id",
            "name",
            "shape_iso",
            "geo_shape",
            "parent_shape_id",
            "parent_name",
            "parent_iso",
        ]:
            if col in out_df.columns:
                out_df[col] = out_df[col].astype("string")
        out_df.to_parquet(output_file, index=False)
        print(f"Saved: {output_file}")
        return out_df
