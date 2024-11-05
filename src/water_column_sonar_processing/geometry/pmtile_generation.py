import os
from pathlib import Path

# from shapely import wkt
# import json
# from shapely.geometry import shape, GeometryCollection
import fiona
import geopandas
import pandas as pd
from shapely.geometry import LineString


class PMTileGeneration(object):
    #######################################################
    def __init__(
        self,
    ):
        print("123")

    #######################################################
    def generate_geojson_feature_collection(self):
        # This was used to read from noaa-wcsd-model-pds bucket geojson files and then to
        # generate the geopandas dataframe which could be exported to another comprehensive
        # geojson file. That
        result = list(Path("/Users/r2d2/Documents/echofish/geojson").rglob("*.json"))
        # result = result[:100]
        iii = 0
        pieces = []
        for iii in range(len(result)):
            file_name = os.path.normpath(result[iii]).split(os.sep)[-1]
            file_stem = os.path.splitext(os.path.basename(file_name))[0]
            geom = geopandas.read_file(result[iii]).iloc[0]["geometry"]
            # TDOO: Filter (0,0) coordinates
            if len(geom.coords.xy[0]) < 2:
                continue
            geom = LineString(list(zip(geom.coords.xy[1], geom.coords.xy[0])))
            pieces.append(
                {
                    "ship_name": os.path.normpath(result[iii]).split(os.sep)[-4],
                    "cruise_name": os.path.normpath(result[iii]).split(os.sep)[-3],
                    "file_stem": file_stem,
                    "file_path": result[iii],
                    "geom": geom,
                }
            )
        df = pd.DataFrame(pieces)
        print(df)
        gps_gdf = geopandas.GeoDataFrame(
            data=df[
                ["ship_name", "cruise_name", "file_stem"]
            ],  # try again with file_stem
            geometry=df["geom"],
            crs="EPSG:4326",
        )
        print(fiona.supported_drivers)
        # gps_gdf.to_file('dataframe.shp', crs='epsg:4326')
        # Convert geojson feature collection to pmtiles
        gps_gdf.to_file("dataframe.geojson", driver="GeoJSON", crs="epsg:4326")
        print("done")
        """
        # need to eliminate visits to null island
        tippecanoe --no-feature-limit -zg --projection=EPSG:4326 -o dataframe.pmtiles -l cruises dataframe.geojson

        https://docs.protomaps.com/pmtiles/create
        PMTiles
        https://drive.google.com/file/d/17Bi-UIXB9IJkIz30BHpiKHXYpCOgRFge/view?usp=sharing

        Viewer
        https://protomaps.github.io/PMTiles/#map=8.91/56.0234/-166.6346
        """

    #######################################################


###########################################################
