import os
from pathlib import Path
import fiona
import s3fs
import numpy as np
import pandas as pd
import xarray as xr
import geopandas
import geopandas as gpd
from shapely.geometry import LineString

from src.water_column_sonar_processing.aws import S3Manager, S3FSManager


class PMTileGeneration(object):
    #######################################################
    def __init__(
        self,
    ):
        print("123")

    #######################################################
    # This uses a local collection of file-level geojson files
    # to create the data
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
            geom = gpd.read_file(result[iii]).iloc[0]["geometry"]
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
        gps_gdf = gpd.GeoDataFrame(
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
    # https://docs.protomaps.com/pmtiles/create
    def pmtile_generator(self):
        level_2_cruises = [
            "HB0706",
            "HB0707",
            "HB0710",
            "HB0711",
            #"HB0802",
            "HB0803",
            "HB0805",
            "HB0806",
            "HB0807",
            "HB0901",
            "HB0902",
            "HB0903",
            "HB0904",
            "HB0905",
            "HB1002",
            "HB1006",
            "HB1102",
            "HB1103",
            "HB1105",
            "HB1201",
            "HB1206",
            "HB1301",
            "HB1303",
            "HB1304",
            "HB1401",
            "HB1402",
            "HB1403",
            "HB1405",
            "HB1501",
            "HB1502",
            "HB1503",
            "HB1506",
            "HB1507",
            "HB1601",
            "HB1603",
            "HB1604",
            "HB1701",
            "HB1702",
            "HB1801",
            "HB1802",
            "HB1803",
            "HB1804",
            "HB1805",
            "HB1806",
            "HB1901",
            "HB1902",
            "HB1903",
            "HB1904",
            "HB1906",
            "HB1907",
            "HB2001",
            "HB2006",
            "HB2007",
            "HB20ORT",
            "HB20TR"
        ]
        s3_fs = s3fs.S3FileSystem(anon=True)
        gps_gdf = geopandas.GeoDataFrame(columns=["ship", "cruise", "geometry"], geometry="geometry", crs="EPSG:4326")
        #
        for iii in range(len(level_2_cruises)):
            print(level_2_cruises[iii])
            cruise_name = level_2_cruises[iii]
            path_to_zarr_store = f"s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/{cruise_name}/EK60/{cruise_name}.zarr"
            file_name = os.path.normpath(path_to_zarr_store).split(os.sep)[-1]
            file_stem = os.path.splitext(os.path.basename(file_name))[0]
            zarr_store = s3fs.S3Map(root=path_to_zarr_store, s3=s3_fs)
            # ---Open Zarr Store--- #
            # TODO: try-except to allow failures
            xr_store = xr.open_zarr(store=zarr_store, consolidated=None)
            print(xr_store.Sv.shape)
            # ---Read Zarr Store Time/Latitude/Longitude--- #
            #time = xr_store.time.values
            latitude = xr_store.latitude.values
            longitude = xr_store.longitude.values
            print(latitude[0], longitude[0])
            # ---Add To GeoPandas Dataframe--- #
            geom = LineString(list(zip(longitude, latitude))).simplify(tolerance=0.0001, preserve_topology=True)
            # TODO: this is tooo slow!!!!!!!
            gps_gdf.loc[iii] = ("Henry_B._Bigelow", file_stem, geom) # (ship, cruise, geometry)
        #
        # print(fiona.supported_drivers) # {'DXF': 'rw', 'CSV': 'raw', 'OpenFileGDB': 'raw', 'ESRIJSON': 'r', 'ESRI Shapefile': 'raw', 'FlatGeobuf': 'raw', 'GeoJSON': 'raw', 'GeoJSONSeq': 'raw', 'GPKG': 'raw', 'GML': 'rw', 'OGR_GMT': 'rw', 'GPX': 'rw', 'MapInfo File': 'raw', 'DGN': 'raw', 'S57': 'r', 'SQLite': 'raw', 'TopoJSON': 'r'}
        #gps_gdf.to_file('dataframe.shp', crs="EPSG:4326", engine="fiona")
        # Convert geojson feature collection to pmtiles
        gps_gdf.to_file("dataframe.geojson", driver="GeoJSON", crs="EPSG:4326", engine="fiona")
        print("done")
        # ---Export Shapefile--- #

"""
# https://docs.protomaps.com/pmtiles/create
#ogr2ogr -t_srs EPSG:4326 data.geojson dataframe.shp
# Only need to do the second one here...
tippecanoe -zg --projection=EPSG:4326 -o data.pmtiles -l cruises dataframe.geojson

TODO:
    run each one of the cruises in a separate ospool workflow.
    each process gets own store
    
"""
###########################################################

# s3_manager = S3Manager()  # endpoint_url=endpoint_url)
# # s3fs_manager = S3FSManager()
# # input_bucket_name = "test_input_bucket"
# # s3_manager.create_bucket(bucket_name=input_bucket_name)
# ship_name = "Henry_B._Bigelow"
# cruise_name = "HB0706"
# sensor_name = "EK60"
#
# # ---Scan Bucket For All Zarr Stores--- #
# # https://noaa-wcsd-zarr-pds.s3.amazonaws.com/index.html#level_2/Henry_B._Bigelow/HB0706/EK60/HB0706.zarr/
# path_to_zarr_store = f"s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0706/EK60/HB0706.zarr"
# s3 = s3fs.S3FileSystem()
# zarr_store = s3fs.S3Map(path_to_zarr_store, s3=s3)
# ds_zarr = xr.open_zarr(zarr_store, consolidated=None)
# print(ds_zarr.Sv.shape)

