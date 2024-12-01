import os
from pathlib import Path
import fiona
import geopandas
import pandas as pd
import s3fs
import xarray as xr
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
    # https://docs.protomaps.com/pmtiles/create
    def get_zarr_store(self):
        print("Getting zarr store")
        s3fs_manager = S3FSManager()

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

