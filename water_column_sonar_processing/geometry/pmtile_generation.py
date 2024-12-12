import glob
import os
from pathlib import Path
import fiona
import s3fs
import numpy as np
import pandas as pd
import xarray as xr
import geopandas
import geopandas as gpd
import pyogrio
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import LineString

MAX_POOL_CONNECTIONS = 64
MAX_CONCURRENCY = 64
MAX_WORKERS = 64
GB = 1024**3


class PMTileGeneration(object):
    #######################################################
    def __init__(
        self,
    ):
        print("123")

    #######################################################
    # This uses a local collection of file-level geojson files to create the data
    def generate_geojson_feature_collection(self):
        # This was used to read from noaa-wcsd-model-pds bucket geojson files and then to
        # generate the geopandas dataframe which could be exported to another comprehensive
        # geojson file. That
        result = list(Path("/Users/r2d2/Documents/echofish/geojson").rglob("*.json"))
        # result = result[:100]
        jjj = 0
        pieces = []
        for jjj in range(len(result)):
            file_name = os.path.normpath(result[jjj]).split(os.sep)[-1]
            file_stem = os.path.splitext(os.path.basename(file_name))[0]
            geom = gpd.read_file(result[jjj]).iloc[0]["geometry"]
            # TDOO: Filter (0,0) coordinates
            if len(geom.coords.xy[0]) < 2:
                continue
            geom = LineString(list(zip(geom.coords.xy[1], geom.coords.xy[0])))
            pieces.append(
                {
                    "ship_name": os.path.normpath(result[jjj]).split(os.sep)[-4],
                    "cruise_name": os.path.normpath(result[jjj]).split(os.sep)[-3],
                    "file_stem": file_stem,
                    "file_path": result[jjj],
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
    # TODO: temporary using this to get info
    def get_info_from_zarr_store(
        self,
        ship_name,
        cruise_names,
    ):
        total_size = 0
        s3_fs = s3fs.S3FileSystem(anon=True)
        for cruise_name in cruise_names:
            path_to_zarr_store = f"s3://noaa-wcsd-zarr-pds/level_2/{ship_name}/{cruise_name}/EK60/{cruise_name}.zarr"
            zarr_store = s3fs.S3Map(root=path_to_zarr_store, s3=s3_fs)
            xr_store = xr.open_zarr(store=zarr_store, consolidated=None)
            print(f'Cruise: {cruise_name}, shape: {xr_store.time.shape[0]}')
            total_size = total_size + xr_store.time.shape[0]

    def get_geospatial_info_from_zarr_store(
        self,
        ship_name,
        cruise_name,
    ):
        """
        Open Zarr store, create geometry, write to geojson, return name
        """
        s3_fs = s3fs.S3FileSystem(anon=True)
        gps_gdf = geopandas.GeoDataFrame(
            columns=["id", "ship", "cruise", "sensor", "geometry"],
            geometry="geometry",
            crs="EPSG:4326"
        )
        path_to_zarr_store = f"s3://noaa-wcsd-zarr-pds/level_2/{ship_name}/{cruise_name}/EK60/{cruise_name}.zarr"
        # file_name = os.path.normpath(path_to_zarr_store).split(os.sep)[-1]
        # file_stem = os.path.splitext(os.path.basename(file_name))[0]
        zarr_store = s3fs.S3Map(root=path_to_zarr_store, s3=s3_fs)
        # ---Open Zarr Store--- #
        # TODO: try-except to allow failures
        print('opening store')
        # xr_store = xr.open_zarr(store=zarr_store, consolidated=False)
        xr_store = xr.open_zarr(store=zarr_store, consolidated=None)
        print(xr_store.Sv.shape)
        # ---Read Zarr Store Time/Latitude/Longitude--- #
        latitude = xr_store.latitude.values
        longitude = xr_store.longitude.values
        if np.isnan(latitude).any() or np.isnan(longitude).any():
            print(f'there was missing lat-lon data for {cruise_name}')
            return None
        # ---Add To GeoPandas Dataframe--- #
        # TODO: experiment with tolerance "0.001"
        geom = LineString(list(zip(longitude, latitude))).simplify(tolerance=0.001, preserve_topology=True)
        gps_gdf.loc[0] = (0, "Henry_B._Bigelow", cruise_name, "EK60", geom)  # (ship, cruise, sensor, geometry)
        gps_gdf.set_index('id', inplace=True)
        gps_gdf.to_file(f"dataframe_{cruise_name}.geojson", driver="GeoJSON") #, engine="pyogrio")
        return cruise_name

    #######################################################
    def open_zarr_stores_with_thread_pool_executor(
            self,
            cruises: list,
    ):
        # 'cruises' is a list of cruises to process
        completed_cruises = []
        try:
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [
                    executor.submit(
                        self.get_geospatial_info_from_zarr_store,
                        "Henry_B._Bigelow",  # ship_name
                        cruise,  # cruise_name
                    )
                    for cruise in cruises
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        completed_cruises.extend([result])
        except Exception as err:
            print(err)
        print("Done opening zarr stores using thread pool.")
        return completed_cruises # Took ~12 minutes

    #######################################################
    # https://docs.protomaps.com/pmtiles/create
    def aggregate_geojson_into_dataframe(
        self
    ):
        """
        iterate through cruises, threadpoolexecute geojson creation, aggregate geojson files into df,
        """
        gps_gdf = geopandas.GeoDataFrame(
            columns=["id", "ship", "cruise", "sensor", "geometry"],
            geometry="geometry",
            crs="EPSG:4326"
        )

        file_type = 'dataframe_*.geojson'
        geojson_files = glob.glob(file_type)
        for jjj in range(len(geojson_files)):
            print(jjj)
            geom = geopandas.read_file(geojson_files[jjj])
            gps_gdf.loc[jjj] = (jjj, geom.ship[0], geom.cruise[0], geom.sensor[0], geom.geometry[0])
            #gps_gdf.loc[0] = (0, "Henry_B._Bigelow", cruise_name, "EK60", geom)  # (ship, cruise, sensor, geometry)
        print(gps_gdf)
        gps_gdf.set_index('id', inplace=True)
        gps_gdf.to_file(f"data.geojson", driver="GeoJSON", engine="pyogrio", layer_options={"ID_GENERATE": "YES"})
        return list(gps_gdf.cruise)

        # gps_gdf.loc[iii] = (iii, "Henry_B._Bigelow", cruise_name, "EK60", geom)  # (ship, cruise, sensor, geometry)
        #print('writing to file')
        #print(gps_gdf)
        # gps_gdf.set_index('id', inplace=True)
        # gps_gdf.to_file(f"dataframe_{cruise_name}.geojson", driver="GeoJSON", engine="pyogrio", layer_options={"ID_GENERATE": "YES"})
        # https://gdal.org/en/latest/drivers/vector/jsonfg.html
        # gps_gdf.to_file(
        #     f"data.geojson",
        #     driver="GeoJSON",
        #     engine="pyogrio",
        #     layer_options={"ID_FIELD": "id"}
        # )
        # gps_gdf.to_file(f"dataframe_{cruise_name}.geojson", driver="GeoJSON", engine="pyogrio", id_generate=True)

# print(fiona.supported_drivers) # {'DXF': 'rw', 'CSV': 'raw', 'OpenFileGDB': 'raw', 'ESRIJSON': 'r', 'ESRI Shapefile': 'raw', 'FlatGeobuf': 'raw', 'GeoJSON': 'raw', 'GeoJSONSeq': 'raw', 'GPKG': 'raw', 'GML': 'rw', 'OGR_GMT': 'rw', 'GPX': 'rw', 'MapInfo File': 'raw', 'DGN': 'raw', 'S57': 'r', 'SQLite': 'raw', 'TopoJSON': 'r'}
#gps_gdf.to_file('dataframe.shp', crs="EPSG:4326", engine="fiona")
# Convert geojson feature collection to pmtiles
#gps_gdf.to_file("dataframe.geojson", driver="GeoJSON", crs="EPSG:4326", engine="fiona")
#print("done")
# ---Export Shapefile--- #



#gps_gdf.set_geometry(col='geometry', inplace=True)
#gps_gdf.__geo_interface__
#gps_gdf.set_index('id', inplace=True)
#gps_gdf.to_file(f"dataframe3.geojson", driver="GeoJSON", crs="EPSG:4326", engine="fiona", index=True)

### this gives the right layer id values
#gps_gdf.to_file(f"dataframe6.geojson", driver="GeoJSON", engine="pyogrio", layer_options={"ID_GENERATE": "YES"})
# jq '{"type": "FeatureCollection", "features": [.[] | .features[]]}' --slurp input*.geojson > output.geojson
#tippecanoe -zg --projection=EPSG:4326 -o water-column-sonar-id.pmtiles -l cruises output.geojson
#tippecanoe -zg --convert-stringified-ids-to-numbers --projection=EPSG:4326 -o water-column-sonar-id.pmtiles -l cruises dataframe*.geojson
# {
# "type": "FeatureCollection",
# "name": "dataframe5",
# "features": [
# { "type": "Feature", "id": 0, "properties": { "id": 0, "ship": "Henry_B._Bigelow", "cruise": "HB0706", "sensor": "EK60" }, "geometry": { "type": "LineString", "coordinates": [ [ -72.120498657226562, 39.659671783447266 ], [ -72.120773315429688, 39.660198211669922 ] ] } },
# { "type": "Feature", "id": 1, "properties": { "id": 1, "ship": "Henry_B._Bigelow", "cruise": "HB0707", "sensor": "EK60" }, "geometry": { "type": "LineString", "coordinates": [ [ -71.797836303710938, 41.003166198730469 ], [ -71.797996520996094, 41.002998352050781 ], [ -71.798583984375, 41.002994537353516 ] ] } },
# { "type": "Feature", "id": 2, "properties": { "id": 2, "ship": "Henry_B._Bigelow", "cruise": "HB0710", "sensor": "EK60" }, "geometry": { "type": "LineString", "coordinates": [ [ -72.489486694335938, 40.331901550292969 ], [ -72.490760803222656, 40.33099365234375 ] ] } }
# ]
# }
"""
# https://docs.protomaps.com/pmtiles/create
#ogr2ogr -t_srs EPSG:4326 data.geojson dataframe.shp
# Only need to do the second one here...
tippecanoe -zg --projection=EPSG:4326 -o data.pmtiles -l cruises dataframe.geojson
tippecanoe -zg --projection=EPSG:4326 -o data.pmtiles -l cruises --coalesce-densest-as-needed --extend-zooms-if-still-dropping dataframe*.geojson
# used this to combine all the geojson files into single pmtile file (2024-12-03):
tippecanoe -zg --projection=EPSG:4326 -o data.pmtiles -l cruises --coalesce-densest-as-needed --extend-zooms-if-still-dropping dataframe*.geojson

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



total = [246847, 89911, 169763, 658047, 887640, 708771, 187099, 3672813, 4095002, 763268, 162727, 189454, 1925270, 3575857, 1031920, 1167590, 3737415, 4099957, 3990725, 3619996, 3573052, 2973090, 55851, 143192, 1550164, 3692819, 668400, 489735, 393260, 1311234, 242989, 4515760, 1303091, 704663, 270645, 3886437, 4204381, 1062090, 428639, 541455, 4206506, 298561, 1279329, 137416, 139836, 228947, 517949]