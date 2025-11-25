import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString

MAX_POOL_CONNECTIONS = 64
MAX_CONCURRENCY = 64
MAX_WORKERS = 64
GB = 1024**3

bucket_name = "noaa-wcsd-zarr-pds"
ship_name = "Henry_B._Bigelow"
sensor_name = "EK60"

# TODO: get pmtiles of all the evr points


class PMTileGeneration(object):
    """
    - iterate through the zarr stores for all cruises
    - generate geojson in geopandas df, simplify linestrings
    - consolidate into singular df, one cruise per row
    - export as geojson
    - using tippecanoe, geojson --> pmtiles w linux command
    - upload to s3
    """

    #######################################################
    def __init__(
        self,
    ):
        self.bucket_name = "noaa-wcsd-zarr-pds"
        self.ship_name = "Henry_B._Bigelow"
        self.sensor_name = "EK60"

    #######################################################
    @staticmethod
    def check_all_cruises(bucket_name, cruises):
        completed = []
        for cruise_name in cruises:
            print(cruise_name)
            try:
                zarr_store = f"{cruise_name}.zarr"
                s3_zarr_store_path = f"{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{zarr_store}"
                kwargs = {"consolidated": False}
                cruise = xr.open_dataset(
                    filename_or_obj=f"s3://{s3_zarr_store_path}",
                    engine="zarr",
                    storage_options={"anon": True},
                    **kwargs,
                )
                width = cruise.Sv.shape[1]
                height = cruise.Sv.shape[0]
                depth = cruise.Sv.shape[2]
                print(
                    f"height: {height}, width: {width}, depth: {depth} = {width * height * depth}"
                )
                lats = cruise.latitude.to_numpy()
                percent_done = np.count_nonzero(~np.isnan(lats)) / width
                if percent_done != 1.0:
                    print(
                        f"percent done: {np.round(percent_done, 2)}, {np.count_nonzero(~np.isnan(cruise.latitude.values))}, {width}"
                    )
                else:
                    completed.append(cruise_name)
            except Exception as err:
                raise RuntimeError(f"Problem parsing Zarr stores, {err}")
        return completed

    #######################################################
    @staticmethod
    def get_cruise_geometry(cruise_name, index):
        print(cruise_name)
        try:
            pieces = []
            zarr_store = f"{cruise_name}.zarr"
            s3_zarr_store_path = f"{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{zarr_store}"
            cruise = xr.open_dataset(
                filename_or_obj=f"s3://{s3_zarr_store_path}",
                engine="zarr",
                storage_options={"anon": True},
                chunks={},
                cache=True,
            )
            latitude_array = cruise.latitude.to_numpy()
            longitude_array = cruise.longitude.to_numpy()
            if np.isnan(latitude_array).any() or np.isnan(longitude_array).any():
                raise RuntimeError(
                    f"There was missing lat-lon dataset for, {cruise_name}"
                )
            geom = LineString(list(zip(longitude_array, latitude_array))).simplify(
                tolerance=0.001,  # preserve_topology=True # 113
            )  # TODO: do speed check, convert linestrings to multilinestrings
            print(len(geom.coords))
            pieces.append(
                {
                    "id": index,
                    "ship_name": ship_name,
                    "cruise_name": cruise_name,
                    "sensor_name": sensor_name,
                    "geom": geom,
                }
            )
            df = pd.DataFrame(pieces)
            gps_gdf = gpd.GeoDataFrame(
                data=df[["id", "ship_name", "cruise_name", "sensor_name"]],
                geometry=df["geom"],
                crs="EPSG:4326",
            )
            print(gps_gdf)
            # {'DXF': 'rw', 'CSV': 'raw', 'OpenFileGDB': 'raw', 'ESRIJSON': 'r', 'ESRI Shapefile': 'raw', 'FlatGeobuf': 'raw', 'GeoJSON': 'raw', 'GeoJSONSeq': 'raw', 'GPKG': 'raw', 'GML': 'rw', 'OGR_GMT': 'rw', 'GPX': 'rw', 'MapInfo File': 'raw', 'DGN': 'raw', 'S57': 'r', 'SQLite': 'raw', 'TopoJSON': 'r'}
            if "GeoJSON" not in fiona.supported_drivers.keys():
                raise RuntimeError("Missing GeoJSON driver")

            gps_gdf.set_index("id", inplace=True)
            # gps_gdf.to_file(f"dataframe_{cruise_name}.geojson", driver="GeoJSON") #, crs="epsg:4326")
            return gps_gdf
        except Exception as err:
            raise RuntimeError(f"Problem parsing Zarr stores, {err}")

    #######################################################
    @staticmethod
    def aggregate_geojson_into_dataframe(geoms):
        gps_gdf = gpd.GeoDataFrame(
            columns=["id", "ship", "cruise", "sensor", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
        for iii, geom in enumerate(geoms):
            gps_gdf.loc[iii] = (
                iii,
                geom.ship_name[iii],
                geom.cruise_name[iii],
                geom.sensor_name[iii],
                geom.geometry[iii],
            )
        gps_gdf.set_index("id", inplace=True)
        gps_gdf.to_file(
            filename="dataset.geojson",
            driver="GeoJSON",
            engine="fiona",  # or "pyogrio"
            layer_options={"ID_GENERATE": "YES"},
            crs="EPSG:4326",
            id_generate=True,  # required for the feature click selection
        )
        print(gps_gdf)

    #######################################################
    def create_collection_geojson(self):
        cruises = [
            "HB0706",
            "HB0707",
            "HB0710",
            "HB0711",
            "HB0802",
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
            "HB20TR",
        ]
        completed_cruises = self.check_all_cruises(
            bucket_name=bucket_name, cruises=cruises
        )  # TODO: threadpool this
        ### create linestring ###
        geometries = []
        for jjj, completed_cruise in enumerate(
            completed_cruises
        ):  # TODO: threadpool this
            geometries.append(
                self.get_cruise_geometry(cruise_name=completed_cruise, index=jjj)
            )
        #
        self.aggregate_geojson_into_dataframe(geoms=geometries)
        #
        print(
            'Now run this: "tippecanoe --no-feature-limit -zg -o dataset.pmtiles -l cruises dataset.geojson --force"'
        )
        # # water-column-sonar-id.pmtiles
        # linux command: "tippecanoe --no-feature-limit -zg -o water-column-sonar-id.pmtiles -l cruises dataset.geojson --force"
        #   note: 'cruises' is the name of the layer
        #   size is ~3.3 MB for the pmtiles
        # then drag-and-drop here: https://pmtiles.io/#map=6.79/39.802/-71.51

    #######################################################
    # TODO: copy the .pmtiles file to the s3 bucket "noaa-wcsd-pds-index"
    #######################################################

    #######################################################
    # TODO: get threadpool working
    # def open_zarr_stores_with_thread_pool_executor(
    #     self,
    #     cruises: list,
    # ):
    #     # 'cruises' is a list of cruises to process
    #     completed_cruises = []
    #     try:
    #         with ThreadPoolExecutor(max_workers=32) as executor:
    #             futures = [
    #                 executor.submit(
    #                     self.get_geospatial_info_from_zarr_store,
    #                     "Henry_B._Bigelow",  # ship_name
    #                     cruise,  # cruise_name
    #                 )
    #                 for cruise in cruises
    #             ]
    #             for future in as_completed(futures):
    #                 result = future.result()
    #                 if result:
    #                     completed_cruises.extend([result])
    #     except Exception as err:
    #         raise RuntimeError(f"Problem, {err}")
    #     print("Done opening zarr stores using thread pool.")
    #     return completed_cruises  # Took ~12 minutes

    #######################################################


###########################################################
