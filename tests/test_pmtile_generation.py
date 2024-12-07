import pytest
import s3fs
from dotenv import find_dotenv, load_dotenv
import xarray as xr

from src.water_column_sonar_processing.geometry import PMTileGeneration


# from src.water_column_sonar_processing.aws import S3Manager

# @pytest.fixture
# def pmtile_generation_test_path(test_path):
#     return test_path["PMTILE_GENERATION_TEST_PATH"]

#######################################################
def setup_module():
    print("setup")
    # env_file = find_dotenv('.env-test')
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)

def teardown_module():
    print("teardown")

# @pytest.fixture(scope="session")
# def zarr_store_base():
#     # path_to_zarr_store = f"s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0706/EK60/HB0706.zarr"
#     # s3 = s3fs.S3FileSystem(anon=True)
#     # zarr_store = s3fs.S3Map(root=path_to_zarr_store, s3=s3)
#     # return zarr_store
#     path_to_zarr_store = f"s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0706/EK60/HB0706.zarr"
#     s3fs.S3FileSystem(anon=True)
#     yield

# def get_zarr():
#     print("test")
#     s3_fs = s3fs.S3FileSystem(anon=True)
#     path_to_zarr_store = f"s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0706/EK60/HB0706.zarr"
#     zarr_store = s3fs.S3Map(root=path_to_zarr_store, s3=s3_fs)
#     foo = xr.open_zarr(store=zarr_store)
#     foo.Sv.shape
#     return foo



# def test_async_s3(pmtile_generation_test_path):
#     s3_fs = s3fs.S3FileSystem(anon=True)
#     path_to_zarr_store = f"s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0706/EK60/HB0706.zarr"
#     zarr_store = s3fs.S3Map(root=path_to_zarr_store, s3=s3_fs)
#     ds_zarr = xr.open_zarr(store=zarr_store, consolidated=None)
#     print(ds_zarr.Sv.shape)
#     # _()

@pytest.mark.skip(reason="This test uses actual data in s3 buckets")
def test_get_geospatial_info_from_zarr_store():
    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0706"
    pmtile_generation = PMTileGeneration()
    processed_cruise = pmtile_generation.get_geospatial_info_from_zarr_store(ship_name, cruise_name)

    assert processed_cruise

@pytest.mark.skip(reason="This test uses actual data in s3 buckets")
def test_open_zarr_stores_with_thread_pool_executor():
    level_2_cruises = [
        "HB0706",
        "HB0707",
        "HB0710",
        # #"HB0802", # problem
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
        #"HB1103", # shapely.errors.GEOSException: IllegalArgumentException: Non-finite envelope bounds passed to index insert
        #  ^ has nans
        "HB1105",
        "HB1201",
        "HB1206",
        "HB1301",
        #"HB1303", # HB1303 missing lat/lon
        "HB1304",
        "HB1401",
        "HB1402",
        "HB1403",
        "HB1405",
        "HB1501",
        "HB1502",
        "HB1503",
        #"HB1506", # missing lat/lon
        "HB1507",
        #"HB1601", # problem, botocore.exceptions.EndpointConnectionError: Could not connect to the endpoint URL: "https://noaa-wcsd-zarr-pds.s3.amazonaws.com/level_2/Henry_B._Bigelow/HB1601/EK60/HB1601.zarr/longitude/907"
        #"HB1603", # HB1603 missing lat/lon
        #"HB1604", # missing lat/lon
        "HB1701",
        #"HB1702", # missing lat/lon
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
    pmtile_generation = PMTileGeneration()
    processed_cruises = pmtile_generation.open_zarr_stores_with_thread_pool_executor(level_2_cruises)
    print(processed_cruises)
    print(level_2_cruises)
    assert len(processed_cruises) == len(level_2_cruises)

@pytest.mark.skip(reason="This test uses actual data in s3 buckets")
def test_aggregate_geojson_into_dataframe():
    pmtile_generation = PMTileGeneration()
    processed_cruises = pmtile_generation.aggregate_geojson_into_dataframe()
    assert len(processed_cruises) > 0
    # creates data.geojson
    # then in the terminal
    # tippecanoe -zg --projection=EPSG:4326 -o water-column-sonar-id.pmtiles -l cruises dataframe.geojson


# @mock_aws
@pytest.mark.skip(reason="no way of currently testing this without accessing actual zarr stores")
def test_pmtile_generator(zarr_store_base, pmtile_generation_test_path):
    # ---Scan Bucket For All Zarr Stores--- #
    # https://noaa-wcsd-zarr-pds.s3.amazonaws.com/index.html#level_2/Henry_B._Bigelow/HB0706/EK60/HB0706.zarr/
    level_2_cruises = [
       "HB0706",
        "HB0707",
        "HB0710",
        "HB0711",
        # # #"HB0802", # problem
        # "HB0803",
        # "HB0805",
        # "HB0806",
        # "HB0807",
        # "HB0901",
        # "HB0902",
        # "HB0903",
        # "HB0904",
        # "HB0905",
        # "HB1002",
        # "HB1006",
        # "HB1102",
        # #"HB1103", # shapely.errors.GEOSException: IllegalArgumentException: Non-finite envelope bounds passed to index insert
        # #  ^ has nans
        # "HB1105",
        # "HB1201",
        # "HB1206",
        # "HB1301",
        # "HB1303",
        # "HB1304",
        # "HB1401",
        # "HB1402",
        # "HB1403",
        # "HB1405",
        # "HB1501",
        # "HB1502",
        # "HB1503",
        # "HB1506",
        # "HB1507",
        # #"HB1601", # problem, botocore.exceptions.EndpointConnectionError: Could not connect to the endpoint URL: "https://noaa-wcsd-zarr-pds.s3.amazonaws.com/level_2/Henry_B._Bigelow/HB1601/EK60/HB1601.zarr/longitude/907"
        # "HB1603",
        # "HB1604",
        # "HB1701",
        # "HB1702",
        # "HB1801",
        # "HB1802",
        # "HB1803",
        # "HB1804",
        # "HB1805",
        # "HB1806",
        # "HB1901",
        # "HB1902",
        # "HB1903",
        # "HB1904",
        # "HB1906",
        # "HB1907",
        # "HB2001",
        # "HB2006",
        # "HB2007",
        # "HB20ORT",
        # "HB20TR"
    ]
    pmtile_generation = PMTileGeneration()
    pmtile_generation.pmtile_generator(level_2_cruises)

    # output should be geojson with multiple features
