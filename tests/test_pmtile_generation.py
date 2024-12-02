import pytest
import s3fs
from dotenv import find_dotenv, load_dotenv
import xarray as xr


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

# @mock_aws
# def test_pmtile_generation(zarr_store_base, pmtile_generation_test_path):
    # ---Scan Bucket For All Zarr Stores--- #
    # https://noaa-wcsd-zarr-pds.s3.amazonaws.com/index.html#level_2/Henry_B._Bigelow/HB0706/EK60/HB0706.zarr/
    print("test")
    s3_fs = s3fs.S3FileSystem(anon=True)
    path_to_zarr_store = f"s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0706/EK60/HB0706.zarr"
    zarr_store = s3fs.S3Map(root=path_to_zarr_store, s3=s3_fs)

    # ---Open Zarr Store--- #
    foo = xr.open_zarr(store=zarr_store, consolidated=None)
    print(foo.Sv.shape)
    time = foo.time.values
    latitude = foo.latitude.values
    longitude = foo.longitude.values
    print(time[0], latitude[0], longitude[0])

    # zarr_store_base2 = s3fs.S3Map(root=path_to_zarr_store, s3=zarr_store_base)
    # ds_zarr = xr.open_zarr(store=zarr_store_base2)
    # print(ds_zarr.Sv.shape)


    #

    # ---Read Zarr Store Time/Latitude/Longitude--- #

    # ---Add To GeoPandas Dataframe--- #

    # ---Export Shapefile--- #


#######################################################
# def test_get_zarr_store():
#     assert False
