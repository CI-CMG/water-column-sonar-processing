import xarray as xr
from dotenv import find_dotenv, load_dotenv
from geometry.line_simplification import LineSimplification

# s3fs.core.setup_logging("DEBUG")


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


# class TestGeometrySimplification(unittest.TestCase):
#     def setup_module(module):
#         print('setup')
#         pass
#
#     @pytest.fixture(scope='session', autouse=True)
#     def load_env(self):
#         # env_file = find_dotenv('.env-test')
#         env_file = find_dotenv('.env-test')
#         load_dotenv(dotenv_path=env_file, override=True)
#
#     def teardown_module(module):
#         print('teardown')


# @mock_s3
def test_speed_check():
    ### check the differences and speed of the data
    # download data
    bucket_name = "noaa-wcsd-zarr-pds"
    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0707"  # HB1906
    sensor_name = "EK60"
    # cruise = xr.open_zarr(
    #     store=f"s3://{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{zarr_store}",
    #     storage_options={'anon': True},
    #     consolidated=False,
    #     use_cftime=False,
    #     chunks={},  # 'auto', None, -1, {}
    # )
    cruise = xr.open_dataset(
        filename_or_obj=f"s3://{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{cruise_name}.zarr",
        storage_options={"anon": True},
        engine="zarr",
        chunks={},
    )
    # cruise = xr.open_dataset(filename_or_obj=store_path, engine="zarr", chunks={})
    times = cruise.time.values
    latitudes = cruise.latitude.values
    longitudes = cruise.longitude.values

    line_simplification = LineSimplification()
    foo = line_simplification.speed_check(times, latitudes, longitudes)
    print(foo)
    # pass in lat/lon/time

    #
    pass


#######################################################
