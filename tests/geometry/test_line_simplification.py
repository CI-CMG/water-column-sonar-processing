import numpy as np
import xarray as xr
from dotenv import find_dotenv, load_dotenv

from water_column_sonar_processing.geometry import LineSimplification


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


def test_filter_coordinates():
    ### check the differences and speed of the data
    #
    # TODO: create offline test data
    #
    bucket_name = "noaa-wcsd-zarr-pds"
    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0707"
    sensor_name = "EK60"
    cruise = xr.open_dataset(
        filename_or_obj=f"s3://{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{cruise_name}.zarr",
        storage_options={"anon": True},
        engine="zarr",
        chunks={},
    )
    longitudes = cruise.longitude.values
    latitudes = cruise.latitude.values

    line_simplification = LineSimplification()
    # filtered_coordinates = line_simplification.kalman_filter(longitudes[3_801:5_501], latitudes[3_801:5_501])
    filtered_coordinates = line_simplification.kalman_filter(
        longitudes[1:1000], latitudes[1:1000]
    )
    print(filtered_coordinates)
    assert filtered_coordinates.shape[1] == 2


# @mock_s3
def test_get_speeds():
    ### check the differences and speed of the data
    #
    # TODO: create offline test data
    #
    bucket_name = "noaa-wcsd-zarr-pds"
    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0707"
    # cruise_name = "HB1906"
    sensor_name = "EK60"
    cruise = xr.open_dataset(
        filename_or_obj=f"s3://{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{cruise_name}.zarr",
        storage_options={"anon": True},
        engine="zarr",
        chunks={},
    )
    times = cruise.time.values
    latitudes = cruise.latitude.values
    longitudes = cruise.longitude.values

    line_simplification = LineSimplification()
    line_speeds = line_simplification.get_speeds(times, latitudes, longitudes)
    print(line_speeds)
    assert int(np.nanmean(line_speeds)) == 7  # 7.54 meters per second


#######################################################
