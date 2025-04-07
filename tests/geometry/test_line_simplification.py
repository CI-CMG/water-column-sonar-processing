import numpy as np
import pytest
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


@pytest.fixture(scope="function")
def line_simplification_tmp_path(test_path):
    return test_path["LINE_SIMPLIFICATION_TEST_PATH"]


def test_filter_coordinates(line_simplification_tmp_path):
    ### check the differences and speed of the dataset

    # open the test dataset
    cruise = xr.open_dataset(
        filename_or_obj=line_simplification_tmp_path.joinpath(
            "HB1906_geospatial_coordinates.zarr"
        ),
        engine="zarr",
    )
    longitudes = cruise.longitude.values
    latitudes = cruise.latitude.values

    # TODO: get better description for what is happening
    # get lat/lon/time, want two things
    # 1) replace values with smoothed positions for speed etc.
    # 2) get geometry for drawing on map viewer
    line_simplification = LineSimplification()
    filtered_coordinates = line_simplification.kalman_filter(
        longitudes[1:1000], latitudes[1:1000]
    )
    print(filtered_coordinates)
    assert filtered_coordinates.shape[1] == 2


# @mock_s3
def test_get_speeds():
    ### check the differences and speed of the dataset
    #
    # TODO: create offline test dataset
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
