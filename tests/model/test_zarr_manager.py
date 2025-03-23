import importlib.metadata
import os
import tempfile
from tempfile import TemporaryDirectory

import numcodecs
import numpy as np
import pytest
import xarray as xr
import zarr
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.aws import S3Manager
from water_column_sonar_processing.model import ZarrManager
from water_column_sonar_processing.utility import Constants


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


# The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future.
# Valid fixture loop scopes are: "function", "class", "module", "package", "session"
@pytest.fixture(scope="function")
def zarr_manager_tmp_path(test_path):
    return test_path["ZARR_MANAGER_TEST_PATH"]


#######################################################
# @pytest.mark.skip(reason="no way of currently testing this")
# @mock_aws
# def test_create_zarr_store(zarr_manager_tmp_path):
#     pass
#
# @pytest.mark.skip(reason="no way of currently testing this")
# @mock_aws
# def test_open_s3_zarr_store_with_zarr(zarr_manager_tmp_path):
#     pass
#
# @pytest.mark.skip(reason="no way of currently testing this")
# @mock_aws
# def test_open_s3_zarr_store_with_xarray(zarr_manager_tmp_path):
#     pass


@mock_aws
def test_zarr_manager():
    # create in a temporary directory and then check there
    tempdir = tempfile.TemporaryDirectory()
    # Tests creating model store and opening with both xarray and
    # model libraries
    # temporary_directory = "/tmp"  # str(tmp_path)
    ship_name = "test_ship"
    cruise_name = "test_cruise"
    sensor_name = "EK60"
    frequencies = [18_000, 38_000, 70_000, 120_000]

    zarr_manager = ZarrManager()
    zarr_manager.create_zarr_store(
        path=tempdir.name,  # This is created in test_resources/zarr_manager/test_cruise.zarr
        ship_name=ship_name,
        cruise_name=cruise_name,  # TODO: just pass stem
        sensor_name=sensor_name,
        frequencies=frequencies,
        width=1201,  # number of ping samples recorded
        min_echo_range=0.50,
        max_echo_range=250.00,  # maximum depth found in cruise
        cruise_min_epsilon=0.50,
        calibration_status=True,
    )

    assert os.path.exists(f"{tempdir.name}/{cruise_name}.zarr")

    # TODO: copy to s3 bucket...
    numcodecs.blosc.use_threads = False
    numcodecs.blosc.set_nthreads(1)

    # synchronizer = model.ProcessSynchronizer(f"/mnt/model/{ship_name}_{cruise_name}.sync")

    cruise_zarr = zarr.open(
        store=f"{tempdir.name}/{cruise_name}.zarr", mode="r"
    )  # synchronizer=synchronizer)
    print(cruise_zarr.info)

    assert cruise_zarr.Sv.shape == (
        500,
        1201,
        len(frequencies),
    )  # (depth, time, frequency)
    assert cruise_zarr.Sv.chunks == (
        Constants.TILE_SIZE.value,
        Constants.TILE_SIZE.value,
        1,
    )  # TODO: use enum?

    # Open Zarr store with Xarray
    # TODO: move to separate test
    file_xr = xr.open_zarr(
        store=f"{tempdir.name}/{cruise_name}.zarr", consolidated=None
    )  # synchronizer=SYNCHRONIZER)
    print(file_xr)

    # for newly initialized model store all the timestamps will be 0 epoch time
    assert file_xr.time.values[0] == np.datetime64("1970-01-01T00:00:00.000000000")
    assert str(file_xr.time.values[0].dtype) == "datetime64[ns]"

    # TODO: test to ensure the dimensions are in proper order
    assert file_xr.Sv.dims == ("depth", "time", "frequency")
    assert file_xr.Sv.shape == (500, 1201, 4)

    assert file_xr.attrs["processing_software_name"] == "echofish"
    assert file_xr.attrs[
        "calibration_status"
    ]  # Note: calibration status default is False
    assert file_xr.attrs["ship_name"] == "test_ship"
    assert file_xr.attrs["cruise_name"] == "test_cruise"
    assert file_xr.attrs["sensor_name"] == "EK60"
    current_project_version = importlib.metadata.version(
        "water_column_sonar_processing"
    )
    assert file_xr.attrs["processing_software_version"] == current_project_version

    assert file_xr.Sv.dtype == "float32"
    assert file_xr.latitude.dtype == "float32"
    assert file_xr.longitude.dtype == "float32"
    assert file_xr.depth.dtype == "float32"
    assert file_xr.time.dtype == "<M8[ns]"
    assert file_xr.frequency.dtype == "float64"  # TODO: There is a problem here
    assert file_xr.bottom.dtype == "float32"

    # TODO: test depths
    # TODO: test compression
    tempdir.cleanup()


@mock_aws
def test_open_zarr_with_zarr_read_write():
    tempdir = tempfile.TemporaryDirectory()

    # TODO: open with model python library and check format
    test_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")

    # create a bucket
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=test_bucket_name)

    # initialize model store
    ship_name = "test_ship"
    cruise_name = "test_cruise"
    sensor_name = "test_sensor"

    zarr_manager = ZarrManager()
    zarr_path = zarr_manager.create_zarr_store(
        path=tempdir.name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        frequencies=[18_000, 38_000, 70_000, 120_000],
        width=1201,  # number of ping samples recorded
        min_echo_range=0.5,
        max_echo_range=250.0,  # maximum depth found in cruise
        cruise_min_epsilon=0.5,
        calibration_status=True,
    )

    # TODO: copy store to bucket
    print(zarr_path)

    # TODO: open model store with model
    # pass

    tempdir.cleanup()


#######################################################
@mock_aws
def test_open_zarr_with_xarray():
    # TODO: open with xarray
    #  [1] check timestamps are in proper format
    #  [2] check that lat/lons are formatted (need data)
    #  [3] check
    # TODO: open with model python library and check format
    ship_name = "Okeanos_Explorer"
    cruise_name = "EX1404L2"
    sensor_name = "EK60"
    # file_name = "EX1404L2_EK60_-D20140908-T020733.raw"

    # temporary_directory = "/tmp"  # str(tmp_path)

    bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")

    # create a bucket
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=bucket_name)

    # initialize model store
    # zarr_name = f"{cruise_name}.model"
    min_echo_range = 0.50
    max_echo_range = 250.0

    zarr_manager = ZarrManager()

    tmp_path = TemporaryDirectory()

    zarr_manager.create_zarr_store(
        path=tmp_path.name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        frequencies=[18_000, 38_000, 70_000, 120_000],
        width=1201,  # number of ping samples recorded
        # height=height,  # TODO: is this redundant with the min & max echo range?
        min_echo_range=min_echo_range,
        max_echo_range=max_echo_range,
        cruise_min_epsilon=min_echo_range,  # TODO: test further
        calibration_status=True,
    )

    # copy store to bucket
    # TODO: create function to get list of files in model store
    # s3_manager.upload_files_to_bucket(
    #       local_path,  # TODO: change to path
    #       s3_path
    # )

    # open model store with model

    # assert root.Sv.shape == (501, 1201, 4)


#######################################################
# @mock_s3
# def test_write_zarr_with_synchronizer(tmp_path):
#     pass


#######################################################
### Test 1 of 5 for depth values ###
def test_get_depth_values_shallow_and_small_epsilon():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=0.17,
        max_echo_range=101,
        cruise_min_epsilon=0.17,
    )
    assert len(depths) == 594
    assert depths[0] == 0.17
    assert depths[-1] == 101


### Test 2 of 5 for depth values ###
def test_get_depth_values_shallow_and_large_epsilon():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=1.31,
        max_echo_range=24,
        cruise_min_epsilon=1.31,
    )
    assert len(depths) == 18
    assert depths[0] == 1.31
    assert depths[-1] == 24


### Test 3 of 5 for depth values ###
def test_get_depth_values_deep_and_small_epsilon():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=0.11,
        max_echo_range=221.1,
        cruise_min_epsilon=0.11,
    )
    assert len(depths) == 2009
    assert depths[0] == 0.11
    assert depths[-1] == 221.1  # TODO: do we want this to be np.ceil(x)


### Test 4 of 5 for depth values ###
def test_get_depth_values_deep_and_large_epsilon():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=1.31,
        max_echo_range=222.2,
        cruise_min_epsilon=1.31,  # int((222.2 - 1.31) / 1.31) + 1 = 169
    )  # 1.31 + 169*1.31 = 222.70
    assert len(depths) == 169
    assert depths[0] == 1.31
    # TODO: would it be better to have whole numbers?
    assert depths[-1] == 222.20


### Test 5 of 5 for depth values ###
def test_get_depth_values_half_meter():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=0.50,
        max_echo_range=250.0,
        cruise_min_epsilon=0.50,
    )
    assert len(depths) == 500
    assert depths[0] == 0.50
    assert depths[-1] == 250


def test_get_depth_values_half_meter_shallow():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=0.50,
        max_echo_range=2.0,
        cruise_min_epsilon=0.50,
    )
    assert len(depths) == 4
    assert depths[0] == 0.50
    assert depths[-1] == 2.0


#######################################################
#######################################################
