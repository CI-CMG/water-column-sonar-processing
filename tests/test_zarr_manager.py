# import os
import zarr
import numcodecs
# import numpy as np
# import xarray as xr
# from moto import mock_s3
from moto import mock_aws
from dotenv import load_dotenv, find_dotenv
# import warnings

# import gc
import os
# import time
# import boto3
# import shutil
import numpy as np
# import pandas as pd
import xarray as xr

from aws_manager.s3_manager import S3Manager
from zarr_manager.zarr_manager import ZarrManager


#######################################################
# def setup_module(module):
def setup_module():
    print('setup')
    env_file = find_dotenv('.env-test')
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print('teardown')

TEMPDIR = "/tmp"

#######################################################
@mock_aws
def test_zarr_manager(tmp_path=TEMPDIR):
    # Tests creating zarr_manager store and opening with both xarray and
    # zarr_manager libraries

    temporary_directory = str(tmp_path)

    ship_name = "test_ship"
    cruise_name = "test_cruise"
    sensor_name = "EK60"
    frequencies = [18_000, 38_000, 70_000, 120_000]

    zarr_manager = ZarrManager()
    zarr_manager.create_zarr_store(
        path=temporary_directory,
        ship_name=ship_name,
        cruise_name=cruise_name,  # TODO: just pass stem
        sensor_name=sensor_name,
        frequencies=frequencies,
        width=1201,  # number of ping samples recorded
        min_echo_range=0.50,
        max_echo_range=250.00,  # maximum depth found in cruise
        calibration_status=True
    )

    assert os.path.exists(f"{temporary_directory}/{cruise_name}.zarr")

    # TODO: copy to s3 bucket...
    numcodecs.blosc.use_threads = False
    numcodecs.blosc.set_nthreads(1)

    # synchronizer = zarr_manager.ProcessSynchronizer(f"/mnt/zarr_manager/{ship_name}_{cruise_name}.sync")

    cruise_zarr = zarr.open(store=f"{temporary_directory}/{cruise_name}.zarr", mode="r")  # synchronizer=synchronizer)
    print(cruise_zarr.info)

    assert cruise_zarr.Sv.shape == (501, 1201, len(frequencies))  # (depth, time, frequency)
    assert cruise_zarr.Sv.chunks == (512, 512, 1)  # TODO: use enum?

    # Open Zarr store with Xarray
    # TODO: move to separate test
    file_xr = xr.open_zarr(store=f"{temporary_directory}/{cruise_name}.zarr", consolidated=None)  # synchronizer=SYNCHRONIZER)
    print(file_xr)

    # for newly initialized zarr_manager store all the timestamps will be 0 epoch time
    assert file_xr.time.values[0] == np.datetime64('1970-01-01T00:00:00.000000000')
    assert str(file_xr.time.values[0].dtype) == 'datetime64[ns]'

    # TODO: test to ensure the dimensions are in proper order
    assert file_xr.Sv.dims == ('depth', 'time', 'frequency')
    assert file_xr.Sv.shape == (501, 1201, 4)

    assert file_xr.attrs['processing_software_name'] == 'echofish'
    assert file_xr.attrs['calibration_status']  # Note: calibration status default is False
    assert file_xr.attrs['ship_name'] == 'test_ship'
    assert file_xr.attrs['cruise_name'] == 'test_cruise'
    assert file_xr.attrs['sensor_name'] == 'EK60'

    assert file_xr.Sv.dtype == 'float32'
    assert file_xr.latitude.dtype == 'float32'
    assert file_xr.longitude.dtype == 'float32'
    assert file_xr.depth.dtype == 'float32'
    assert file_xr.time.dtype == "<M8[ns]"
    assert file_xr.frequency.dtype == "float64"

    # TODO: test depths
    # TODO: test compression



#######################################################
@mock_aws
def test_open_zarr_with_zarr_read_write(tmp_path):
    temporary_directory = str(tmp_path)

    # TODO: open with zarr_manager python library and check format
    test_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")

    # create a bucket
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=test_bucket_name)

    # initialize zarr_manager store
    ship_name = "test_ship"
    cruise_name = "test_cruise"
    sensor_name = "test_sensor"

    zarr_manager = ZarrManager()
    zarr_path = zarr_manager.create_zarr_store(
        path=temporary_directory,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        frequencies=[18_000, 38_000, 70_000, 120_000],
        width=1201,  # number of ping samples recorded
        # height=123,
        min_echo_range=0.5,
        max_echo_range=250.0,  # maximum depth found in cruise
        calibration_status=True
    )

    # TODO: copy store to bucket

    # TODO: open zarr_manager store with zarr_manager
    pass


#######################################################
@mock_aws
def test_open_zarr_with_xarray(tmp_path):
    # TODO: open with xarray
    #  [1] check timestamps are in proper format
    #  [2] check that lat/lons are formatted (need data)
    #  [3] check
    # TODO: open with zarr_manager python library and check format
    ship_name = "Okeanos_Explorer"
    cruise_name = "EX1404L2"
    sensor_name = "EK60"
    # file_name = "EX1404L2_EK60_-D20140908-T020733.raw"

    temporary_directory = str(tmp_path)

    bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")

    # create a bucket
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=bucket_name)

    # initialize zarr_manager store
    # zarr_name = f"{cruise_name}.zarr_manager"
    min_echo_range = 0.50
    max_echo_range = 250.0

    zarr_manager = ZarrManager()

    height = len(zarr_manager.get_depth_values(
        min_echo_range=min_echo_range,
        max_echo_range=max_echo_range
    ))

    zarr_manager.create_zarr_store(
        path=temporary_directory,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        frequencies=[18_000, 38_000, 70_000, 120_000],
        width=1201,  # number of ping samples recorded
        # height=height,  # TODO: is this redundant with the min & max echo range?
        min_echo_range=min_echo_range,
        max_echo_range=max_echo_range,
        calibration_status=True
    )

    # copy store to bucket
    # TODO: create function to get list of files in zarr_manager store
    # s3_manager.upload_files_to_bucket(
    #       local_path,  # TODO: change to path
    #       s3_path
    # )

    # open zarr_manager store with zarr_manager

    #assert root.Sv.shape == (501, 1201, 4)


#######################################################
# @mock_s3
# def test_write_zarr_with_synchronizer(tmp_path):
#     pass


#######################################################
### Test 1 of 5 for depth values ###
def test__get_depth_values_shallow_and_small_epsilon():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=0.17,
        max_echo_range=101,
    )
    assert len(depths) == 595
    assert depths[0] == 0.17
    assert depths[-1] == 101

### Test 2 of 5 for depth values ###
def test__get_depth_values_shallow_and_large_epsilon():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=1.31,
        max_echo_range=24,
    )
    assert len(depths) == 19
    assert depths[0] == 1.31
    assert depths[-1] == 24

### Test 3 of 5 for depth values ###
def test__get_depth_values_deep_and_small_epsilon():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=0.11,
        max_echo_range=221.1,
    )
    assert len(depths) == 2011
    assert depths[0] == 0.11
    assert depths[-1] == 221.1  # TODO: do we want this to be np.ceil(x)

### Test 4 of 5 for depth values ###
def test__get_depth_values_deep_and_large_epsilon():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=1.31,
        max_echo_range=222.2,
    )
    assert len(depths) == 170
    assert depths[0] == 1.31
    # TODO: would it be better to have whole numbers?
    assert depths[-1] == 222.2

### Test 5 of 5 for depth values ###
def test__get_depth_values_half_meter():
    zarr_manager = ZarrManager()
    depths = zarr_manager.get_depth_values(
        min_echo_range=0.50,
        max_echo_range=250.,
    )
    assert len(depths) == 501
    assert depths[0] == 0.50
    assert depths[-1] == 250

#######################################################
#######################################################
