import os

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

from water_column_sonar_processing.aws import S3Manager
from water_column_sonar_processing.geometry import Spatiotemporal
from water_column_sonar_processing.model import ZarrManager
from water_column_sonar_processing.utility import Constants


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


@pytest.fixture(scope="module")
def moto_server():
    """Fixture to run a mocked AWS server for testing."""
    # Note: pass `port=0` to get a random free port.
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def spatiotemporal_test_path(test_path):
    return test_path["SPATIOTEMPORAL_TEST_PATH"]


#######################################################
@mock_aws
def test_spatiotemporal(spatiotemporal_test_path, tmp_path, moto_server):
    """
    # TODO: need to find a small file to test with, put into test bucket, read from there into
    This test takes a Zarr store that has been written to the noaa-wcsd-zarr bucket but hasn't
    had the speed and distance metrics added to it yet. It reads in the data and then writes out
    the respective variables.
    """

    s3_manager = S3Manager(endpoint_url=moto_server)

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0707"
    sensor_name = "EK60"

    output_bucket_name = "test_output_bucket"

    s3_manager.create_bucket(bucket_name=output_bucket_name)
    print(s3_manager.list_buckets())

    zarr_prefix = os.path.join(
        str(Constants.LEVEL_2.value), ship_name, cruise_name, sensor_name
    )
    s3_manager.upload_zarr_store_to_s3(
        output_bucket_name=output_bucket_name,
        local_directory=spatiotemporal_test_path,
        object_prefix=zarr_prefix,
        cruise_name="HB0707",
    )
    assert len(s3_manager.list_objects(bucket_name=output_bucket_name, prefix="")) > 9

    test_zarr_manager_before = ZarrManager()
    zarr_store_before = test_zarr_manager_before.open_l2_zarr_store_with_xarray(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        bucket_name=output_bucket_name,
        endpoint_url=moto_server,
    )
    assert np.all(np.isnan(zarr_store_before.distance.values))  # all are nan

    # write speed & distance
    spatiotemporal = Spatiotemporal()
    spatiotemporal.add_speed_and_distance(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        bucket_name=output_bucket_name,
        endpoint_url=moto_server,
    )

    test_zarr_manager = ZarrManager()
    test_output_zarr_store = test_zarr_manager.open_l2_zarr_store_with_xarray(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        bucket_name=output_bucket_name,
        endpoint_url=moto_server,
    )
    #
    assert np.mean(test_output_zarr_store.speed.values) > 16  # 16.420208
    assert np.mean(test_output_zarr_store.distance.values) > 9  # 9.899247
