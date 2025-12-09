import pytest
from dotenv import find_dotenv, load_dotenv


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


@pytest.fixture
def spatiotemporal_test_path(test_path):
    return test_path["SPATIOTEMPORAL_TEST_PATH"]


#######################################################
# @mock_s3
def test_spatiotemporal(spatiotemporal_test_path, tmp_path):
    """
    # TODO: need to find a small file to test with, put into test bucket, read from there into
    This test takes a Zarr store that has been written to the noaa-wcsd-zarr bucket but hasn't
    had the speed and distance metrics added to it yet. It reads in the data and then writes out
    the respective variables.
    """
    bucket_name = "noaa-wcsd-pds"
    file_name = "D20070724-T042400.raw"
    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0706"
    sensor_name = "EK60"

    s3_path = f"s3://{bucket_name}/data/raw/{ship_name}/{cruise_name}/{sensor_name}/{file_name}"
    print(s3_path)

    assert 1 == 1
