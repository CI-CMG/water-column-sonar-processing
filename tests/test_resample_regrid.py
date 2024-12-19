import pytest
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.cruise import ResampleRegrid


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-prod")  # functional test
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


@pytest.fixture
def resample_regrid_test_path(test_path):
    return test_path["RESAMPLE_REGRID_TEST_PATH"]

#######################################################

### Test Interpolation ###
@mock_aws
@pytest.mark.skip(reason="no way of currently testing resample regrid")
def test_resample_regrid(resample_regrid_test_path):
    # TODO:
    #  need to create output zarr store
    #  need to create geojson
    #  need to populate dynamodb
    #

    # Opens s3 input model store as xr and writes data to output model store

    # HB0706 - 53 files
    # bucket_name = 'noaa-wcsd-model-pds'
    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0706"
    sensor_name = "EK60"
    # file_name = "D20070719-T232718.model"  # first file
    # file_name = "D20070720-T021024.model"  # second file
    # file_name = "D20070720-T224031.model"  # third file, isn't in dynamodb
    # "D20070719-T232718.model"
    # file_name_stem = Path(file_name).stem  # TODO: remove
    table_name = "r2d2-dev-echofish-EchoFish-File-Info"

    resample_regrid = ResampleRegrid()
    resample_regrid.resample_regrid(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
    )


@mock_aws
@pytest.mark.skip(reason="TODO: implement this")
def test_interpolate(resample_regrid_test_path):
    # Get two raw files with extreme range differences between the two,
    # generate zarr stores,
    # get the last part of the first file and first part of the second file
    # and write out to new single zarr store ...save in test resources
    # read in the file here
    """
    Possible test files:
        Henry_B._Bigelow HB0707 D20070712-T124906.raw
            max_echo_range: 249.792, min_echo_range: 0.19, num_ping_time_dropna: 7706
            raw 158 MB
        Henry_B._Bigelow HB0707 D20070712-T152416.raw
            max_echo_range: 999.744, min_echo_range: 0.19, num_ping_time_dropna: 4871
            raw 200 MB

    """
    pass

#######################################################
#######################################################
#######################################################
