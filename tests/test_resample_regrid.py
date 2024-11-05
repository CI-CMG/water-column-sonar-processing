import pytest
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.cruise.resample_regrid import ResampleRegrid


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-prod")  # functional test
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


#######################################################


### Test Interpolation ###
@mock_aws
@pytest.mark.skip(reason="no way of currently testing this")
def test_resample_regrid():
    # Opens s3 input model store as xr and writes data to output model store
    resample_regrid = ResampleRegrid()

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

    resample_regrid.resample_regrid(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
    )


#######################################################
#######################################################
#######################################################
