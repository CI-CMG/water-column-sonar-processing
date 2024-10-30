# from moto import mock_s3
from dotenv import load_dotenv, find_dotenv
# from pathlib import Path
from cruise.resample_regrid import ResampleRegrid
import pytest
from moto import mock_s3

#######################################################
def setup_module(module):
    print('setup')

    # env_file = find_dotenv('.env-test')
    env_file = find_dotenv('.env-prod')  # functional test

    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module(module):
    print('teardown')


#######################################################

### Test Interpolation ###
@mock_s3
@pytest.mark.skip(reason="no way of currently testing this")
def test_resample_regrid():
    # Opens s3 input zarr_manager store as xr and writes data to output zarr_manager store
    resample_regrid = ResampleRegrid()

    # HB0706 - 53 files
    # bucket_name = 'noaa-wcsd-zarr_manager-pds'
    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0706"
    sensor_name = "EK60"
    # file_name = "D20070719-T232718.zarr_manager"  # first file
    #file_name = "D20070720-T021024.zarr_manager"  # second file
    #file_name = "D20070720-T224031.zarr_manager"  # third file, isn't in dynamodb
    # "D20070719-T232718.zarr_manager"
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