import time

import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.aws import S3Manager
from water_column_sonar_processing.index import IndexManager


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    # env_file = find_dotenv(".env-prod")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


@pytest.fixture
def index_test_path(test_path):
    return test_path["INDEX_TEST_PATH"]


#######################################################
@pytest.mark.skip(reason="no way of currently testing this")
@mock_aws
def test_get_calibration_information(index_test_path):  # good
    """
    Reads the calibrated_cruises.csv file and determines which cruises have calibration information saved.
    """
    input_bucket_name = "test-noaa-wcsd-pds"
    calibration_bucket = "test-noaa-wcsd-pds-index"
    calibration_key = "calibrated_cruises.csv"

    # TODO: create bucket
    s3_manager = S3Manager()

    # output_bucket_name = "test_output_bucket"
    s3_manager.create_bucket(bucket_name=calibration_bucket)
    # TODO: put objects in the output bucket so they can be deleted
    s3_manager.list_buckets()
    s3_manager.upload_file(  # TODO: upload to correct bucket
        filename=index_test_path.joinpath(calibration_key),
        bucket_name=calibration_bucket,
        key=calibration_key,
    )

    # TODO: why do i need the bucket name?
    index_manager = IndexManager(input_bucket_name, calibration_bucket, calibration_key)

    calibration_information = index_manager.get_calibration_information()
    assert "DP06_EK80" in list(calibration_information["DATASET_NAME"])
    assert "DY1906" in list(calibration_information["DATASET_NAME"])
    assert "AL0806" not in list(calibration_information["DATASET_NAME"])


# @mock_s3
# def test_index_manager(tmp_path):
#     input_bucket_name = 'noaa-wcsd-pds'
#     calibration_bucket = 'noaa-wcsd-pds-index'
#     calibration_key = 'calibrated_crusies.csv'
#
#     index = IndexManager(
#         input_bucket_name,
#         calibration_bucket,
#         calibration_key
#     )
#
#     all_ek60_data = index.index()
#     print(all_ek60_data)


# Note: this cruise has odd files where raw files aren't the first parsed
#  some sort of .ek5 files are the first ones paginated
# https://noaa-wcsd-pds.s3.amazonaws.com/index.html#data/raw/David_Starr_Jordan/DS0604/EK60/
@pytest.mark.skip(reason="no way of currently testing this")
def test_get_first_raw_file(tmp_path):
    # TODO: create bucket with test files, add one ek5 and one raw
    #  Right now this is using prod credentials
    input_bucket_name = "noaa-wcsd-pds"
    calibration_bucket = "noaa-wcsd-pds-index"
    calibration_key = "calibrated_crusies.csv"
    index_manager = IndexManager(input_bucket_name, calibration_bucket, calibration_key)
    foo = index_manager.get_first_raw_file(
        ship_name="David_Starr_Jordan", cruise_name="DS0604", sensor_name="EK60"
    )
    assert (
        foo == "data/raw/David_Starr_Jordan/DS0604/EK60/DSJ0604-D20060406-T035914.raw"
    )


# TODO: mock this, right now it is generating csvs for all ek60 cruises
#  instead should be called in ospool
@pytest.mark.skip(reason="no way of currently testing this")
def test_get_all_cruise_raw_files(tmp_path):
    start_time = time.time()
    # TODO: scan bucket to find a set of the smallest files
    input_bucket_name = "noaa-wcsd-pds"
    calibration_bucket = "noaa-wcsd-pds-index"
    calibration_key = "calibrated_crusies.csv"

    index_manager = IndexManager(input_bucket_name, calibration_bucket, calibration_key)

    ship_prefixes = index_manager.list_ships(prefix="data/raw/")
    cruise_prefixes = index_manager.list_cruises(ship_prefixes=ship_prefixes)
    ek60_cruise_prefixes = index_manager.list_ek60_cruises(
        cruise_prefixes=cruise_prefixes
    )
    print(len(ek60_cruise_prefixes))  # 1333 cruises > 479 ek60 prefixed >
    ek60_cruise_prefixes = [i for i in ek60_cruise_prefixes if "Henry_B._Bigelow" in i]
    print(len(ek60_cruise_prefixes))  # 1333 cruises > 64 henry bigelow cruises
    all_files = []
    for iii in ek60_cruise_prefixes:
        print(iii)
        ship_name = iii.split("/")[2]
        cruise_name = iii.split("/")[3]
        ### get raw file to scan datagram ###
        select_key = index_manager.get_first_raw_file(
            ship_name=ship_name, cruise_name=cruise_name, sensor_name="EK60"
        )
        ### check if datagram is ek60 ###
        datagram = index_manager.scan_datagram(select_key=select_key)
        if datagram == "CON0":  # if ek60
            print(f"{cruise_name} is ek60")
            # TODO: this is currently writing to csv, TODO: write to dynamodb
            ### create csv file with all raw file paths ###
            # index_manager.get_raw_files_csv(
            #     ship_name=ship_name, cruise_name=cruise_name, sensor_name="EK60"
            # )
            ### just get df ###
            all_files = all_files + index_manager.get_raw_files_list(
                ship_name=ship_name, cruise_name=cruise_name, sensor_name="EK60"
            )
        else:
            print(f"First datagram of {cruise_name} is not ek60.")
    print(len(all_files))
    df = pd.DataFrame(all_files)
    df.to_csv(f"{tmp_path}/Henry_B._Bigelow.csv", index=False, sep=" ", header=False)
    print(f"{tmp_path}/Henry_B._Bigelow.csv")
    print("--- %s seconds elapsed ---" % (time.time() - start_time))  # ~4 minutes


#######################################################

# TODO: for post analysis of coverage
#  need to check each cruise has same number of files in noaa-wcsd-pds and noaa-wcsd-model-pds buckets

# bigelow_cruises = [
#         "data/raw/Henry_B._Bigelow/HB0706/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0707/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0710/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0711/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0802/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0803/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0805/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0806/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0807/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0901/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0902/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0903/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0904/EK60/",
#         "data/raw/Henry_B._Bigelow/HB0905/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1002/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1006/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1102/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1103/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1105/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1201/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1206/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1301/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1303/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1304/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1401/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1402/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1403/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1405/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1501/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1502/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1503/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1506/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1507/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1601/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1603/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1604/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1701/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1702/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1801/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1802/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1803/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1804/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1805/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1806/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1901/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1902/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1903/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1904/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1906/EK60/",
#         "data/raw/Henry_B._Bigelow/HB1907/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2001/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2006/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2007/EK60/",
#         "data/raw/Henry_B._Bigelow/HB20ORT/EK60/",
#         "data/raw/Henry_B._Bigelow/HB20TR/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2101/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2102/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2103/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2201/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2202/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2203/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2204/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2205/EK60/",
#         "data/raw/Henry_B._Bigelow/HB2206/EK60/",
#     ]
