import os
import numpy as np
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.aws import DynamoDBManager, S3Manager
from water_column_sonar_processing.cruise.create_empty_zarr_store import (
    CreateEmptyZarrStore,
)


# TEMPDIR = "/tmp"
#######################################################
def setup_module():
    print("setup")

    env_file = find_dotenv(".env-test")
    # env_file = find_dotenv('.env-prod')
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


#######################################################
@mock_aws
def test_create_empty_zarr_store():
    dynamo_db_manager = DynamoDBManager()
    s3_manager = S3Manager()

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0707"  # HB0706 (53 files), HB0707 (12 files)
    sensor_name = "EK60"
    table_name = "water-column-sonar-table"
    # table_name = "ru-dev-echofish2-EchoFish-File-Info"

    # TODO:
    # [0] create bucket with test files
    input_bucket_name = os.environ.get("INPUT_BUCKET_NAME")
    output_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")
    s3_manager.create_bucket(bucket_name=input_bucket_name)
    s3_manager.create_bucket(bucket_name=output_bucket_name)
    print(s3_manager.list_buckets())
    # TODO: put at the path where it needs to be deleted
    s3_manager.put(bucket_name=input_bucket_name, key="the_key", body="the_body")

    # [1] create dynamodb table
    dynamo_db_manager.create_water_column_sonar_table(
        table_name="water-column-sonar-table"
    )

    # [2] bootstrap w/ test data
    test_channels = [
        "GPT  18 kHz 009072056b0e 2 ES18-11",
        "GPT  38 kHz 0090720346bc 1 ES38B",
        "GPT 120 kHz 0090720580f1 3 ES120-7C",
        "GPT 200 kHz 009072034261 4 ES200-7C",
    ]
    frequency = [18_000, 38_000, 120_000, 200_000]
    file_name = [
        "D20070711-T182032.raw",
        "D20070711-T210709.raw",
        "D20070712-T004447.raw",
        "D20070712-T033431.raw",
        "D20070712-T061745.raw",
        "D20070712-T100505.raw",
        "D20070712-T124906.raw",
        "D20070712-T152416.raw",
        "D20070712-T171804.raw",
        "D20070712-T201647.raw",
        "D20070712-T202050.raw",
        "D20070712-T231759.raw",
    ]
    max_echo_range = [
        249.792,
        249.792,
        249.792,
        249.792,
        249.792,
        249.792,
        249.792,
        999.744,
        249.792,
        249.792,
        249.792,
        249.792,
    ]
    min_echo_range = 0.25
    num_ping_time_dropna = [
        9778,
        9742,
        9780,
        9775,
        9733,
        3207,
        7705,
        4869,
        9566,
        158,
        9760,
        5838,
    ]
    start_time = [
        "2007-07-11T18:20:32.657Z",
        "2007-07-11T21:07:09.360Z",
        "2007-07-12T00:44:47.610Z",
        "2007-07-12T03:34:31.579Z",
        "2007-07-12T06:17:45.579Z",
        "2007-07-12T10:05:05.579Z",
        "2007-07-12T12:49:06.313Z",
        "2007-07-12T15:24:16.032Z",
        "2007-07-12T17:18:04.032Z",
        "2007-07-12T20:16:47.985Z",
        "2007-07-12T20:20:50.079Z",
        "2007-07-12T23:17:59.454Z",
    ]
    end_time = [
        "2007-07-11T21:07:08.360Z",
        "2007-07-12T00:44:45.610Z",
        "2007-07-12T03:34:30.579Z",
        "2007-07-12T06:17:44.579Z",
        "2007-07-12T10:05:02.579Z",
        "2007-07-12T12:45:42.438Z",
        "2007-07-12T15:24:01.032Z",
        "2007-07-12T17:18:03.032Z",
        "2007-07-12T20:15:54.157Z",
        "2007-07-12T20:19:25.985Z",
        "2007-07-12T23:17:58.454Z",
        "2007-07-13T00:55:17.454Z",
    ]
    zarr_path = [
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070711-T182032.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070711-T210709.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T004447.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T033431.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T061745.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T100505.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T124906.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T152416.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T171804.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T201647.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T202050.zarr",
        "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T231759.zarr",
    ]
    for iii in range(0, len(file_name)):
        dynamo_db_manager.update_item(
            table_name=table_name,
            key={
                "FILE_NAME": {"S": file_name[iii]},  # Partition Key
                "CRUISE_NAME": {"S": cruise_name},  # Sort Key
            },
            expression_attribute_names={
                "#CH": "CHANNELS",
                "#ET": "END_TIME",
                "#ED": "ERROR_DETAIL",
                "#FR": "FREQUENCIES",
                "#MA": "MAX_ECHO_RANGE",
                "#MI": "MIN_ECHO_RANGE",
                "#ND": "NUM_PING_TIME_DROPNA",
                "#PS": "PIPELINE_STATUS",  # testing this updated
                "#PT": "PIPELINE_TIME",  # testing this updated
                "#SE": "SENSOR_NAME",
                "#SH": "SHIP_NAME",
                "#ST": "START_TIME",
                "#ZB": "ZARR_BUCKET",
                "#ZP": "ZARR_PATH",
            },
            expression_attribute_values={
                ":ch": {"L": [{"S": i} for i in test_channels]},
                ":et": {"S": end_time[iii]},
                ":ed": {"S": ""},
                ":fr": {"L": [{"N": str(i)} for i in frequency]},
                ":ma": {"N": str(np.round(max_echo_range[iii], 4))},
                ":mi": {"N": str(np.round(min_echo_range, 4))},
                ":nd": {"N": str(num_ping_time_dropna[iii])},
                ":ps": {"S": "PROCESSING_RESAMPLE_AND_WRITE_TO_ZARR_STORE"},
                ":pt": {"S": "2023-10-02T08:08:08Z"},
                ":se": {"S": sensor_name},
                ":sh": {"S": ship_name},
                ":st": {"S": start_time[iii]},
                ":zb": {"S": output_bucket_name},
                ":zp": {"S": zarr_path[iii]},
            },
            update_expression=(
                "SET "
                "#CH = :ch, "
                "#ET = :et, "
                "#ED = :ed, "
                "#FR = :fr, "
                "#MA = :ma, "
                "#MI = :mi, "
                "#ND = :nd, "
                "#PS = :ps, "
                "#PT = :pt, "
                "#SE = :se, "
                "#SH = :sh, "
                "#ST = :st, "
                "#ZB = :zb, "
                "#ZP = :zp"
            ),
        )

    # [3] create new zarr store and upload
    create_empty_zarr_store = CreateEmptyZarrStore()
    create_empty_zarr_store.create_cruise_level_zarr_store(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
        tempdir="/tmp",
    )

    assert os.path.exists(f"/tmp/{cruise_name}.zarr")
    # TODO: should actually assert in the bucket
    # mount and verify:
    # shape
    # variables
    #


#######################################################
#######################################################
# april 9th, 2024, last run of all henry bigelow cruises for carrie's presentation
#    cruises = ["HB0802", "HB0803", "HB0805", "HB0807", "HB0901", "HB0902", "HB0904", "HB0905", "HB1002", "HB1006", "HB1102", "HB1103", "HB1105", "HB1201", "HB1206", "HB1301", "HB1303", "HB1304", "HB1401", "HB1403", "HB1405", "HB1501", "HB1502", "HB1503", "HB1506", "HB1507", "HB1601", "HB1603", "HB1604", "HB1701", "HB1702", "HB1801", "HB1802", "HB1803", "HB1804", "HB1805", "HB1806", "HB1901", "HB1902", "HB1903", "HB1906", "HB1907", "HB2001"]
#######################################################
