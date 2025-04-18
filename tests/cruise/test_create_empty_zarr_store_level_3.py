import numpy as np
import pytest
import xarray as xr
import zarr
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws
from moto.server import ThreadedMotoServer

from water_column_sonar_processing.aws import DynamoDBManager, S3FSManager, S3Manager
from water_column_sonar_processing.cruise import CreateEmptyZarrStoreLevel3


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
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def create_empty_zarr_test_path(test_path):
    return test_path["CREATE_EMPTY_ZARR_TEST_PATH"]


#######################################################
# TODO: this works but should maybe follow approach where i resample from Level 2 data?
@mock_aws()
# @pytest.mark.skip(reason="For future implementations")
def test_create_empty_zarr_store_level_3(create_empty_zarr_test_path, moto_server):
    dynamo_db_manager = DynamoDBManager()
    s3_manager = S3Manager(endpoint_url=moto_server)

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0707"  # HB0706 (53 files), HB0707 (12 files)
    sensor_name = "EK60"
    table_name = "water-column-sonar-table"

    # [0] create bucket with test files
    input_bucket_name = "test_input_bucket"
    output_bucket_name = "test_output_bucket"

    s3_manager.create_bucket(bucket_name=input_bucket_name)
    s3_manager.create_bucket(bucket_name=output_bucket_name)
    print(s3_manager.list_buckets())

    # Put dummy zarr store there to delete beforehand to test delete
    s3_manager.upload_file(
        filename=create_empty_zarr_test_path.joinpath("HB0707.zarr/.zmetadata"),
        bucket_name=output_bucket_name,
        key="level_3/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zmetadata",
    )
    s3_manager.upload_file(
        filename=create_empty_zarr_test_path.joinpath("HB0707.zarr/.zattrs"),
        bucket_name=output_bucket_name,
        key="level_3/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zattrs",
    )
    assert len(s3_manager.list_objects(bucket_name=output_bucket_name, prefix="")) > 1

    # [1] create dynamodb table
    dynamo_db_manager.create_water_column_sonar_table(table_name=table_name)

    # [2] bootstrap w/ test dataset
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
        999.744,  # note: different depth
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
    water_level = 0.0

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
                # "#ED": "ERROR_DETAIL",
                "#FR": "FREQUENCIES",
                "#MA": "MAX_ECHO_RANGE",
                "#MI": "MIN_ECHO_RANGE",
                "#ND": "NUM_PING_TIME_DROPNA",
                # "#PS": "PIPELINE_STATUS",  # testing this updated
                "#PT": "PIPELINE_TIME",  # testing this updated
                "#SE": "SENSOR_NAME",
                "#SH": "SHIP_NAME",
                "#ST": "START_TIME",
                "#WL": "WATER_LEVEL",
            },
            expression_attribute_values={
                ":ch": {"L": [{"S": i} for i in test_channels]},
                ":et": {"S": end_time[iii]},
                # ":ed": {"S": ""},
                ":fr": {"L": [{"N": str(i)} for i in frequency]},
                ":ma": {"N": str(np.round(max_echo_range[iii], 4))},
                ":mi": {"N": str(np.round(min_echo_range, 4))},
                ":nd": {"N": str(num_ping_time_dropna[iii])},
                # ":ps": {"S": "PROCESSING_RESAMPLE_AND_WRITE_TO_ZARR_STORE"},
                ":pt": {"S": "2023-10-02T08:08:08Z"},
                ":se": {"S": sensor_name},
                ":sh": {"S": ship_name},
                ":st": {"S": start_time[iii]},
                ":wl": {"N": str(np.round(water_level, 2))},
            },
            update_expression=(
                "SET "
                "#CH = :ch, "
                "#ET = :et, "
                # "#ED = :ed, "
                "#FR = :fr, "
                "#MA = :ma, "
                "#MI = :mi, "
                "#ND = :nd, "
                # "#PS = :ps, "
                "#PT = :pt, "
                "#SE = :se, "
                "#SH = :sh, "
                "#ST = :st, "
                "#WL = :wl"
            ),
        )

    # [3] create new zarr store and upload
    create_empty_zarr_store_level_3 = CreateEmptyZarrStoreLevel3()
    create_empty_zarr_store_level_3.create_cruise_level_zarr_store_level_3(
        output_bucket_name=output_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
    )

    assert (
        len(
            s3_manager.list_objects(
                bucket_name=output_bucket_name,
                prefix="level_3/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/",
            )
        )
        > 1
    )
    assert (
        "level_3/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zmetadata"
        in s3_manager.list_objects(
            bucket_name=output_bucket_name,
            prefix="level_3/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/",
        )
    )
    # mount and verify:
    s3fs_manager = S3FSManager(endpoint_url=moto_server)
    s3_path = f"{output_bucket_name}/level_3/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr"
    zarr_store = s3fs_manager.s3_map(s3_path)

    # --- Open with Zarr --- #
    root = zarr.open(store=zarr_store, mode="r")
    print(root.info)
    assert root.Sv.shape == (1001, 89911, 4)

    # --- Open with Xarray --- #
    ds = xr.open_dataset(zarr_store, engine="zarr")
    print(ds)
    assert ds.Sv.size == 360003644  # ~0.34 GB ==> 25% of the size
    assert set(list(ds.variables)) == {
        "Sv",
        "bottom",
        "speed",
        "depth",
        "frequency",
        "latitude",
        "longitude",
        "time",
    }
    assert ds.speed.attrs == {
        "long_name": "Nautical miles per hour",
        "standard_name": "speed",
        "units": "Knots",
    }
    assert set(ds.speed.dims) == {"time"}


#######################################################
#######################################################
# april 9th, 2024, last run of all henry bigelow cruises for carrie's presentation
#    cruises = ["HB0802", "HB0803", "HB0805", "HB0807", "HB0901", "HB0902", "HB0904", "HB0905", "HB1002", "HB1006", "HB1102", "HB1103", "HB1105", "HB1201", "HB1206", "HB1301", "HB1303", "HB1304", "HB1401", "HB1403", "HB1405", "HB1501", "HB1502", "HB1503", "HB1506", "HB1507", "HB1601", "HB1603", "HB1604", "HB1701", "HB1702", "HB1801", "HB1802", "HB1803", "HB1804", "HB1805", "HB1806", "HB1901", "HB1902", "HB1903", "HB1906", "HB1907", "HB2001"]
#######################################################
