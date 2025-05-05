import gc

# import hvplot.pandas
# import hvplot.xarray
import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

from water_column_sonar_processing.aws import DynamoDBManager, S3Manager
from water_column_sonar_processing.cruise import CreateEmptyZarrStore, ResampleRegrid
from water_column_sonar_processing.model import ZarrManager
from water_column_sonar_processing.processing import RawToZarr


#######################################################
def setup_module():
    print("setup")
    # env_file = find_dotenv(".env-prod")  # functional test
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
def resample_regrid_test_path(test_path):
    return test_path["RESAMPLE_REGRID_TEST_PATH"]


#######################################################

### Test Interpolation ###
# @mock_aws
# @pytest.mark.skip(reason="no way of currently testing resample regrid")
# def test_resample_regrid_functional(resample_regrid_test_path):
#     # TODO:
#     #  need to create output zarr store
#     #  need to create geojson
#     #  need to populate dynamodb
#
#     # Opens s3 input model store as xr and writes dataset to output model store
#
#     # HB0706 - 53 files
#     # bucket_name = 'noaa-wcsd-model-pds'
#     ship_name = "Henry_B._Bigelow"
#     cruise_name = "HB0707"
#     sensor_name = "EK60"
#     table_name = "prod-echofish"
#
#     # TODO: create zarr store & write to s3 bucket
#
#     resample_regrid = ResampleRegrid()
#     resample_regrid.resample_regrid(
#         ship_name=ship_name,
#         cruise_name=cruise_name,
#         sensor_name=sensor_name,
#         table_name=table_name,
#     )


@mock_aws
def test_resample_regrid(resample_regrid_test_path, moto_server):
    # TODO: HB0707 isn't good _enough_ test because the MIN_ECHO_RANGE doesn't change

    # TODO: set up db w all 12 dataset file info for HB0707,
    # iterate through 2 files ("D20070712-T100505.raw", "D20070712-T152416.raw") and do resample/regrid
    # verify the output is gridded as expected

    dynamo_db_manager = DynamoDBManager()
    s3_manager = S3Manager(endpoint_url=moto_server)

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0707"  # HB0707 (12 files)
    sensor_name = "EK60"
    table_name = "water-column-sonar-table"

    # [0] create bucket with test files
    l0_test_bucket_name = "l0_test_bucket"
    l1_l2_test_bucket_name = "l1_l2_test_input_bucket"

    s3_manager.create_bucket(bucket_name=l0_test_bucket_name)
    s3_manager.create_bucket(bucket_name=l1_l2_test_bucket_name)
    print(s3_manager.list_buckets())

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
    file_names = [
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
    max_echo_range = [  # does not account for water_level
        249.792,
        249.792,
        249.792,
        249.792,
        249.792,
        249.792,
        249.792,  # used for test
        999.744,  # note: different depth, resample should be [0.19 to 1001.744]
        249.792,
        249.792,
        249.792,
        249.792,
    ]
    min_echo_range = 0.20  # does not account for water_level
    num_ping_time_dropna = [
        9779,
        9743,
        9781,
        9776,
        9734,
        3208,
        7705,  # should be 7705
        4869,  #
        9567,
        159,
        9761,
        5839,
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
    water_level = [  # TODO: create synthetic values?
        0.0,  # 1.0,
        0.0,  # 2.0,
        0.0,  # 3.0,
        0.0,  # 4.0,
        0.0,  # 5.0,
        0.0,  # 4.0,
        0.0,  # 3.0,  #
        0.0,  # 2.0,  #
        0.0,  # 1.0,
        0.0,  # 0.0,
        0.0,  # 1.0,
        0.0,  # 2.0,
    ]
    for iii in range(0, len(file_names)):
        dynamo_db_manager.update_item(
            table_name=table_name,
            key={
                "FILE_NAME": {"S": file_names[iii]},  # Partition Key
                "CRUISE_NAME": {"S": cruise_name},  # Sort Key
            },
            expression_attribute_names={
                "#CH": "CHANNELS",
                "#ET": "END_TIME",
                "#FR": "FREQUENCIES",
                "#MA": "MAX_ECHO_RANGE",
                "#MI": "MIN_ECHO_RANGE",
                "#ND": "NUM_PING_TIME_DROPNA",
                "#PT": "PIPELINE_TIME",
                "#SE": "SENSOR_NAME",
                "#SH": "SHIP_NAME",
                "#ST": "START_TIME",
                "#WL": "WATER_LEVEL",
            },
            expression_attribute_values={
                ":ch": {"L": [{"S": i} for i in test_channels]},
                ":et": {"S": end_time[iii]},
                ":fr": {"L": [{"N": str(int(i))} for i in frequency]},
                ":ma": {"N": str(np.round(max_echo_range[iii], 4))},
                ":mi": {"N": str(np.round(min_echo_range, 4))},
                ":nd": {"N": str(num_ping_time_dropna[iii])},
                ":pt": {"S": "2023-10-02T08:08:08Z"},
                ":se": {"S": sensor_name},
                ":sh": {"S": ship_name},
                ":st": {"S": start_time[iii]},
                ":wl": {"N": str(np.round(water_level[iii], 2))},
            },
            update_expression=(
                "SET "
                "#CH = :ch, "
                "#ET = :et, "
                "#FR = :fr, "
                "#MA = :ma, "
                "#MI = :mi, "
                "#ND = :nd, "
                "#PT = :pt, "
                "#SE = :se, "
                "#SH = :sh, "
                "#ST = :st, "
                "#WL = :wl"
            ),
        )

    # [3] create new zarr store and upload
    create_empty_zarr_store = CreateEmptyZarrStore()
    # TODO: this is out of order, needs to happen after raw-to-zarr L0-to-L1
    create_empty_zarr_store.create_cruise_level_zarr_store(
        output_bucket_name=l1_l2_test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
        # tempdir="/tmp", # TODO: create better tmp directory for testing
    )

    # Assert dataset is in the bucket
    # 'level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.model/tmp/HB0707.zarr/.zattrs'
    assert (
        len(
            s3_manager.list_objects(
                bucket_name=l1_l2_test_bucket_name,
                prefix="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/",
            )
        )
        > 1
    )
    assert (
        "level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zmetadata"
        in s3_manager.list_objects(
            bucket_name=l1_l2_test_bucket_name,
            prefix="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/",
        )
    )
    # mount and verify:
    # s3fs_manager = S3FSManager(endpoint_url=moto_server)
    # s3_path = f"{output_bucket_name}/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr"
    # zarr_store = s3fs_manager.s3_map(s3_path)

    # TODO: PROBLEM NEED TO DO RAW-TO-ZARR CONVERSIONS FOR FILES SO THAT I CAN
    #  USE THE FILE-LEVEL ZARR STORES AS INPUTS.
    s3_manager.upload_file(
        filename=resample_regrid_test_path.joinpath("D20070712-T124906.raw"),
        bucket_name=l0_test_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB0707/EK60/D20070712-T124906.raw",
    )
    s3_manager.upload_file(
        filename=resample_regrid_test_path.joinpath("D20070712-T124906.bot"),
        bucket_name=l0_test_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB0707/EK60/D20070712-T124906.bot",
    )

    s3_manager.upload_file(
        filename=resample_regrid_test_path.joinpath("D20070712-T152416.raw"),
        bucket_name=l0_test_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB0707/EK60/D20070712-T152416.raw",
    )
    s3_manager.upload_file(
        filename=resample_regrid_test_path.joinpath("D20070712-T152416.bot"),
        bucket_name=l0_test_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB0707/EK60/D20070712-T152416.bot",
    )
    assert len(s3_manager.list_objects(bucket_name=l0_test_bucket_name, prefix="")) == 4

    raw_to_zarr = RawToZarr()
    gc.collect()
    # mock the returned water_level to override 0's, was working but didn't work for writing to zarr
    # thing = xarray.Dataset
    # thing.water_level = Mock()
    # thing.water_level.values = 3.0

    raw_to_zarr.raw_to_zarr(
        table_name=table_name,
        input_bucket_name=l0_test_bucket_name,
        output_bucket_name=l1_l2_test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        raw_file_name="D20070712-T124906.raw",
        endpoint_url=moto_server,
        include_bot=True,  # added bot
    )
    raw_to_zarr.raw_to_zarr(
        table_name=table_name,
        input_bucket_name=l0_test_bucket_name,
        output_bucket_name=l1_l2_test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        raw_file_name="D20070712-T152416.raw",
        endpoint_url=moto_server,
        include_bot=True,  # added bot
    )
    gc.collect()
    number_of_files_xx = s3_manager.list_objects(
        bucket_name=l1_l2_test_bucket_name,
        prefix=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/",
    )
    assert len(number_of_files_xx) > 72  # 1402

    resample_regrid = ResampleRegrid()
    resample_regrid.resample_regrid(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
        bucket_name=l1_l2_test_bucket_name,
        # TODO: this needs to be passed for each respective file, TEST ONLY TWO
        override_select_files=["D20070712-T124906.raw"],
        endpoint_url=moto_server,
    )

    resample_regrid.resample_regrid(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
        bucket_name=l1_l2_test_bucket_name,
        # TODO: this needs to be passed for each respective file, TEST ONLY TWO
        override_select_files=["D20070712-T152416.raw"],
        endpoint_url=moto_server,
    )
    # TODO: verify that the two files in question were properly resampled and regridded
    # check a couple of samples that are adjacent to one another
    # assert 2 > 1

    test_zarr_manager = ZarrManager()
    test_output_zarr_store = test_zarr_manager.open_l2_zarr_store_with_xarray(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        bucket_name=l1_l2_test_bucket_name,
        endpoint_url=moto_server,
    )
    print(test_output_zarr_store.Sv)

    # because we only processed two files, there should be missing values
    assert np.isnan(np.sum(test_output_zarr_store.latitude.values))

    start_time = np.datetime64("2007-07-12T12:49:06.313")
    end_time = np.datetime64("2007-07-12T17:18:03.032")
    select_times = (test_output_zarr_store.time > start_time) & (
        test_output_zarr_store.time < end_time
    )
    # TODO: check the range of depths for the output data
    # check selected timestamps and verify all latitude/longitude/times are updated
    # test_output_zarr_store.latitude.sel(time=slice('2007-07-12T12:49:06.313Z', '2007-07-12T17:18:03.032Z')).values
    assert not np.isnan(
        np.sum(
            test_output_zarr_store.where(cond=select_times, drop=True).latitude.values
        )
    )
    assert not np.isnan(
        np.sum(
            test_output_zarr_store.where(cond=select_times, drop=True).longitude.values
        )
    )

    # TODO: times are initialized to '1970-01-01T00:00:00.000000000', need way to check updates
    #  maybe monotonic increasing
    # assert not np.isnan(np.sum(test_output_zarr_store.where(cond=(select_times), drop=True).time.values))

    # TODO: assert that the test_output_zarr_store.Sv at specific depth equals the input files

    # TODO: check the bottom values were written correctly
    assert np.nanmax(test_output_zarr_store.bottom.values) == pytest.approx(
        970.11835
    )  # uses 18 kHz only
    assert np.nanmin(test_output_zarr_store.bottom.values) == pytest.approx(15.936)


@mock_aws
def test_resample_regrid_water_level(resample_regrid_test_path, moto_server):
    # Convert file and ensure that water_level offset is captured
    # Need to test:
    #   https://noaa-wcsd-pds.s3.amazonaws.com/data/raw/Henry_B._Bigelow/HB1906/EK60/D20191106-T001906.raw
    #   d20191106_t001906-t115734_Zsc-DWBA-Schools_All-RegionDefs.evr

    dynamo_db_manager = DynamoDBManager()
    s3_manager = S3Manager(endpoint_url=moto_server)

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB1906"
    sensor_name = "EK60"
    table_name = "water-column-sonar-table"

    # create dynamodb table
    dynamo_db_manager.create_water_column_sonar_table(table_name=table_name)

    # create bucket with test files
    l0_test_bucket_name = "l0_test_bucket"
    l1_l2_test_bucket_name = "l1_l2_test_input_bucket"
    s3_manager.create_bucket(bucket_name=l0_test_bucket_name)
    s3_manager.create_bucket(bucket_name=l1_l2_test_bucket_name)
    print(s3_manager.list_buckets())

    # D20191106-T001906.raw
    # s3_manager.upload_file(
    #     filename=resample_regrid_test_path.joinpath("D20191106-T001906.raw"),
    #     bucket_name=l0_test_bucket_name,
    #     key="dataset/raw/Henry_B._Bigelow/HB1906/EK60/D20191106-T001906.raw",
    # )
    # s3_manager.upload_file(
    #     filename=resample_regrid_test_path.joinpath("D20191106-T001906.bot"),
    #     bucket_name=l0_test_bucket_name,
    #     key="dataset/raw/Henry_B._Bigelow/HB1906/EK60/D20191106-T001906.bot",
    # )

    s3_manager.upload_file(
        filename=resample_regrid_test_path.joinpath("D20191106-T034434.raw"),
        bucket_name=l0_test_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB1906/EK60/D20191106-T034434.raw",
    )
    s3_manager.upload_file(
        filename=resample_regrid_test_path.joinpath("D20191106-T034434.bot"),
        bucket_name=l0_test_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB1906/EK60/D20191106-T034434.bot",
    )

    s3_manager.upload_file(
        filename=resample_regrid_test_path.joinpath("D20191106-T042540.raw"),
        bucket_name=l0_test_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB1906/EK60/D20191106-T042540.raw",
    )
    s3_manager.upload_file(
        filename=resample_regrid_test_path.joinpath("D20191106-T042540.bot"),
        bucket_name=l0_test_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB1906/EK60/D20191106-T042540.bot",
    )

    assert len(s3_manager.list_objects(bucket_name=l0_test_bucket_name, prefix="")) > 2

    raw_to_zarr = RawToZarr()
    gc.collect()
    raw_to_zarr.raw_to_zarr(
        table_name=table_name,
        input_bucket_name=l0_test_bucket_name,
        output_bucket_name=l1_l2_test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        raw_file_name="D20191106-T034434.raw",
        endpoint_url=moto_server,
        include_bot=True,
    )
    raw_to_zarr.raw_to_zarr(
        table_name=table_name,
        input_bucket_name=l0_test_bucket_name,
        output_bucket_name=l1_l2_test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        raw_file_name="D20191106-T042540.raw",
        endpoint_url=moto_server,
        include_bot=True,
    )
    gc.collect()

    cruise_df_before = dynamo_db_manager.get_table_as_df(
        cruise_name=cruise_name,
        table_name=table_name,
    )
    print(cruise_df_before)

    # create new zarr store and upload
    create_empty_zarr_store = CreateEmptyZarrStore()
    # TODO: this is out of order, needs to happen after raw-to-zarr L0-to-L1
    create_empty_zarr_store.create_cruise_level_zarr_store(
        output_bucket_name=l1_l2_test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
        # tempdir="/tmp", # TODO: create better tmp directory for testing
    )

    # Assert dataset is in the bucket
    assert (
        len(
            s3_manager.list_objects(
                bucket_name=l1_l2_test_bucket_name,
                prefix="level_2/Henry_B._Bigelow/HB1906/EK60/HB1906.zarr/",
            )
        )
        > 1
    )
    assert (
        "level_2/Henry_B._Bigelow/HB1906/EK60/HB1906.zarr/.zmetadata"
        in s3_manager.list_objects(
            bucket_name=l1_l2_test_bucket_name,
            prefix="level_2/Henry_B._Bigelow/HB1906/EK60/HB1906.zarr/",
        )
    )

    number_of_files_xx = s3_manager.list_objects(
        bucket_name=l1_l2_test_bucket_name,
        prefix=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/",
    )
    assert len(number_of_files_xx) > 100  # 912

    cruise_df_l0_l1 = dynamo_db_manager.get_table_as_df(
        cruise_name=cruise_name,
        table_name=table_name,
    )
    print(cruise_df_l0_l1)

    resample_regrid = ResampleRegrid()
    resample_regrid.resample_regrid(  # water_level == 7.5
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
        bucket_name=l1_l2_test_bucket_name,
        # TODO: this needs to be passed for each respective file, TEST ONLY TWO
        override_select_files=["D20191106-T034434.raw"],
        endpoint_url=moto_server,
    )
    resample_regrid.resample_regrid(  # water_level == 7.5
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
        bucket_name=l1_l2_test_bucket_name,
        # TODO: this needs to be passed for each respective file, TEST ONLY TWO
        override_select_files=["D20191106-T042540.raw"],
        endpoint_url=moto_server,
    )

    test_zarr_manager = ZarrManager()
    test_output_zarr_store = test_zarr_manager.open_l2_zarr_store_with_xarray(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        bucket_name=l1_l2_test_bucket_name,
        endpoint_url=moto_server,
    )
    assert np.isclose(test_output_zarr_store.Sv.depth[0].values, 0.0)
    assert np.isclose(test_output_zarr_store.Sv.depth[-1].values, 507.4)
    assert len(test_output_zarr_store.Sv.depth) == 2538
    # cruise_select = test_output_zarr_store.sel(
    #     time=slice(
    #         "2019-11-06T04:20:00", "2019-11-06T04:30:00"
    #     )  # 2019-11-06T00:19:08.052599040, 2019-11-06T01:00:15.752830976
    # )

    # To plot for diagnostics
    # Sv38 = cruise_select.sel(frequency=38_000).Sv.hvplot().opts(invert_yaxis=True)
    # hvplot.show(Sv38)  # it starts at 215.2 m and ends at 217 to 218.2 m

    select_in_noise = test_output_zarr_store.sel(
        time=slice("2019-11-06T04:24:44", "2019-11-06T04:25:14"),
        frequency=38_000,
        depth=slice(215, 218),  # slice across ctd sidescan
    )  # noise inside is ~-53.13 dB
    assert np.isclose(int(np.nanmean(select_in_noise.Sv)), -53)

    select_outside_noise = test_output_zarr_store.sel(
        time=slice("2019-11-06T04:24:44", "2019-11-06T04:25:14"),
        frequency=38_000,
        depth=slice(212, 215),  # slice across ctd sidescan
    )  # noise outside is ~-80.59 dB
    assert np.isclose(int(np.nanmean(select_outside_noise.Sv)), -80)


@mock_aws
@pytest.mark.skip(reason="TODO: implement this")
def test_interpolate(resample_regrid_test_path):
    # Get two raw files with extreme range differences between the two,
    # generate zarr stores,
    # get the last part of the first file and first part of the second file
    # and write out to new zarr stores ...save in test resources
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
