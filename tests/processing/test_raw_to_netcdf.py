from pathlib import Path

import pytest
import xarray as xr
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

from water_column_sonar_processing.aws import DynamoDBManager, S3Manager
from water_column_sonar_processing.processing import RawToNetCDF


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


@pytest.fixture(scope="module")
def moto_server():  # TODO: add "async"
    """Fixture to run a mocked AWS server for testing."""
    # Note: pass `port=0` to get a random free port.
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def raw_to_netcdf_test_path(test_path):
    return test_path["RAW_TO_ZARR_TEST_PATH"]


#######################################################
#######################################################
# Test dataset with less than 4 points, only has 2
# ship_name = "Henry_B._Bigelow"
# cruise_name = "HB0706"
# sensor_name = "EK60"
# file_name = "D20070720-T224031.raw" # 84 KB


#######################################################
@mock_aws
def test_raw_to_netcdf(moto_server, raw_to_netcdf_test_path, tmp_path):
    table_name = "water-column-sonar-table"
    s3_manager = S3Manager(endpoint_url=moto_server)

    input_bucket_name = "test_input_bucket"
    output_bucket_name = "test_output_bucket"

    s3_manager.create_bucket(bucket_name=input_bucket_name)
    s3_manager.create_bucket(bucket_name=output_bucket_name)
    assert len(s3_manager.list_buckets()["Buckets"]) == 2

    s3_manager.upload_file(
        filename=raw_to_netcdf_test_path.joinpath(
            "D20070724-T042400.raw"
        ),  # "./test_resources/D20070724-T042400.raw",
        bucket_name=input_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.raw",
    )
    s3_manager.upload_file(  # TODO: this uses resource, try to use client
        filename=raw_to_netcdf_test_path.joinpath(
            "D20070724-T042400.bot"
        ),  # "test_resources/raw_to_zarr/D20070724-T042400.bot",
        bucket_name=input_bucket_name,
        key="dataset/raw/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.bot",
    )
    assert len(s3_manager.list_objects(bucket_name=input_bucket_name, prefix="")) == 2

    # Put stale geojson to test deleting
    s3_manager.create_bucket(bucket_name=output_bucket_name)
    s3_manager.upload_file(
        filename=raw_to_netcdf_test_path.joinpath("D20070724-T042400.json"),
        bucket_name=output_bucket_name,
        key="spatial/geojson/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.json",
    )
    # Put zarr store there to test does not get deleted
    s3_manager.upload_file(
        filename=raw_to_netcdf_test_path.joinpath("D20070724-T042400.zarr/.zmetadata"),
        bucket_name=output_bucket_name,
        key="level_1/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.zarr/.zmetadata",
    )
    s3_manager.upload_file(
        filename=raw_to_netcdf_test_path.joinpath("D20070724-T042400.zarr/.zattrs"),
        bucket_name=output_bucket_name,
        key="level_1/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.zarr/.zattrs",
    )
    s3_manager.upload_file(  # hack to put something there
        filename=raw_to_netcdf_test_path.joinpath("D20070724-T042400.nc"),
        bucket_name=output_bucket_name,
        key="level_1/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.nc",
    )
    assert len(s3_manager.list_objects(bucket_name=output_bucket_name, prefix="")) == 4

    assert len(s3_manager.list_buckets()["Buckets"]) == 2

    dynamo_db_manager = DynamoDBManager()

    # ---Create Empty Table--- #
    dynamo_db_manager.create_water_column_sonar_table(table_name=table_name)

    #
    # missing bootstrap of dataset here?
    #

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0706"
    sensor_name = "EK60"
    # file_name = "D20070711-T182032.raw"
    # file_name = "D20070720-T224031.raw" # 84 KB
    raw_file_name = "D20070724-T042400.raw"  # 1.5 MB use this for testing
    # bottom_file_name = f"{Path(raw_file_name).stem}.bot"

    # TODO: move this into the raw_to_zarr function
    # s3_file_path = f"dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{raw_file_name}"
    # s3_bottom_file_path = f"dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{bottom_file_name}"
    # s3_manager.download_file(bucket_name=input_bucket_name, key=s3_file_path, file_name=raw_file_name)
    # s3_manager.download_file(bucket_name=input_bucket_name, key=s3_bottom_file_path, file_name=bottom_file_name)

    number_of_files_before = s3_manager.list_objects(
        bucket_name=output_bucket_name,
        prefix=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/",
    )
    assert len(number_of_files_before) == 3

    raw_to_netcdf = RawToNetCDF()
    raw_to_netcdf.raw_to_netcdf(
        table_name=table_name,
        input_bucket_name=input_bucket_name,
        output_bucket_name=output_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        raw_file_name=raw_file_name,
    )  # two files are now uploaded

    # Problem: missing nc file
    number_of_files = s3_manager.list_objects(
        bucket_name=output_bucket_name,
        prefix=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/",
    )
    # Ensure that all the files were uploaded properly, zarr was undisturbed
    assert len(number_of_files) == 4  # with two .nc files

    # TODO: check the dynamodb dataframe to see if info is updated there
    # ---Verify Data is Populated in Table--- #
    # df_after = dynamo_db_manager.get_table_as_df(
    #     # ship_name=ship_name,
    #     cruise_name=cruise_name,
    #     # sensor_name, # TODO: need to add this back for EK80
    #     table_name=table_name,
    # )
    # print(df_after)
    # assert df_after.shape == (1, 13)

    # mount and verify:
    file_stem = Path(raw_file_name).stem
    # s3_path = f"s3://{output_bucket_name}/level_1/{ship_name}/{cruise_name}/{sensor_name}/{file_stem}.nc"

    s3_manager.download_file(  # TODO: delete file
        bucket_name=output_bucket_name,
        key=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/{file_stem}.nc",
        file_name=tmp_path.joinpath(f"{file_stem}.nc"),
        # tmp_path.joinpath(f"{file_stem}.nc")
    )

    # --- Open with Xarray --- #
    ds = xr.open_dataset(
        filename_or_obj=tmp_path.joinpath(f"{file_stem}.nc"), engine="netcdf4"
    )
    assert len(ds.attrs) >= 10

    s3_manager.download_file(
        bucket_name=output_bucket_name,
        key=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/{file_stem}_computed_Sv.nc",
        file_name=tmp_path.joinpath(f"{file_stem}_computed_Sv.nc"),
    )
    ds = xr.open_dataset(
        filename_or_obj=tmp_path.joinpath(f"{file_stem}_computed_Sv.nc"),
        engine="netcdf4",
    )
    print(ds)
    assert ds.Sv.shape == (4, 36, 2604)


#######################################################
#######################################################
