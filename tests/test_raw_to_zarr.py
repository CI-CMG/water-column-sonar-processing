import pytest
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws
from pathlib import Path

from water_column_sonar_processing.aws import DynamoDBManager, S3Manager
from water_column_sonar_processing.processing import RawToZarr


# TEMPDIR = "/tmp"
# test_bucket = "mybucket"
ip_address = "127.0.0.1"
port = 5555
endpoint_url = f"http://{ip_address}:{port}"
table_name = "test_table"


#######################################################
# def setup_module():
#     print("setup")
#     env_file = find_dotenv(".env-test")
#     load_dotenv(dotenv_path=env_file, override=True)
#
#
# def teardown_module():
#     print("teardown")
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)

def teardown_module():
    print("teardown")

@pytest.fixture
def raw_to_zarr_test_path(test_path):
    return test_path["RAW_TO_ZARR_TEST_PATH"]


# @pytest.fixture(scope="module")
# def s3_base():
#     s3_server = ThreadedMotoServer(ip_address=ip_address, port=port)
#     s3_server.start()
#     yield
#     s3_server.stop()


#######################################################
#######################################################
# Test data with less than 4 points, only has 2
# ship_name = "Henry_B._Bigelow"
# cruise_name = "HB0706"
# sensor_name = "EK60"
# file_name = "D20070720-T224031.raw" # 84 KB

#######################################################
# @mock_aws(config={"core": {"service_whitelist": ["dynamodb", "s3"]}})
# @mock_aws(config={"core": {"service_whitelist": ["dynamodb"]}})
@mock_aws
def test_raw_to_zarr(raw_to_zarr_test_path):
    #def test_raw_to_zarr(s3_base):
    s3_manager = S3Manager()#endpoint_url=endpoint_url)
    s3_manager.list_buckets()
    # s3_client = s3_session.client(service_name="s3", endpoint_url=f"http://{ip_address}:{port}")
    # s3_client.list_buckets()
    # s3_manager = S3Manager()# input_endpoint_url=f"http://{ip_address}:{port}", output_endpoint_url=f"http://{ip_address}:{port}")
    input_bucket_name = "test_input_bucket"
    output_bucket_name = "test_output_bucket"
    s3_manager.create_bucket(bucket_name=input_bucket_name)
    s3_manager.create_bucket(bucket_name=output_bucket_name)
    # TODO: put objects in the output bucket so they can be deleted
    s3_manager.list_buckets()
    s3_manager.upload_file( # TODO: upload to correct bucket
         filename=raw_to_zarr_test_path.joinpath("D20070724-T042400.raw"), # "./test_resources/D20070724-T042400.raw",
         bucket_name=input_bucket_name,
         key="data/raw/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.raw"
    )
    s3_manager.upload_file( # TODO: this uses resource, try to use client
        filename=raw_to_zarr_test_path.joinpath("D20070724-T042400.bot"), # "test_resources/raw_to_zarr/D20070724-T042400.bot",
        bucket_name=input_bucket_name,
        key="data/raw/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.bot"
    )
    assert len(s3_manager.list_objects(bucket_name=input_bucket_name, prefix="")) == 2

    # TODO: put stale geojson & zarr store to test deleting
    s3_manager.create_bucket(bucket_name=output_bucket_name)
    s3_manager.upload_file(
        filename=raw_to_zarr_test_path.joinpath("D20070724-T042400.json"),
        bucket_name=output_bucket_name,
        key="spatial/geojson/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.json"
    )
    # TODO: put zarr store there to delete beforehand # TODO: Test if zarr store already exists
    s3_manager.upload_file(
        filename=raw_to_zarr_test_path.joinpath("D20070724-T042400.zarr/.zmetadata"),
        bucket_name=output_bucket_name,
        key="level_1/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.zarr/.zmetadata"
    )
    s3_manager.upload_file(
        filename=raw_to_zarr_test_path.joinpath("D20070724-T042400.zarr/.zattrs"),
        bucket_name=output_bucket_name,
        key="level_1/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.zarr/.zattrs"
    )
    assert len(s3_manager.list_objects(bucket_name=output_bucket_name, prefix="")) > 1

    assert len(s3_manager.list_buckets()["Buckets"]) == 2

    dynamo_db_manager = DynamoDBManager()

    # ---Create Empty Table--- #
    dynamo_db_manager.create_water_column_sonar_table(table_name=table_name)

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0706"
    sensor_name = "EK60"
    # file_name = "D20070711-T182032.raw"
    #file_name = "D20070720-T224031.raw" # 84 KB
    raw_file_name = "D20070724-T042400.raw"  # 1 MB use this for testing
    # bottom_file_name = f"{Path(raw_file_name).stem}.bot"

    # TODO: move this into the raw_to_zarr function
    # s3_file_path = f"data/raw/{ship_name}/{cruise_name}/{sensor_name}/{raw_file_name}"
    # s3_bottom_file_path = f"data/raw/{ship_name}/{cruise_name}/{sensor_name}/{bottom_file_name}"
    # s3_manager.download_file(bucket_name=input_bucket_name, key=s3_file_path, file_name=raw_file_name)
    # s3_manager.download_file(bucket_name=input_bucket_name, key=s3_bottom_file_path, file_name=bottom_file_name)

    raw_to_zarr = RawToZarr()
    raw_to_zarr.raw_to_zarr(
        table_name=table_name,
        input_bucket_name=input_bucket_name,
        output_bucket_name=output_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        raw_file_name=raw_file_name
    )

    # TODO: test if zarr store is accessible in the s3 bucket
    number_of_files = s3_manager.list_objects(bucket_name=output_bucket_name, prefix=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/")
    # Ensure that all the files were uploaded properly
    assert len(number_of_files) == 72

    # TODO: check the dynamodb dataframe to see if info is updated there
    # ---Verify Data is Populated in Table--- #
    df_after = dynamo_db_manager.get_table_as_df(
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        table_name=table_name,
    )
    print(df_after)
    assert df_after.shape == (1, 15)

    # #######################################################################
    # self.__update_processing_status(
    #     file_name=input_file_name,
    #     cruise_name=cruise_name,
    #     pipeline_status='SUCCESS_RAW_TO_ZARR'
    # )
    # #######################################################################

#######################################################
#######################################################
