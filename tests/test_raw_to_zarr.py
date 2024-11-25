import pytest
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer
from water_column_sonar_processing.aws import DynamoDBManager, S3Manager
from water_column_sonar_processing.processing.raw_to_zarr import RawToZarr
from pathlib import Path

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
def test_raw_to_zarr():
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
         filename="./test_resources/D20070724-T042400.raw",
         bucket_name=input_bucket_name,
         key="data/raw/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.raw"
    )
    s3_manager.upload_file( # TODO: this uses resource, try to use client
        filename="./test_resources/D20070724-T042400.bot",
        bucket_name=input_bucket_name,
        key="data/raw/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.bot"
    )
    assert len(s3_manager.list_objects(bucket_name=input_bucket_name, prefix="")) == 2

    # TODO: put stale geojson & zarr store to test deleting
    s3_manager.create_bucket(bucket_name="test_output_bucket")
    assert len(s3_manager.list_buckets()["Buckets"]) == 2

    # s3fs = S3FileSystem(endpoint_url=endpoint_url)
    dynamo_db_manager = DynamoDBManager()# endpoint_url=endpoint_url)

    # ---Create Empty Table--- #
    dynamo_db_manager.create_water_column_sonar_table(table_name=table_name)

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0706"
    sensor_name = "EK60"
    # file_name = "D20070711-T182032.raw"
    #file_name = "D20070720-T224031.raw" # 84 KB
    raw_file_name = "D20070724-T042400.raw"  # 1 MB use this for testing
    bottom_file_name = f"{Path(raw_file_name).stem}.bot"

    # TODO: Test if zarr store already exists

    # TODO: move this into the raw_to_zarr function
    # s3_file_path = f"s3://{input_bucket_name}/data/raw/{ship_name}/{cruise_name}/{sensor_name}/{file_name}"
    s3_file_path = f"data/raw/{ship_name}/{cruise_name}/{sensor_name}/{raw_file_name}"
    s3_bottom_file_path = f"data/raw/{ship_name}/{cruise_name}/{sensor_name}/{bottom_file_name}"
    s3_manager.download_file(bucket_name=input_bucket_name, key=s3_file_path, file_name=raw_file_name)
    s3_manager.download_file(bucket_name=input_bucket_name, key=s3_bottom_file_path, file_name=bottom_file_name)

    raw_to_zarr = RawToZarr()  # endpoint_url=endpoint_url)
    raw_to_zarr.raw_to_zarr(
        table_name=table_name,
        input_bucket_name="test_input_bucket",
        output_bucket_name="test_output_bucket",
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        raw_file_name=raw_file_name
    )

    # TODO: test if zarr store is accessible in the s3 bucket

    # TODO: check the dynamodb dataframe to see if info is updated there

    # #######################################################################
    # self.__upload_files_to_output_bucket(store_name, output_zarr_prefix)
    # #######################################################################
    # self.__update_processing_status(
    #     file_name=input_file_name,
    #     cruise_name=cruise_name,
    #     pipeline_status='SUCCESS_RAW_TO_ZARR'
    # )
    # #######################################################################
    # TODO: remove sns stuff self.__publish_done_message(input_message)

#######################################################
#######################################################
