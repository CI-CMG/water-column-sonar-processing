import boto3
# import numcodecs
# import numpy as np
import pytest
# import xarray as xr
# import zarr
from s3fs import S3FileSystem
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

from tests.test_s3_manager import input_bucket_name
from water_column_sonar_processing.aws import DynamoDBManager
from water_column_sonar_processing.aws.s3_manager import S3Manager
# from water_column_sonar_processing.model.zarr_manager import ZarrManager
from water_column_sonar_processing.processing.raw_to_zarr import RawToZarr

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


@pytest.fixture(scope="module")
def s3_base():
    s3_server = ThreadedMotoServer(ip_address=ip_address, port=port)
    s3_server.start()
    yield
    s3_server.stop()

#######################################################
#######################################################
# Test data with less than 4 points, only has 2
# ship_name = "Henry_B._Bigelow"
# cruise_name = "HB0706"
# sensor_name = "EK60"
# file_name = "D20070720-T224031.raw" # 84 KB

#######################################################
# @mock_aws(config={"core": {"service_whitelist": ["dynamodb", "s3"]}})
# @mock_aws(config={"core": {"service_whitelist": ["dynamodb", "s3"]}})
@mock_aws
def test_raw_to_zarr(s3_base):
    # s3_session = boto3.Session()
    # s3_client = s3_session.client(service_name="s3", endpoint_url=f"http://{ip_address}:{port}")
    # s3_client.list_buckets()
    s3_manager = S3Manager(input_endpoint_url=f"http://{ip_address}:{port}", output_endpoint_url=f"http://{ip_address}:{port}")
    s3_manager.create_bucket(bucket_name="test_input_bucket")
    s3_manager.upload_file(body="./test_resources/D20070724-T042400.bot", bucket="test_input_bucket", key="D20070724-T042400.bot")
    s3_manager.upload_file(
        body="./test_resources/D20070724-T042400.raw",
        bucket="test_input_bucket",
        key="data/raw/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.raw"
   )
    s3_manager.upload_file(
        body="./test_resources/D20070724-T042400.bot",
        bucket="test_input_bucket",
        key="data/raw/Henry_B._Bigelow/HB0706/EK60/D20070724-T042400.bot"
    )
    s3_manager.create_bucket(bucket_name="test_output_bucket")

    # s3fs = S3FileSystem(endpoint_url=endpoint_url)
    dynamo_db_manager = DynamoDBManager()

    # s3_client.create_bucket(Bucket="test_input_bucket")
    # s3_client.create_bucket(Bucket="test_output_bucket")

    # ---Create Table--- #
    # TODO: move create tabel into DynamoDBManager
    dynamo_db_manager.create_table(
        table_name=table_name,
        key_schema=[
            {
                "AttributeName": "FILE_NAME",
                "KeyType": "HASH",
            },
            {
                "AttributeName": "CRUISE_NAME",
                "KeyType": "RANGE",
            },
        ],
        attribute_definitions=[
            {"AttributeName": "FILE_NAME", "AttributeType": "S"},
            {"AttributeName": "CRUISE_NAME", "AttributeType": "S"},
        ],
    )

    ship_name = "Henry_B._Bigelow"
    #cruise_name = "HB0707"
    cruise_name = "HB0706"
    sensor_name = "EK60"
    #file_name = "D20070711-T182032.raw"
    # file_name = "D20070720-T224031.raw" # 84 KB
    file_name = "D20070724-T042400.raw" # 1 MB

    # SET UP S3 CLIENT
    # SET UP S3FS
    # CREATE DYNAMODB [mocked]
    # CREATE INPUT BUCKET [real]
    # CREATE OUTPUT BUCKET [mocked]
    # Update_processing_status PROCESSING_RAW_TO_ZARR in DynamoDB
    # Test if zarr store already exists
    # TODO: try without downloading file self.__s3.download_file(bucket_name=self.__input_bucket, key=bucket_key, file_name=input_file_name)
    raw_to_zarr = RawToZarr()
    raw_to_zarr.raw_to_zarr(ship_name=ship_name, cruise_name=cruise_name, sensor_name=sensor_name, file_name=file_name)
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
