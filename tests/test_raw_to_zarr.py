import os

import numcodecs
import numpy as np
import xarray as xr
import zarr
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.aws.s3_manager import S3Manager
from water_column_sonar_processing.model.zarr_manager import ZarrManager

# TEMPDIR = "/tmp"


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


#######################################################
def teardown_module():
    print("teardown")


#######################################################
def test_raw_to_zarr():
    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB0707"
    sensor_name = "EK60"
    file_name = "D20070711-T182032.raw"
    # SET UP S3 CLIENT
    # SET UP S3FS
    # CREATE DYNAMODB
    # CREATE INPUT BUCKET
    # CREATE OUTPUT BUCKET
    # Update_processing_status PROCESSING_RAW_TO_ZARR in DynamoDB
    # Test if zarr store already exists
    # TODO: try without downloading file self.__s3.download_file(bucket_name=self.__input_bucket, key=bucket_key, file_name=input_file_name)
    # self.__create_local_zarr_store(
    #     raw_file_name=input_file_name,
    #     cruise_name=cruise_name,
    #     sensor_name=sensor_name,
    #     output_zarr_prefix=output_zarr_prefix,
    #     store_name=store_name
    # )
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
