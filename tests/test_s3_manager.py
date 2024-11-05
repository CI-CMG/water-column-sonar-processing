import os

from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.aws.s3_manager import S3Manager, chunked

input_bucket_name = "example_input_bucket"
output_bucket_name = "example_output_bucket"


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


#######################################################
@mock_aws
def test_s3_manager():
    # test-input-bucket
    test_bucket_name = os.environ.get("INPUT_BUCKET_NAME")

    # --- set up initial resources --- #
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())
    # s3.create_bucket(bucket_name=test_bucket_name)

    # --- tests the src --- #
    s3_manager.put(bucket_name=test_bucket_name, key="the_key", body="the_body")

    s3_object = s3_manager.get(bucket_name=test_bucket_name, key="the_key")

    body = s3_object.get()["Body"].read().decode("utf-8")

    assert body == "the_body"

    all_buckets = s3_manager.list_buckets()
    print(all_buckets)


#######################################################
# TODO: Tests
#######################################################
# tests chunked
def test_chunked():
    objects_to_process = [1, 2, 3, 4]
    for batch in chunked(ll=objects_to_process, n=2):
        assert len(batch) == 2


#######################################################
@mock_aws
def test_create_bucket():
    test_bucket_name = os.environ.get("INPUT_BUCKET_NAME")

    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name="test123")
    s3_manager.create_bucket(bucket_name=test_bucket_name)
    s3_manager.create_bucket(bucket_name="test456")

    assert len(s3_manager.list_buckets()["Buckets"]) == 3
    assert "test-input-bucket" in [
        i["Name"] for i in s3_manager.list_buckets()["Buckets"]
    ]


@mock_aws
def test_list_buckets():
    pass


@mock_aws
def test_upload_files_with_thread_pool_executor():
    pass


@mock_aws
def test_list_objects():
    pass


@mock_aws
def test_get_child_objects():
    pass


@mock_aws
def test_download_file():
    pass


@mock_aws
def test_delete_object():
    pass


@mock_aws
def test_delete_objects():
    pass


@mock_aws
def test_put():
    pass


@mock_aws
def test_get():
    pass


#######################################################
# create_bucket
# upload_file(s)_with_thread_pool_executor() change name
#   upload 6 files
# list_objects
# download_object one object
# delete_object
#   one obj
# delete_objects in batches
#######################################################
