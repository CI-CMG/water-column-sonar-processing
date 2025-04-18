import os
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.aws import S3Manager, chunked

# from water_column_sonar_processing.aws.s3_manager import S3Manager, chunked

input_bucket_name = "example_input_bucket"
output_bucket_name = "example_output_bucket"


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


@pytest.fixture
def s3_manager_test_path(test_path):
    return test_path["S3_MANAGER_TEST_PATH"]


#######################################################
def test_create_file(tmp_path):
    content = "file_content"
    # d = tmp_path / "sub"
    # d.mkdir()
    # tmp_path.mkdir()
    # print(d)
    p = tmp_path / "hello.txt"
    p.write_text(content, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == content
    assert len(list(tmp_path.iterdir())) == 1
    # assert 0


# TODO: fix problem where this is creating remaining files
@mock_aws
def test_s3_manager(tmp_path):
    # test-input-bucket
    test_bucket_name = os.environ.get("INPUT_BUCKET_NAME")

    # --- set up initial resources --- #
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())

    # --- tests the src --- #
    # TODO: create tmp directory with tmp file and upload that
    s3_manager.put(bucket_name=test_bucket_name, key="the_key", body="the_body")
    s3_manager.list_objects(bucket_name=test_bucket_name, prefix="")

    s3_object = s3_manager.get_object(bucket_name=test_bucket_name, key_name="the_key")

    body = s3_object["Body"].read().decode("utf-8")

    assert body == "the_body"

    all_buckets = s3_manager.list_buckets()
    print(all_buckets)

    file_path = tmp_path / "the_file.txt"
    s3_manager.download_file(
        bucket_name=test_bucket_name, key="the_key", file_name=file_path
    )

    assert len(list(tmp_path.iterdir())) == 1


#######################################################
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
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name="test_bucket_123")
    s3_manager.create_bucket(bucket_name="test_bucket_456")

    assert "test_bucket_123" in [
        i["Name"] for i in s3_manager.list_buckets()["Buckets"]
    ]
    assert "test_bucket_456" in [
        i["Name"] for i in s3_manager.list_buckets()["Buckets"]
    ]


@pytest.mark.skip(reason="TODO: implement this")
@mock_aws
def test_upload_files_with_thread_pool_executor():
    pass


@mock_aws
def test_list_objects(s3_manager_test_path):
    s3_manager = S3Manager()

    test_bucket_name = "test_bucket"

    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())

    s3_manager.upload_file(
        filename=s3_manager_test_path.joinpath("HB0707.zarr/.zmetadata"),
        bucket_name=test_bucket_name,
        key="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zmetadata",
    )
    s3_manager.upload_file(
        filename=s3_manager_test_path.joinpath("HB0707.zarr/.zattrs"),
        bucket_name=test_bucket_name,
        key="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zattrs",
    )

    assert len(s3_manager.list_objects(bucket_name=test_bucket_name, prefix="")) == 2

    list_of_found_objects = s3_manager.list_objects(
        bucket_name=test_bucket_name,
        prefix="level_2/Henry_B._Bigelow/HB0707/",
    )

    assert len(list_of_found_objects) == 2
    assert (
        "level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zattrs"
        in list_of_found_objects
    )
    assert (
        "level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zmetadata"
        in list_of_found_objects
    )


@mock_aws
def test_get_child_objects(s3_manager_test_path):
    s3_manager = S3Manager()

    test_bucket_name = "test_bucket"

    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())

    s3_manager.upload_file(
        filename=s3_manager_test_path.joinpath("HB0707.zarr/.zmetadata"),
        bucket_name=test_bucket_name,
        key="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zmetadata",
    )
    s3_manager.upload_file(
        filename=s3_manager_test_path.joinpath("HB0707.zarr/.zattrs"),
        bucket_name=test_bucket_name,
        key="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zattrs",
    )

    assert len(s3_manager.list_objects(bucket_name=test_bucket_name, prefix="")) == 2

    found_objects = s3_manager.get_child_objects(
        bucket_name=test_bucket_name,
        sub_prefix="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr",
        # file_suffix: str = None,
    )

    assert len(found_objects) > 0


@mock_aws
def test_download_file(s3_manager_test_path, tmp_path):
    s3_manager = S3Manager()

    test_bucket_name = "test_bucket"

    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())

    body = open(s3_manager_test_path.joinpath("HB0707.zarr/.zmetadata"), "r").read()
    s3_manager.put(
        bucket_name=test_bucket_name,
        key="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zmetadata",
        body=body,
    )

    assert len(s3_manager.list_objects(bucket_name=test_bucket_name, prefix="")) == 1

    s3_manager.download_file(
        bucket_name=test_bucket_name,
        key="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zmetadata",
        file_name=tmp_path / ".zmetadata",
    )

    assert Path(tmp_path / ".zmetadata").exists()


@pytest.mark.skip(reason="TODO: implement this")
@mock_aws
def test_delete_object():
    pass


@pytest.mark.skip(reason="TODO: implement this")
@mock_aws
def test_delete_objects():
    pass


@mock_aws
def test_put(s3_manager_test_path):
    s3_manager = S3Manager()

    # [0] create bucket with test files
    test_bucket_name = "test_bucket"

    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())

    body = open(s3_manager_test_path.joinpath("HB0707.zarr/.zmetadata"), "r").read()
    s3_manager.put(
        bucket_name=test_bucket_name,
        key="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/.zmetadata",
        body=body,
    )

    assert len(s3_manager.list_objects(bucket_name=test_bucket_name, prefix="")) == 1


@mock_aws
def test_get(s3_manager_test_path):
    s3_manager = S3Manager()

    test_bucket_name = "test_bucket"

    s3_manager.create_bucket(bucket_name=test_bucket_name)

    print(s3_manager.list_buckets())

    body = open(s3_manager_test_path.joinpath("HB0707.zarr/foo"), "r").read()
    s3_manager.put(
        bucket_name=test_bucket_name,
        key="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/foo",
        body=body,
    )

    assert len(s3_manager.list_objects(bucket_name=test_bucket_name, prefix="")) == 1

    metadata = s3_manager.get_object(
        bucket_name=test_bucket_name,
        key_name="level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/foo",
    )

    metadata_body = metadata["Body"].read().decode("utf-8")

    assert metadata_body == "123"


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
