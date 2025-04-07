import numpy as np
import pytest
import xarray as xr
import zarr
from dotenv import find_dotenv, load_dotenv

# from s3fs import S3FileSystem # TODO: shouldn't import this
from moto import mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

from water_column_sonar_processing.aws import S3FSManager, S3Manager

test_bucket = "test_bucket123"
ip_address = "127.0.0.1"
port = 5555
endpoint_url = f"http://{ip_address}:{port}"


# TODO: https://github.com/open-metadata/OpenMetadata/pull/17805#issue-2519402796
# switch from moto to minio


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
    # Note: pass `port=0` to get a random free port.
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def s3fs_manager_test_path(test_path):
    return test_path["S3FS_MANAGER_TEST_PATH"]


#####################################################################
#####################################################################
# @mock_aws
# def test_load_all_files(moto_server, s3fs_manager_test_path):
#     # TODO: get rid of direct s3fs uses
#     s3_session = boto3.Session() # TODO: don't do this primitive like this
#     s3_client = s3_session.client(service_name="s3", endpoint_url=endpoint_url)
#     s3_client.list_buckets()
#
#     # s3fs = S3FileSystem(endpoint_url=endpoint_url)
#
#     with open(tmp_path / "test.foo1", "w") as file:
#         file.write("test123")
#
#     with open(tmp_path / "test.foo2", "w") as file:
#         file.write("test456")
#
#     s3_client.create_bucket(Bucket=test_bucket)
#     s3_client.upload_file(tmp_path / "test.foo1", test_bucket, "test.foo1")
#     s3_client.list_objects(Bucket=test_bucket)
#     s3fs.put_file(tmp_path / "test.foo2", f"s3://{test_bucket}/test.foo2")
#
#     all_objects = s3fs.ls(f"{test_bucket}")
#     assert len(all_objects) == 2


@mock_aws
# @pytest.mark.skip(reason="no way of currently testing add_file with s3fs")
def test_s3_map(moto_server, s3fs_manager_test_path, tmp_path):
    s3fs_manager = S3FSManager(endpoint_url=moto_server)
    # test_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")

    s3_manager = S3Manager(endpoint_url=moto_server)
    s3_manager.create_bucket(bucket_name=test_bucket)
    print(s3_manager.list_buckets())

    # --- Create Local Zarr Store --- #
    # temporary_directory = "/tmp"  # str(tmp_path)
    zarr_path = f"{tmp_path}/example.model"
    ds = xr.Dataset(
        {
            "a": (("y", "x"), np.random.rand(6).reshape(2, 3)),
            "b": (("y", "x"), np.random.rand(6).reshape(2, 3)),
        },
        coords={"y": [0, 1], "x": [10, 20, 30]},
    )
    # TODO: write zarr store directly to the s3 bucket?!
    ds.to_zarr(zarr_path)  # , zarr_format=2)

    # --- Upload to S3 --- #
    # TODO: just copy from a to b
    # foo = s3_manager.upload_files_to_bucket(local_directory=zarr_path, object_prefix='ship/cruise/sensor/example.model', bucket_name=test_bucket_name)
    # s3_manager.upload_file(zarr_path + '/.zmetadata', test_bucket_name, 'ship/cruise/sensor/example.model/.zmetadata')

    # s3fs_manager.upload_data(
    #     bucket_name=test_bucket_name,
    #     file_path=zarr_path,
    #     file_name='ship/cruise/sensor/example.model'
    # )

    # s3_object = s3_manager.get(bucket_name=test_bucket_name, key="ship/cruise/sensor/example.model/.zmetadata")
    # body = s3_object.get()["Body"].read().decode("utf-8")
    # print(body)

    ### The file is there, trying to copy with boto3, then mount with s3fs.... incompatible version of s3fs

    # assert s3_manager.folder_exists_and_not_empty(test_bucket_name, "/example.model")
    # TODO: get this working with s3 client
    s3fs_manager.upload_data(
        bucket_name=test_bucket, file_path=zarr_path, prefix="ship/cruise/sensor"
    )

    assert s3fs_manager.exists(f"s3://{test_bucket}/ship/cruise/sensor/example.model")

    found = s3_manager.list_objects(test_bucket, "ship/cruise/sensor/example.model")
    print(found)
    s3_object = s3_manager.get_object(
        bucket_name=test_bucket, key_name="ship/cruise/sensor/example.model/.zgroup"
    )
    body = s3_object.get("Body").read().decode("utf-8")
    print(body)

    s3_store = s3fs_manager.s3_map(
        s3_zarr_store_path=f"s3://{test_bucket}/ship/cruise/sensor/example.model"
    )

    # --- Test S3Map Opening Zarr store with Zarr for Writing --- #
    cruise_zarr = zarr.open(
        store=s3_store, mode="r+"
    )  # , synchronizer=synchronizer) # TODO: test synchronizer
    print(cruise_zarr.info)

    # TODO: test with actual write

    # --- Test S3Map Opening Zarr store with Xarray for Reading --- #
    # TODO: test SYNCHRONIZER as shared file in output bucket mounted via s3fs
    s3_zarr_xr = xr.open_zarr(
        store=s3_store, consolidated=None
    )  # synchronizer=SYNCHRONIZER
    print(s3_zarr_xr.info)

    assert s3_zarr_xr.a.shape == (2, 3)

    # Write new dataset to subset
    cruise_zarr.a[0, 1] = 42

    assert s3_zarr_xr.a[0, 1].values == 42


#####################################################################
#####################################################################
#####################################################################
