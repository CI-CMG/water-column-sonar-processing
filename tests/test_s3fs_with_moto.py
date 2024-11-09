import boto3
import pytest
# import requests
# import s3fs
from s3fs import S3FileSystem
# import fs
# from pyarrow import fs, FileSelector
# import pyarrow as pa
# from pyarrow import fs
# from pyarrow.fs import FileSelector
# from moto import mock_aws
from moto import dynamodb, mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

test_bucket = "mybucket"
ip_address = "127.0.0.1"
port = 5555
endpoint_url = f"http://{ip_address}:{port}"

@pytest.fixture(scope="module")
def s3_base():
    s3_server = ThreadedMotoServer(ip_address=ip_address, port=port)
    s3_server.start()
    yield
    s3_server.stop()

@mock_aws
def test_load_all_files(s3_base):
    s3_session = boto3.Session()
    s3_client = s3_session.client(service_name="s3", endpoint_url=f"http://{ip_address}:{port}")
    s3_client.list_buckets()

    s3fs = S3FileSystem(endpoint_url=endpoint_url)

    with open("test.foo1", "w") as file:
        file.write("test123")

    with open("test.foo2", "w") as file:
        file.write("test456")

    s3_client.create_bucket(Bucket=test_bucket)
    s3_client.upload_file("test.foo1", test_bucket, "test.foo1")
    s3_client.list_objects(Bucket='mybucket')
    s3fs.put_file("test.foo2", f"s3://{test_bucket}/test.foo2")

    all_objects = s3fs.ls(f"{test_bucket}")
    assert len(all_objects) == 2

# @mock_aws
# @pytest.mark.skip(reason="not working currently, not intended to use this module anyway")
# def test_s3fs_with_moto():
#     # server = ThreadedMotoServer(ip_address="127.0.0.1", port=5000)
#     print('asdf')
    # server = ThreadedMotoServer(port=0)
    # server.start()
    # host, port = server.get_host_and_port()
    #
    # with open("test.foo1", "w") as file:
    #     file.write("test123")
    #
    # with open("test.bar", "w") as file:
    #     file.write("test456")
    #
    # # Create a mock S3 bucket
    # #s3 = boto3.client("s3", region_name='us-east-1', endpoint_url="http://localhost:5000")
    # s3_session = boto3.Session() # aws_access_key_id=os.environ.get("ACCESS_KEY_ID"), aws_secret_access_key=os.environ.get("SECRET_ACCESS_KEY"))
    # s3_client = s3_session.client(service_name="s3", endpoint_url=f"http://{host}:{port}")
    # sts_client = s3_session.client(service_name="sts", endpoint_url=f"http://{host}:{port}")
    # session_token = sts_client.get_session_token()
    # s3_client.list_buckets()
    # s3_client.create_bucket(Bucket="mybucket")
    # s3_client.upload_file("test.foo1", "mybucket", "test.foo1")
    # s3_client.list_objects(Bucket='mybucket')
    #
    # # s3fs = S3FileSystem(session=s3_session, endpoint_url="http://localhost:5000")
    # s3fs2 = S3FileSystem(token=session_token, use_ssl=False, endpoint_url=f"http://{host}:{port}")
    # s3fs2.ls("s3://mybucket")
    # # Read the file from the mock bucket
    # with s3fs2.open("s3://mybucket/test.foo1", "r") as f:
    #     content = f.read()
    #
    # # s3fs._copy("mybucket/test.foo", "mybucket/test.foo123")
    # s3fs2.put("test.bar", f"s3://mybucket/test.bar")
    # s3fs2.ls('s3://mybucket')
    # # fs.put("test.txt", "mybucket/test.txt")
    # assert content == "test.txt"