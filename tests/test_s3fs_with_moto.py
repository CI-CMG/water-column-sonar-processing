import boto3
import os
import s3fs

# from moto import mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

# @mock_aws
def test_s3fs_with_moto():
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=5555)
    server.start()
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    if "AWS_SESSION_TOKEN" not in os.environ:
        os.environ["AWS_SESSION_TOKEN"] = "foo"

    # Create a mock S3 bucket
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="mybucket")

    # Use s3fs to interact with the mock bucket
    fs = s3fs.S3FileSystem(anon=False)
    fs.put("test.txt", "mybucket/test.txt")

    # Read the file from the mock bucket
    with fs.open("mybucket/test.txt", "r") as f:
        content = f.read()

    assert content == "test.txt"