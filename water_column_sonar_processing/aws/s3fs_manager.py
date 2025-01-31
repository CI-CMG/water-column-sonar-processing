import os
from typing import Optional
import s3fs

# TODO: S3FS_LOGGING_LEVEL=DEBUG


class S3FSManager:
    #####################################################################
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
    ):
        self.endpoint_url = endpoint_url
        # self.input_bucket_name = os.environ.get("INPUT_BUCKET_NAME")
        # self.output_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")
        self.s3_region = os.environ.get("AWS_REGION", default="us-east-1")
        self.s3fs = s3fs.S3FileSystem(
            # asynchronous=False,
            endpoint_url=endpoint_url,
            key=os.environ.get("OUTPUT_BUCKET_ACCESS_KEY"),
            secret=os.environ.get("OUTPUT_BUCKET_SECRET_ACCESS_KEY"),
        )

    # s3_fs = s3fs.S3FileSystem( # TODO: use s3fs_manager?
    #     anon=True,
    #     client_kwargs={
    #         "endpoint_url": moto_server,
    #         "region_name": "us-east-1",
    #     },
    # )
    #####################################################################
    def s3_map(
        self,
        s3_zarr_store_path,  # f's3://{bucket}/{input_zarr_path}'
    ):
        # The "s3_zarr_store_path" is defined as f's3://{bucket}/{input_zarr_path}'
        # create=False, not false because will be writing
        # return s3fs.S3Map(root=s3_zarr_store_path, s3=self.s3fs, check=True)
        return s3fs.S3Map(
            root=s3_zarr_store_path,
            s3=self.s3fs
        )  # create=False, not false because will be writing

    #####################################################################
    # def add_file(self, filename):
    #     full_path = f"{os.getenv('OUTPUT_BUCKET_NAME')}/testing/{filename}"
    #     print(full_path)
    #
    #     self.s3fs.touch(full_path)
    #     ff = self.s3fs.ls(f"{os.getenv('OUTPUT_BUCKET_NAME')}/")
    #
    #     print(ff)

    #####################################################################
    def upload_data(
            self,
            bucket_name,
            file_path,
            prefix
    ):
        # TODO: this works in theory but use boto3 to upload files
        s3_path = f"s3://{bucket_name}/{prefix}/"
        s3_file_system = self.s3fs
        s3_file_system.put(file_path, s3_path, recursive=True)

    #####################################################################
    def exists(
        self,
        s3_path,
    ):
        # s3_file_system =
        return self.s3fs.exists(s3_path)


    #####################################################################
