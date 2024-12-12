import os

import s3fs

# TODO: S3FS_LOGGING_LEVEL=DEBUG


class S3FSManager:
    #####################################################################
    def __init__(
        self,
    ):
        self.__s3_region = os.environ.get("AWS_REGION", default="us-east-1")
        self.s3fs = s3fs.S3FileSystem(
            key=os.environ.get("OUTPUT_BUCKET_ACCESS_KEY"),
            secret=os.environ.get("OUTPUT_BUCKET_SECRET_ACCESS_KEY"),
            # asynchronous=True
            # use_ssl=False,
            # skip_instance_cache=True,
            # default_block_size='100MB',  # if no specific value is given at all time. The built-in default is 5MB
            # client_kwargs={
            #     "region_name": self.__s3_region
            # }
        )

    #####################################################################
    def add_file(self, filename):
        full_path = f"{os.getenv('OUTPUT_BUCKET_NAME')}/testing/{filename}"
        print(full_path)

        self.s3fs.touch(full_path)
        ff = self.s3fs.ls(f"{os.getenv('OUTPUT_BUCKET_NAME')}/")

        print(ff)

    #####################################################################
    def upload_data(self, bucket_name, file_path, prefix):
        # TODO: this works in theory but use boto3 to upload files
        s3_path = f"s3://{bucket_name}/{prefix}/"
        s3_file_system = self.s3fs
        s3_file_system.put(file_path, s3_path, recursive=True)

    #####################################################################
    def s3_map(
        self,
        s3_zarr_store_path,  # f's3://{bucket}/{input_zarr_path}'
    ):
        # The "s3_zarr_store_path" is defined as f's3://{bucket}/{input_zarr_path}'
        # create=False, not false because will be writing
        # return s3fs.S3Map(root=s3_zarr_store_path, s3=self.s3fs, check=True)
        return s3fs.S3Map(
            root=s3_zarr_store_path, s3=self.s3fs
        )  # create=False, not false because will be writing

    #####################################################################
    def exists(
        self,
        geo_json_s3_path,
    ):
        s3_file_system = self.s3fs
        return s3_file_system.exists(path=geo_json_s3_path)

    #####################################################################
    # def put(
    #         self
    # ):
    #     s3_file_system = self.s3fs
    #     return
