import json
import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from typing import Optional

import boto3
import botocore
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

MAX_POOL_CONNECTIONS = 64
MAX_CONCURRENCY = 64
MAX_WORKERS = 64
GB = 1024**3


#########################################################################
def chunked(ll: list, n: int) -> Generator:
    # Yields successively n-sized chunks from ll.
    for i in range(0, len(ll), n):
        yield ll[i : i + n]


class S3Manager:
    #####################################################################
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
    ):
        self.endpoint_url = endpoint_url
        self.s3_region = os.environ.get("AWS_REGION", default="us-east-1")
        self.s3_client_config = Config(max_pool_connections=MAX_POOL_CONNECTIONS)
        self.s3_transfer_config = TransferConfig(
            max_concurrency=MAX_CONCURRENCY,
            use_threads=True,
            max_bandwidth=None,
            multipart_threshold=10 * GB,
        )
        self.s3_session = boto3.Session(
            aws_access_key_id=os.environ.get("ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("SECRET_ACCESS_KEY"),
            region_name=self.s3_region,
        )
        self.s3_client = self.s3_session.client(
            service_name="s3",
            config=self.s3_client_config,
            region_name=self.s3_region,
            endpoint_url=self.endpoint_url,
        )
        self.s3_resource = boto3.resource(
            service_name="s3",
            config=self.s3_client_config,
            region_name=self.s3_region,
            endpoint_url=self.endpoint_url,
        )
        self.s3_session_noaa_wcsd_zarr_pds = boto3.Session(
            aws_access_key_id=os.environ.get("OUTPUT_BUCKET_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("OUTPUT_BUCKET_SECRET_ACCESS_KEY"),
            region_name=self.s3_region,
        )
        self.s3_client_noaa_wcsd_zarr_pds = self.s3_session_noaa_wcsd_zarr_pds.client(
            service_name="s3",
            config=self.s3_client_config,
            region_name=self.s3_region,
            endpoint_url=self.endpoint_url,
        )
        self.s3_resource_noaa_wcsd_zarr_pds = (
            self.s3_session_noaa_wcsd_zarr_pds.resource(
                service_name="s3",
                config=self.s3_client_config,
                region_name=self.s3_region,
                endpoint_url=self.endpoint_url,
            )
        )
        #
        self.paginator = self.s3_client.get_paginator("list_objects_v2")
        self.paginator_noaa_wcsd_zarr_pds = (
            self.s3_client_noaa_wcsd_zarr_pds.get_paginator("list_objects_v2")
        )

    #####################################################################
    # tested
    def create_bucket(
        self,
        bucket_name: str,
    ):
        """
        Note: this function is only really meant to be used for creating test
        buckets. It allows public read of all objects.
        """
        # https://github.com/aodn/aodn_cloud_optimised/blob/e5035495e782783cc8b9e58711d63ed466420350/test_aodn_cloud_optimised/test_schema.py#L7
        # public_policy = {
        #     "Version": "2012-10-17",
        #     "Statement": [
        #         {
        #             "Effect": "Allow",
        #             "Principal": "*",
        #             "Action": "s3:GetObject",
        #             "Resource": f"arn:aws:s3:::{bucket_name}/*",
        #         }
        #     ],
        # }
        response1 = self.s3_client.create_bucket(Bucket=bucket_name, ACL="public-read")
        print(response1)
        # response = self.s3_client.put_bucket_policy(
        #     Bucket=bucket_name, Policy=json.dumps(public_policy)
        # )
        # print(response)

    #####################################################################
    # tested
    def list_buckets(self):
        client = self.s3_client
        return client.list_buckets()

    #####################################################################
    def upload_nodd_file(
        self,
        file_name: str,
        key: str,
        output_bucket_name: str,
    ):
        """
        Used to upload a single file, e.g. the GeoJSON file to the NODD bucket
        """
        self.s3_resource_noaa_wcsd_zarr_pds.Bucket(output_bucket_name).upload_file(
            Filename=file_name, Key=key
        )
        return key

    #####################################################################
    def upload_files_with_thread_pool_executor(
        self,
        output_bucket_name: str,
        all_files: list,
    ):
        # 'all_files' is passed a list of lists: [[local_path, s3_key], [...], ...]
        all_uploads = []
        try:  # TODO: problem with threadpool here, missing child files
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        self.upload_nodd_file,  # TODO: verify which one is using this
                        all_file[0],  # file_name
                        all_file[1],  # key
                        output_bucket_name,  # output_bucket_name
                    )
                    for all_file in all_files
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        all_uploads.extend([result])
        except Exception as err:
            raise RuntimeError(f"Problem, {err}")

        print("Done uploading files using threading pool.")
        return all_uploads

    #####################################################################
    # tested
    def upload_zarr_store_to_s3(
        self,
        output_bucket_name: str,
        local_directory: str,
        object_prefix: str,
        cruise_name: str,
    ) -> None:
        print("uploading model store to s3")
        try:
            #
            print("Starting upload with thread pool executor.")
            # # 'all_files' is passed a list of lists: [[local_path, s3_key], [...], ...]
            all_files = []
            for subdir, dirs, files in os.walk(f"{local_directory}/{cruise_name}.zarr"):
                for file in files:
                    local_path = os.path.join(subdir, file)
                    # TODO: find a better method for splitting strings here:
                    # 'level_2/Henry_B._Bigelow/HB0806/EK60/HB0806.zarr/.zattrs'
                    # s3_key = f"{object_prefix}/{cruise_name}.zarr{local_path.split(f'{cruise_name}.zarr')[-1]}"
                    s3_key = os.path.join(
                        object_prefix,
                        os.path.join(
                            subdir[subdir.find(f"{cruise_name}.zarr") :], file
                        ),
                    )
                    all_files.append([local_path, s3_key])
            self.upload_files_with_thread_pool_executor(
                output_bucket_name=output_bucket_name,
                all_files=all_files,
            )
            print("Done uploading with thread pool executor.")
        except Exception as err:
            raise RuntimeError(f"Problem uploading zarr store to s3, {err}")

    #####################################################################
    # tested
    def upload_file(
        self,
        filename: str,
        bucket_name: str,
        key: str,
    ):
        self.s3_resource.Bucket(bucket_name).upload_file(Filename=filename, Key=key)

    #####################################################################
    # tested
    def check_if_object_exists(self, bucket_name, key_name) -> bool:
        s3_manager2 = S3Manager()
        s3_manager2.list_objects(bucket_name=bucket_name, prefix=key_name)
        s3_client_noaa_wcsd_zarr_pds = self.s3_client_noaa_wcsd_zarr_pds
        try:
            s3_client_noaa_wcsd_zarr_pds.head_object(Bucket=bucket_name, Key=key_name)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # The object does not exist.
                return False
            elif e.response["Error"]["Code"] == 403:
                # Unauthorized, including invalid bucket
                return False
            else:
                # Something else has gone wrong.
                raise

    #####################################################################
    # tested
    def list_objects(self, bucket_name, prefix):  # noaa-wcsd-pds and noaa-wcsd-zarr-pds
        # TODO: this isn't working for geojson detecting objects!!!!!!!
        # analog to "find_children_objects"
        # Returns a list of key strings for each object in bucket defined by prefix
        # s3_client = self.s3_client
        keys = []
        # paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = self.paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in page_iterator:
            if "Contents" in page.keys():
                keys.extend([k["Key"] for k in page["Contents"]])
        return keys

    #####################################################################
    # TODO: change name to "directory"
    # def folder_exists_and_not_empty(self, bucket_name: str, path: str) -> bool:
    #     if not path.endswith("/"):
    #         path = path + "/"
    #     # s3_client = self.s3_client
    #     resp = self.list_objects(
    #         bucket_name=bucket_name, prefix=path
    #     )  # TODO: this is returning root folder and doesn't include children or hidden folders
    #     # resp = s3_client.list_objects(Bucket=bucket, Prefix=path, Delimiter='/', MaxKeys=1)
    #     return "Contents" in resp

    #####################################################################
    # private
    def __paginate_child_objects(
        self,
        bucket_name: str,
        sub_prefix: str = None,
    ) -> list:
        page_iterator = self.s3_client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket_name, Prefix=sub_prefix
        )
        objects = []
        for page in page_iterator:
            if "Contents" in page.keys():
                objects.extend(page["Contents"])
        return objects

    #####################################################################
    # tested
    def get_child_objects(
        self,
        bucket_name: str,
        sub_prefix: str,
        file_suffix: str = None,
    ) -> list:
        print("Getting child objects")
        raw_files = []
        try:
            children = self.__paginate_child_objects(
                bucket_name=bucket_name,
                sub_prefix=sub_prefix,
            )
            if file_suffix is None:
                raw_files = children
            else:
                for child in children:
                    # Note: Any files with predicate 'NOISE' are to be ignored
                    # see: "Bell_M._Shimada/SH1507" cruise for more details.
                    if child["Key"].endswith(file_suffix) and not os.path.basename(
                        child["Key"]
                    ).startswith("NOISE"):
                        raw_files.append(child["Key"])
                return raw_files
        except ClientError as err:
            print(f"Problem was encountered while getting s3 files: {err}")
            raise
        print(f"Found {len(raw_files)} files.")
        return raw_files

    #####################################################################
    # tested
    def get_object(  # noaa-wcsd-pds or noaa-wcsd-zarr-pds
        self,
        bucket_name,
        key_name,
    ):
        # Meant for getting singular objects from a bucket, used by indexing lambda
        # can also return byte range potentially.
        print(f"Getting object {key_name} from {bucket_name}")
        try:
            response = self.s3_client.get_object(
                Bucket=bucket_name,
                Key=key_name,
            )
            # status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            # if status == 200:
            print(f"Done getting object {key_name} from {bucket_name}")
            return response
        except ClientError as err:
            print(f"Problem was encountered while getting s3 file: {err}")
            raise

    #####################################################################
    # tested
    def download_file(
        self,
        bucket_name,
        key,
        file_name,  # path to where the file will be saved
    ):
        try:
            self.s3_client.download_file(
                Bucket=bucket_name, Key=key, Filename=file_name
            )
            # TODO: if bottom file doesn't exist, don't fail downloader
            print("downloaded file")
        except Exception as err:
            raise RuntimeError(f"Problem was encountered while downloading_file, {err}")

    #####################################################################
    # tested
    def delete_nodd_objects(  # nodd-bucket
        self,
        bucket_name,
        objects: list,
    ):
        try:
            print(f"Deleting {len(objects)} objects in {bucket_name} in batches.")
            objects_to_delete = []
            for obj in objects:
                objects_to_delete.append({"Key": obj["Key"]})
            # Note: request can contain a list of up to 1000 keys
            for batch in chunked(ll=objects_to_delete, n=1000):
                # An error occurred (SlowDown) when calling the DeleteObjects operation (reached max retries: 4):
                # Please reduce your request rate.
                sleep(0.5)
                #
                self.s3_client_noaa_wcsd_zarr_pds.delete_objects(
                    Bucket=bucket_name, Delete={"Objects": batch}
                )
            print("Deleted files.")
        except Exception as err:
            raise RuntimeError(f"Problem was encountered while deleting objects, {err}")

    #####################################################################
    # tested
    def delete_nodd_object(  # only used to delete geojson it looks like?! Remove.
        self,
        bucket_name,
        key_name,
    ):
        try:
            print(f"Deleting {key_name} objects in {bucket_name}.")
            self.s3_client_noaa_wcsd_zarr_pds.delete_object(
                Bucket=bucket_name, Key=key_name
            )
            print("Deleted file.")
        except Exception as err:
            raise RuntimeError(f"Problem was encountered while deleting objects, {err}")

    #####################################################################
    # tested
    def put(self, bucket_name, key, body):  # noaa-wcsd-model-pds
        try:
            self.s3_client.put_object(
                Bucket=bucket_name, Key=key, Body=body
            )  # "Body" can be a file
        except Exception as err:
            raise RuntimeError(f"Problem was encountered putting object, {err}")

    #####################################################################
    # tested
    def read_s3_json(
        self,
        ship_name,
        cruise_name,
        sensor_name,
        file_name_stem,
        output_bucket_name,  # TODO: change to just bucket_name
    ) -> str:
        try:
            resource = self.s3_resource_noaa_wcsd_zarr_pds
            content_object = resource.Object(
                bucket_name=output_bucket_name,
                key=f"spatial/geojson/{ship_name}/{cruise_name}/{sensor_name}/{file_name_stem}.json",
            ).get()
            file_content = content_object["Body"].read().decode("utf-8")
            json_content = json.loads(file_content)
            return json_content
        except Exception as err:
            raise RuntimeError(f"Exception encountered reading s3 GeoJSON, {err}")


#########################################################################
