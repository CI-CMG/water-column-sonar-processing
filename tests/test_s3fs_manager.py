import os
from typing import Any  # Callable
from unittest.mock import MagicMock

import aiobotocore.awsrequest
import aiobotocore.endpoint
import aiohttp
import aiohttp.client_reqrep
import aiohttp.typedefs
import botocore.awsrequest
import botocore.model
import numpy as np

# import pytest
import xarray as xr
import zarr
from dotenv import find_dotenv, load_dotenv

# from moto import mock_aws
import moto
from moto.moto_server.threaded_moto_server import ThreadedMotoServer


from water_column_sonar_processing.aws.s3_manager import S3Manager
from water_column_sonar_processing.aws.s3fs_manager import S3FSManager


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)
    # https://docs.getmoto.org/en/latest/docs/server_mode.html
    # free_port = 5000


def teardown_module():
    print("teardown")


'''
#####################################################################
# ### ATTEMPT ONE #@###
# class MockAWSResponse(aiobotocore.awsrequest.AioAWSResponse):
#     """
#     Mocked AWS Response.
#     https://github.com/aio-libs/aiobotocore/issues/755
#     https://gist.github.com/giles-betteromics/12e68b88e261402fbe31c2e918ea4168
#     """
#
#     def __init__(self, response: botocore.awsrequest.AWSResponse):
#         self._moto_response = response
#         self.status_code = response.status_code
#         self.raw = MockHttpClientResponse(response)
#
#     # adapt async methods to use moto's response
#     async def _content_prop(self) -> bytes:
#         return self._moto_response.content
#
#     async def _text_prop(self) -> str:
#         return self._moto_response.text


# class MockHttpClientResponse(aiohttp.client_reqrep.ClientResponse):
#     """
#     Mocked HTP Response.
#     See <MockAWSResponse> Notes
#     """
#
#     def __init__(self, response: botocore.awsrequest.AWSResponse):
#         """
#         Mocked Response Init.
#         """
#         # super().__init__(response: botocore.awsrequest.AWSResponse)
#         # super().__init__()
#
#         # async def read(self: MockHttpClientResponse, n: int = -1) -> bytes:
#         async def read(self, n, int=-1) -> bytes:
#             return response.content
#
#         self.content = MagicMock(aiohttp.StreamReader)
#         self.content.read = read
#         self.response = response
#
#     @property
#     def raw_headers(self) -> Any:
#         # def raw_headers(self) -> aiohttp.typedefs.RawHeaders:
#         """
#         Return the headers encoded the way that aiobotocore expects them.
#         """
#         return {
#             k.encode("utf-8"): str(v).encode("utf-8")
#             for k, v in self.response.headers.items()
#         }.items()


# @pytest.fixture(scope="session", autouse=True)
# def patch_aiobotocore() -> None:
#     """
#     Pytest Fixture Supporting S3FS Mocks.
#     See <MockAWSResponse> Notes
#     """
#     def factory(original: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
#         """
#         Response Conversion Factory.
#         """
#         def patched_convert_to_response_dict(
#                 http_response: botocore.awsrequest.AWSResponse,
#                 operation_model: botocore.src.OperationModel,
#         ) -> Any:
#             return original(MockAWSResponse(http_response), operation_model)
#         return patched_convert_to_response_dict
#
#     aiobotocore.endpoint.convert_to_response_dict = factory(aiobotocore.endpoint.convert_to_response_dict)
#
# @pytest.fixture(scope="session")
# def patch_AWSResponse() -> None:
#     """Patch bug in botocore, see https://github.com/aio-libs/aiobotocore/issues/755"""
#
#     if moto.core.botocore_stubber.MockRawResponse.__name__ == "MockRawResponse":
#
#         def factory(original: Callable) -> Callable:
#             def patched_convert_to_response_dict(
#                     http_response: botocore.awsrequest.AWSResponse,
#                     operation_model: botocore.src.OperationModel,
#             ):
#                 return original(MockAWSResponse(http_response), operation_model)
#
#             return patched_convert_to_response_dict
#
#         aiobotocore.endpoint.convert_to_response_dict = factory(
#             aiobotocore.endpoint.convert_to_response_dict
#         )
#
#         # def factory_2(original: Callable) -> Callable:
#         #     def patched_looks_like_special_case_error(response, **kwargs):
#         #         return original(MockAWSResponse(response), **kwargs)
#         #
#         #     return patched_looks_like_special_case_error
#         #
#         # aiobotocore.handlers._looks_like_special_case_error = factory_2(
#         #     aiobotocore.handlers._looks_like_special_case_error
#         # )
#
#         class PatchedMockRawResponse(moto.core.botocore_stubber.MockRawResponse):
#             async def read(self, size=None):
#                 return super().read()
#
#             def stream(self, **kwargs):  # pylint: disable=unused-argument
#                 contents = super().read()
#                 while contents:
#                     yield contents
#                     contents = super().read()
#
#         class PatchedAWSResponse(botocore.awsrequest.AWSResponse):
#             raw_headers = {}  # type: ignore
#
#             async def read(self):  # type: ignore
#                 return self.text.encode()
#
#         moto.core.botocore_stubber.MockRawResponse = PatchedMockRawResponse
#         botocore.awsrequest.AWSResponse = PatchedAWSResponse

#####################################################################
# ### ATTEMPT TWO #@###
# T = TypeVar("T")
# R = TypeVar("R")
#
# @dataclass
# class _PatchedAWSReponseContent:
#     """Patched version of `botocore.awsrequest.AWSResponse.content`"""
#     content: bytes | Awaitable[bytes]
#
#     def __await__(self) -> Iterator[bytes]:
#         async def _generate_async() -> bytes:
#             if isinstance(self.content, Awaitable):
#                 return await self.content
#             else:
#                 return self.content
#
#         return _generate_async().__await__()
#
#     def decode(self, encoding: str) -> str:
#         assert isinstance(self.content, bytes)
#         return self.content.decode(encoding)
#
#
# class PatchedAWSResponse:
#     """Patched version of `botocore.awsrequest.AWSResponse`"""
#
#     def __init__(self, response: botocore.awsrequest.AWSResponse) -> None:
#         self._response = response
#         self.status_code = response.status_code
#         self.headers = response.headers
#         self.url = response.url
#         self.content = _PatchedAWSReponseContent(response.content)
#         self.raw = response.raw
#         if not hasattr(self.raw, "raw_headers"):
#             # self.raw.raw_headers = {}
#             self.raw.raw_headers = {}
#         # if not hasattr(self.raw, "raw_headers"):
#         #     self.raw.raw_headers = {
#         #         k.encode("utf-8"): str(v).encode("utf-8")
#         #         for k, v in self._response.headers.items()
#         #     }.items()
#
#
# class PatchedRetryContext(botocore.retries.standard.RetryContext):
#     """Patched version of `botocore.retries.standard.RetryContext`"""
#
#     def __init__(self, *args, **kwargs):
#         if kwargs.get("http_response"):
#             kwargs["http_response"] = PatchedAWSResponse(kwargs["http_response"])
#         super().__init__(*args, **kwargs)
#
#
# def _factory(
#         original: Callable[[botocore.awsrequest.AWSResponse, T], Awaitable[R]]
# ) -> Callable[[botocore.awsrequest.AWSResponse, T], Awaitable[R]]:
#     async def patched_convert_to_response_dict(http_response: botocore.awsrequest.AWSResponse, operation_model: T) -> R:
#         return await original(PatchedAWSResponse(http_response), operation_model)  # type: ignore[arg-type]
#
#     return patched_convert_to_response_dict
#
#
# aiobotocore.endpoint.convert_to_response_dict = _factory(aiobotocore.endpoint.convert_to_response_dict)  # type: ignore[assignment]
# botocore.retries.standard.RetryContext = PatchedRetryContext
'''


#####################################################################
#####################################################################
# @mock_aws
# @pytest
# def test_add_file(tmp_path):
def test_add_file():
    # https://github.com/fsspec/s3fs/blob/2c074502c2d6a9be0d3f05eb678f4cc5add2e7e5/s3fs/tests/test_s3fs.py#L76
    # https://github.com/search?q=mock_aws+s3fs&type=code
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=5555)
    server.start()
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"

    s3fs_manager = S3FSManager()
    test_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")

    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())

    # --- Create Local Zarr Store --- #
    temporary_directory = "/tmp"  # str(tmp_path)
    zarr_path = f"{temporary_directory}/example.model"
    ds = xr.Dataset(
        {
            "a": (("y", "x"), np.random.rand(6).reshape(2, 3)),
            "b": (("y", "x"), np.random.rand(6).reshape(2, 3)),
        },
        coords={"y": [0, 1], "x": [10, 20, 30]},
    )
    ds.to_zarr(zarr_path, zarr_version=2)  # TODO: jump to version 3

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
    # assert s3fs_manager.exists(f"{test_bucket_name}/ship/cruise/sensor/example.model")
    #
    #
    # TODO: need to upload the file!!
    #
    # s3_manager.upload_zarr_files_to_bucket(
    #     bucket_name=test_bucket_name,
    #     local_directory=zarr_path,
    #     remote_directory="ship/cruise/sensor",
    # )
    # TODO: get this working with s3 client
    s3fs_manager.upload_data(
        bucket_name=test_bucket_name, file_path=zarr_path, prefix="ship/cruise/sensor"
    )

    found = s3_manager.list_objects(
        test_bucket_name, "ship/cruise/sensor/example.model"
    )
    print(found)
    s3_object = s3_manager.get(
        bucket_name=test_bucket_name, key="ship/cruise/sensor/example.model/.zgroup"
    )
    body = s3_object.get()["Body"].read().decode("utf-8")
    print(body)

    s3_store = s3fs_manager.s3_map(
        s3_zarr_store_path=f"s3://{test_bucket_name}/ship/cruise/sensor/example.model"
    )

    # --- Test S3Map Opening Zarr store with Zarr for Writing --- #
    cruise_zarr = zarr.open(
        store=s3_store, mode="r+"
    )  # , synchronizer=synchronizer) # TODO: test synchronizer
    print(cruise_zarr.info)

    # --- Test S3Map Opening Zarr store with Xarray for Reading --- #
    # TODO: test SYNCHRONIZER as shared file in output bucket mounted via s3fs
    s3_zarr_xr = xr.open_zarr(
        store=s3_store, consolidated=None
    )  # synchronizer=SYNCHRONIZER
    print(s3_zarr_xr.info)

    assert s3_zarr_xr.a.shape == (2, 3)

    # Write new data to subset
    cruise_zarr.a[0, 1] = 42

    assert s3_zarr_xr.a[0, 1].values == 42

    # server.stop()
    server.stop()


#####################################################################
#####################################################################
#####################################################################
