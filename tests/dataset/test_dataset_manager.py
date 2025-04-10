import os

import numpy as np
import pytest
import xarray
import xbatcher.generators
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

# from water_column_sonar_processing.utility import Constants
from water_column_sonar_processing.aws import S3Manager
from water_column_sonar_processing.dataset import DatasetManager
from water_column_sonar_processing.utility.constants import BatchShape

# @pytest.fixture(scope="session")
# def clear_default_boto3_session():
#     boto3.DEFAULT_SESSION = None


def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


@pytest.fixture(scope="module")
def moto_server():
    """Fixture to run a mocked AWS server for testing."""
    # Note: pass `port=0` to get a random free port.
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


def teardown_module():
    print("teardown")


@pytest.fixture
def dataset_test_path(test_path):
    return test_path["DATASET_TEST_PATH"]


# def test_generate_test_data():
#     # This function is only meant to be run once to create offline test dataset
#     bucket_name = 'noaa-wcsd-zarr-pds'
#     ship_name = "Henry_B._Bigelow"
#     cruise_name = "HB1906"
#     sensor_name = "EK60"
#     zarr_store = f'{cruise_name}.zarr'
#     s3_zarr_store_path = f"{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{zarr_store}"
#     cruise = xr.open_dataset(f"s3://{s3_zarr_store_path}", engine='zarr', storage_options={'anon': True})
#     # ~34 kB of dataset
#     # HB1906_loader = cruise.isel(depth=slice(None, 512), time=slice(None, 512), drop=True)
#     HB1906_loader = cruise.isel(time=slice(None, 100000), drop=True) # (2507, 4_228_924, 4)
#     print(HB1906_loader)
#     #HB1906_loader.to_zarr(store='HB1906.zarr', mode="w", consolidated=True)
#     print('saved')


@mock_aws
def test_open_xarray_dataset(dataset_test_path, moto_server):
    test_bucket_name = os.environ.get("INPUT_BUCKET_NAME")

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB1906"
    sensor_name = "EK60"

    # # --- set up initial resources --- #
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())

    # --- upload test zarr store to mocked bucket --- #
    s3_manager.upload_zarr_store_to_s3(
        output_bucket_name=test_bucket_name,
        local_directory=dataset_test_path,
        object_prefix=os.path.join("level_2", ship_name, cruise_name, sensor_name),
        cruise_name=cruise_name,
    )

    list_of_found_objects = s3_manager.list_objects(
        bucket_name=test_bucket_name,
        prefix=f"level_2/{ship_name}/{cruise_name}/",
    )
    assert len(list_of_found_objects) == 30

    # --- test loading zarr store from mocked s3 bucket --- #
    dataset_manager = DatasetManager()
    sv_dataset = dataset_manager.open_xarray_dataset(
        bucket_name=test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        endpoint_url=moto_server,
    )

    # --- verify dataset shape --- #
    assert isinstance(sv_dataset.Sv, xarray.DataArray)
    assert sv_dataset.Sv.values.dtype == "float32"
    assert sv_dataset.Sv.shape == (64, 32, 4)
    assert sv_dataset.Sv.tile_size == 512

    # First four values of DataArray first frequency
    # 0,-88.06821,-88.39746
    # 1,-85.19297,-103.90151
    assert np.allclose(
        sv_dataset.Sv.values[:2, :2, 0],
        np.array([[-88.06821, -88.39746], [-85.19297, -103.90151]]),
    )
    # TODO: verify metrics about depth/time/frequency/etc.
    print("done")


@mock_aws
def test_setup_xbatcher(dataset_test_path, moto_server):
    test_bucket_name = os.environ.get("INPUT_BUCKET_NAME")

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB1906"
    sensor_name = "EK60"

    # --- set up initial resources --- #
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())

    # --- upload test zarr store to mocked bucket --- #
    s3_manager.upload_zarr_store_to_s3(
        output_bucket_name=test_bucket_name,
        local_directory=dataset_test_path,
        object_prefix=os.path.join("level_2", ship_name, cruise_name, sensor_name),
        cruise_name=cruise_name,
    )

    list_of_found_objects = s3_manager.list_objects(
        bucket_name=test_bucket_name,
        prefix=f"level_2/{ship_name}/{cruise_name}/",
    )
    assert len(list_of_found_objects) == 30

    # --- test loading zarr store from mocked s3 bucket --- #
    dataset_manager = DatasetManager()
    sv_batch_generator = dataset_manager.setup_xbatcher(
        bucket_name=test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        endpoint_url=moto_server,
    )

    # --- verify generator operation --- #
    assert isinstance(sv_batch_generator, xbatcher.generators.BatchGenerator)

    # Generates batch of Sv DataArray
    for batch in sv_batch_generator:
        assert batch.shape == (
            BatchShape.DEPTH.value,
            BatchShape.TIME.value,
            BatchShape.FREQUENCY.value,
        )
        assert batch.dtype == "float32"
        assert np.allclose(batch.depth.values[[0, -1]], np.array([6.2, 6.4]))
        assert batch.time.values[0] == np.datetime64("2019-09-03T17:19:02.683080960")
        assert batch.time.values[-1] == np.datetime64("2019-09-03T17:19:04.684547072")

        assert np.allclose(
            batch.frequency.values, np.array([18000.0, 38000.0, 120000.0, 200000.0])
        )

        assert np.allclose(
            batch.data[:, :, 0],
            np.array(
                [[-88.06821, -88.39746, -79.93099], [-85.19297, -103.90151, -84.41688]]
            ),
        )

        break  # TODO: only checking first batch, check second

    print("done")


@mock_aws
def test_create_keras_dataloader(dataset_test_path, moto_server):
    test_bucket_name = os.environ.get("INPUT_BUCKET_NAME")

    ship_name = "Henry_B._Bigelow"
    cruise_name = "HB1906"
    sensor_name = "EK60"

    # --- set up initial resources --- #
    s3_manager = S3Manager()
    s3_manager.create_bucket(bucket_name=test_bucket_name)
    print(s3_manager.list_buckets())

    # --- upload test zarr store to mocked bucket --- #
    s3_manager.upload_zarr_store_to_s3(
        output_bucket_name=test_bucket_name,
        local_directory=dataset_test_path,
        object_prefix=os.path.join("level_2", ship_name, cruise_name, sensor_name),
        cruise_name=cruise_name,
    )

    list_of_found_objects = s3_manager.list_objects(
        bucket_name=test_bucket_name,
        prefix=f"level_2/{ship_name}/{cruise_name}/",
    )
    assert len(list_of_found_objects) == 30

    # --- test loading zarr store from mocked s3 bucket --- #
    dataset_manager = DatasetManager()
    train_dataloader = dataset_manager.create_keras_dataloader(
        bucket_name=test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        endpoint_url=moto_server,
    )

    # Extract a batch from the DataLoader
    for train_features, train_labels in train_dataloader:
        # assert train_features.shape = (X, Y, Z, W)
        assert np.allclose(train_labels.numpy(), train_features.numpy())
        assert np.allclose(
            train_features[0, :, :, 0].numpy(),
            np.array(
                [[-88.06821, -88.39746, -79.93099], [-85.19297, -103.90151, -84.41688]]
            ),
        )

        break  # TODO: only testing the first batch

    print("done")

    # 0,-88.06821,-88.39746
    # 1,-85.19297,-103.90151
    # assert np.isclose(np.mean(train_features), -86.18572)

    # TODO: figure out how to skip when all the data is nan
    # figure out how to pad missing data
    # test with actual nn
    # plot
