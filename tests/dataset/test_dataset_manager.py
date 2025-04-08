import os

import numpy as np
import pytest
import xarray as xr
import xbatcher.generators
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

from water_column_sonar_processing.aws import S3Manager
from water_column_sonar_processing.dataset import DatasetManager


def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


@pytest.fixture(scope="module")
def moto_server():
    """Fixture to run a mocked AWS server for testing."""
    # Note: pass `port=0` to get a random free port.
    server = ThreadedMotoServer(port=0)
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
#     HB1906_loader = cruise.isel(depth=slice(None, 512), time=slice(None, 512), drop=True)
#     print(HB1906_loader)
#     #HB1906_loader.to_zarr(store='HB1906.zarr', mode="w", consolidated=True)
#     print('saved')


@mock_aws
def test_open_xarray_dataset(dataset_test_path, moto_server):
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
    sv_dataset = dataset_manager.open_xarray_dataset(
        bucket_name=test_bucket_name,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        endpoint_url=moto_server,
    )

    # --- verify dataset shape --- #
    assert isinstance(sv_dataset, xr.core.dataarray.Dataset)
    assert sv_dataset.Sv.values.dtype == "float32"
    assert sv_dataset.Sv.shape == (64, 32, 4)
    assert sv_dataset.Sv.tile_size == 512

    # TODO: verify metrics about depth/time/frequency/etc.


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
        pass

    assert batch.Sv.shape == (8, 8, 4)
    assert batch.Sv.dtype == "float32"
    assert np.allclose(batch.depth.values[[0, -1]], np.array([17.4, 18.8]))
    assert batch.time.values[0] == np.datetime64("2019-09-03T17:19:26.710808064")
    assert batch.time.values[-1] == np.datetime64("2019-09-03T17:19:33.721990912")

    assert np.allclose(
        batch.frequency.values, np.array([18000.0, 38000.0, 120000.0, 200000.0])
    )


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
        print(train_features[0, :, :, 0])
        print(train_labels[0, :, :, 0])
        if np.isnan(train_features).all():
            print("_+_+_+_+_ all nan, skip _+_+_+_+_+_+_+_")
        print("___________________")
        break

    assert np.isclose(np.mean(train_features), -86.18572)

    # TODO: figure out how to skip when all the data is nan
    # figure out how to pad missing data
    # test with actual nn
    # plot
