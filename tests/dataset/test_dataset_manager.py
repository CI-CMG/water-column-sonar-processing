import os

import numpy as np
import pytest
import xarray
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
    dataset_manager = DatasetManager(
        bucket_name=test_bucket_name,
        ship_name="Henry_B._Bigelow",
        cruise_name="HB1906",
        sensor_name="EK60",
        endpoint_url=moto_server,
    )
    sv_dataset = dataset_manager.open_xarray_dataset()

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
    # TODO: verify second batch sample

    # TODO: verify metrics about depth/time/frequency/etc.
    print("done")


# @pytest.mark.skip(reason="WIP")
@mock_aws
def test_dataset_batcher(dataset_test_path, moto_server):
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
    dataset_manager = DatasetManager(
        bucket_name=test_bucket_name,
        ship_name="Henry_B._Bigelow",
        cruise_name="HB1906",
        sensor_name="EK60",
        endpoint_url=moto_server,
    )
    sv_batch_generator = dataset_manager.dataset_batcher()

    # TODO: iterate through x dimensions
    for sv_batch in sv_batch_generator:
        # pass batch into tensorflow dataset
        print(sv_batch)  # bottoms = np.array([5, np.nan, 3, 2])

    # list(itertools.islice(sv_batch_generator(), 13))
    # list(itertools.islice(sv_batch_generator()))
    # list(cruise_batch_generator())
    # batch
    # cuts_depth = np.arange(0, 2, sv_shape[0])
    # cuts_time = np.arange(0, 3, sv_shape[1])
    # cuts_frequency = np.arange(0, 4, sv_shape[2])

    # --- verify generator operation --- #
    # assert isinstance(sv_batch_generator, xbatcher.generators.BatchGenerator)

    # Generates batch of Sv DataArray
    # for batch in sv_batch_generator:
    #     assert batch.shape == (
    #         BatchShape.DEPTH.value,
    #         BatchShape.TIME.value,
    #         BatchShape.FREQUENCY.value,
    #     )
    #     assert batch.dtype == "float32"
    #     assert np.allclose(batch.depth.values[[0, -1]], np.array([6.2, 6.4]))
    #     assert batch.time.values[0] == np.datetime64("2019-09-03T17:19:02.683080960")
    #     assert batch.time.values[-1] == np.datetime64("2019-09-03T17:19:04.684547072")
    #
    #     assert np.allclose(
    #         batch.frequency.values, np.array([18000.0, 38000.0, 120000.0, 200000.0])
    #     )
    #
    #     assert np.allclose(
    #         batch.data[:, :, 0],
    #         np.array(
    #             [[-88.06821, -88.39746, -79.93099], [-85.19297, -103.90151, -84.41688]]
    #         ),
    #     )
    #
    #     break  # TODO: only checking first batch, check second

    print("done")


# @mock_aws
# @pytest.mark.skip(reason="This will not work")
# def test_setup_xbatcher(dataset_test_path, moto_server):
#     test_bucket_name = os.environ.get("INPUT_BUCKET_NAME")
#
#     ship_name = "Henry_B._Bigelow"
#     cruise_name = "HB1906"
#     sensor_name = "EK60"
#
#     # --- set up initial resources --- #
#     s3_manager = S3Manager()
#     s3_manager.create_bucket(bucket_name=test_bucket_name)
#     print(s3_manager.list_buckets())
#
#     # --- upload test zarr store to mocked bucket --- #
#     s3_manager.upload_zarr_store_to_s3(
#         output_bucket_name=test_bucket_name,
#         local_directory=dataset_test_path,
#         object_prefix=os.path.join("level_2", ship_name, cruise_name, sensor_name),
#         cruise_name=cruise_name,
#     )
#
#     list_of_found_objects = s3_manager.list_objects(
#         bucket_name=test_bucket_name,
#         prefix=f"level_2/{ship_name}/{cruise_name}/",
#     )
#     assert len(list_of_found_objects) == 30
#
#     # --- test loading zarr store from mocked s3 bucket --- #
#     dataset_manager = DatasetManager()
#     sv_batch_generator = dataset_manager.setup_xbatcher(
#         bucket_name=test_bucket_name,
#         ship_name=ship_name,
#         cruise_name=cruise_name,
#         sensor_name=sensor_name,
#         endpoint_url=moto_server,
#     )
#
#     # --- verify generator operation --- #
#     assert isinstance(sv_batch_generator, xbatcher.generators.BatchGenerator)
#
#     # Generates batch of Sv DataArray
#     for batch in sv_batch_generator:
#         assert batch.shape == (
#             BatchShape.DEPTH.value,
#             BatchShape.TIME.value,
#             BatchShape.FREQUENCY.value,
#         )
#         assert batch.dtype == "float32"
#         assert np.allclose(batch.depth.values[[0, -1]], np.array([6.2, 6.4]))
#         assert batch.time.values[0] == np.datetime64("2019-09-03T17:19:02.683080960")
#         assert batch.time.values[-1] == np.datetime64("2019-09-03T17:19:04.684547072")
#
#         assert np.allclose(
#             batch.frequency.values, np.array([18000.0, 38000.0, 120000.0, 200000.0])
#         )
#
#         assert np.allclose(
#             batch.data[:, :, 0],
#             np.array(
#                 [[-88.06821, -88.39746, -79.93099], [-85.19297, -103.90151, -84.41688]]
#             ),
#         )
#
#         break  # TODO: only checking first batch, check second
#
#     print("done")


# @mock_aws
# @pytest.mark.skip(reason="This will not work")
# def test_create_keras_dataloader(dataset_test_path, moto_server):
#     test_bucket_name = os.environ.get("INPUT_BUCKET_NAME")
#
#     ship_name = "Henry_B._Bigelow"
#     cruise_name = "HB1906"
#     sensor_name = "EK60"
#
#     # --- set up initial resources --- #
#     s3_manager = S3Manager()
#     s3_manager.create_bucket(bucket_name=test_bucket_name)
#     print(s3_manager.list_buckets())
#
#     # --- upload test zarr store to mocked bucket --- #
#     s3_manager.upload_zarr_store_to_s3(
#         output_bucket_name=test_bucket_name,
#         local_directory=dataset_test_path,
#         object_prefix=os.path.join("level_2", ship_name, cruise_name, sensor_name),
#         cruise_name=cruise_name,
#     )
#
#     list_of_found_objects = s3_manager.list_objects(
#         bucket_name=test_bucket_name,
#         prefix=f"level_2/{ship_name}/{cruise_name}/",
#     )
#     assert len(list_of_found_objects) == 30
#
#     # --- test loading zarr store from mocked s3 bucket --- #
#     dataset_manager = DatasetManager()
#     train_dataloader = dataset_manager.create_keras_dataloader(
#         bucket_name=test_bucket_name,
#         ship_name=ship_name,
#         cruise_name=cruise_name,
#         sensor_name=sensor_name,
#         endpoint_url=moto_server,
#     )
#
#     # Extract a batch from the DataLoader
#     for train_features, train_labels in train_dataloader:
#         # assert train_features.shape = (X, Y, Z, W)
#         assert np.allclose(train_labels.numpy(), train_features.numpy())
#         assert np.allclose(
#             train_features[0, :, :, 0].numpy(), # first batch/sample/frequency
#             np.array(
#                 [[-88.06821, -88.39746, -79.93099], [-85.19297, -103.90151, -84.41688]]
#             ),
#         )
#
#         break  # TODO: only testing the first batch
#
#     print("done")
#
#     # 0,-88.06821,-88.39746
#     # 1,-85.19297,-103.90151
#     # assert np.isclose(np.mean(train_features), -86.18572)
#
#     # TODO: figure out how to skip when all the data is nan
#     # figure out how to pad missing data
#     # test with actual nn
#     # plot


# class Autoencoder(Model):
#   # https://www.tensorflow.org/tutorials/generative/autoencoder
#   def __init__(self, latent_dim, shape):
#     super(Autoencoder, self).__init__()
#     self.latent_dim = latent_dim
#     self.shape = shape
#     self.encoder = tensorflow.keras.Sequential([
#       layers.Flatten(),
#       layers.Dense(latent_dim, activation='linear'),
#     ])
#     self.decoder = tensorflow.keras.Sequential([
#       layers.Dense(tensorflow.math.reduce_prod(shape).numpy(), activation='linear'),
#       layers.Reshape(shape)
#     ])
#
#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded


# @mock_aws
# @pytest.mark.skip(reason="This will not work")
# def test_create_keras_ml(dataset_test_path, moto_server):
#     test_bucket_name = os.environ.get("INPUT_BUCKET_NAME")
#
#     ship_name = "Henry_B._Bigelow"
#     cruise_name = "HB1906"
#     sensor_name = "EK60"
#
#     # --- set up initial resources --- #
#     s3_manager = S3Manager()
#     s3_manager.create_bucket(bucket_name=test_bucket_name)
#     print(s3_manager.list_buckets())
#
#     # --- upload test zarr store to mocked bucket --- #
#     s3_manager.upload_zarr_store_to_s3(
#         output_bucket_name=test_bucket_name,
#         local_directory=dataset_test_path,
#         object_prefix=os.path.join("level_2", ship_name, cruise_name, sensor_name),
#         cruise_name=cruise_name,
#     )
#
#     list_of_found_objects = s3_manager.list_objects(
#         bucket_name=test_bucket_name,
#         prefix=f"level_2/{ship_name}/{cruise_name}/",
#     )
#     assert len(list_of_found_objects) == 30
#
#     # --- test loading zarr store from mocked s3 bucket --- #
#     dataset_manager = DatasetManager()
#     train_dataloader = dataset_manager.create_keras_dataloader(
#         bucket_name=test_bucket_name,
#         ship_name=ship_name,
#         cruise_name=cruise_name,
#         sensor_name=sensor_name,
#         endpoint_url=moto_server,
#     )
#
#     shape = (2, 3, 4)
#     latent_dim = 32
#     autoencoder = Autoencoder(latent_dim, shape)
#     autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
#     autoencoder.fit(
#         train_dataloader,
#         epochs=3,
#         verbose=2,
#         batch_size=5,
#         shuffle=False,
#     )
#     autoencoder.summary()
#
#     for test_sampleA, test_sampleB in train_dataloader:
#         break
#
#     # print('a')
#     # print(test_sampleA)
#     # print('b')
#     # print(test_sampleB)
#     encoded_values = autoencoder.encoder(test_sampleA.numpy()).numpy()
#     print('encoded')
#     print(encoded_values)
#     decoded_values = autoencoder.decoder(encoded_values).numpy()
#     print(decoded_values)
#
#     # Define a simple feedforward neural network
#     # model = models.Sequential(
#     #     [
#     #         layers.Flatten(input_shape=(2, 3, 4)),  # Flatten input
#     #         layers.Dense(16, activation='relu'),  # Fully connected layer with 128 units
#     #         layers.Dense(4, activation='softmax'),  # Output layer for 10 classes
#     #     ]
#     # )
#     #
#     # # Compile the model
#     # model.compile(
#     #     optimizer=optimizers.Adam(learning_rate=0.001),
#     #     loss='sparse_categorical_crossentropy',
#     #     metrics=['accuracy'],
#     # )
#     #
#     # # Display model summary
#     # model.summary()
#     #
#     # epochs = 5
#     #
#     # model.fit(
#     #     train_dataloader,  # Pass the DataLoader directly
#     #     epochs=epochs,
#     #     verbose=1,  # Print progress during training
#     # )
#     #
#     # # Visualize a prediction on a sample image
#     # for train_features, train_labels in train_dataloader:
#     #     img = train_features[0].numpy().squeeze()
#     #     label = train_labels[0].numpy()
#     #     # predicted_label = tf.argmax(model.predict(train_features[:1]), axis=1).numpy()[0]
#     #     #
#     #     # plt.imshow(img, cmap='gray')
#     #     # plt.title(f'True Label: {label}, Predicted: {predicted_label}')
#     #     # plt.show()
#     #     # break
