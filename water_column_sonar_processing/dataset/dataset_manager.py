import tensorflow as tf
import xarray as xr
import xbatcher
import xbatcher.loaders.keras

from water_column_sonar_processing.aws import S3FSManager


class DatasetManager:
    """
    Dataset manager does three things.
    1) Opens zarr store in s3 bucket with xarray and returns masked dataset
    2) Loads Xarray DataSet with Xbatcher
    3) Loads Xbatcher batches into tensorflow dataset
    """

    def __init__(
        self,
    ):
        self.dtype = "float32"

    def open_xarray_dataset(
        self,
        bucket_name: str,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        endpoint_url: str = None,
    ) -> xr.Dataset:
        # Opens Zarr store in s3 bucket as Xarray Dataset and masks as needed
        try:
            s3_path = f"s3://{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{cruise_name}.zarr"

            s3fs_manager = S3FSManager(endpoint_url=endpoint_url)
            store_s3_map = s3fs_manager.s3_map(s3_zarr_store_path=s3_path)

            ds = xr.open_dataset(
                filename_or_obj=store_s3_map,
                engine="zarr",
                chunks=None,
                cache=True,
            )

            # Mask all sub-bottom dataset
            ds_select = ds.where(ds.depth < ds.bottom)
            return ds_select

        except Exception as err:
            raise RuntimeError(f"Problem opening Zarr store from S3 with Xarray, {err}")

    def setup_xbatcher(
        self,
        bucket_name: str,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        endpoint_url: str = None,
    ):  # -> xbatcher.generators.BatchGenerator:
        try:
            sv_dataset = self.open_xarray_dataset(
                bucket_name=bucket_name,
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                endpoint_url=endpoint_url,
            )

            patch_input_dims = dict(depth=16, time=8, frequency=4)
            patch_input_overlap = dict(depth=0, time=0, frequency=0)

            # def preprocess_batch(batch):
            #     # example: add a new variable to each batch
            #     # TODO: can i get this to do something
            #     return batch.Sv  # .Sv.fillna(0)

            batch_generator = xbatcher.BatchGenerator(
                ds=sv_dataset.Sv,  # TODO: need to get the depth out of this somehow?
                input_dims=patch_input_dims,
                input_overlap=patch_input_overlap,
                # batch_dims={ "depth": 8, "time": 8, "frequency": 4 }, # no idea what this is doing
                concat_input_dims=False,
                preload_batch=False,  # Load each batch dynamically
                # cache=None, # TODO: figure this out
                # cache_preprocess=preprocess_batch,  # https://xbatcher.readthedocs.io/en/latest/user-guide/caching.html
            )

            return batch_generator
        except Exception as err:
            raise RuntimeError(f"Problem setting up xbatcher, {err}")

    def create_keras_dataloader(
        self,
        bucket_name: str,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        endpoint_url: str = None,
        batch_size: int = 3,
    ):
        x_batch_generator = self.setup_xbatcher(
            bucket_name=bucket_name,
            ship_name=ship_name,
            cruise_name=cruise_name,  # TODO: move all these to constructor
            sensor_name=sensor_name,
            endpoint_url=endpoint_url,
        )

        def transform(
            x,
        ):  # TODO: do normalization here... [-100, 0] w mean at -65, clip?
            return x + 0.01  # (x + 50.) / 100.
            # return np.clip(x, -60, -50)

        dataset = xbatcher.loaders.keras.CustomTFDataset(
            X_generator=x_batch_generator,
            y_generator=x_batch_generator,
            transform=transform,
            target_transform=transform,
        )

        train_dataloader = tf.data.Dataset.from_generator(
            generator=lambda: iter(dataset),
            output_signature=(
                tf.TensorSpec(
                    shape=(8, 8, 4), dtype=tf.float32
                ),  # TODO: parameterize dimension
                tf.TensorSpec(shape=(8, 8, 4), dtype=tf.float32),
            ),
        )

        # TODO: somehow skip when all nan here???
        return train_dataloader.batch(batch_size=batch_size)
