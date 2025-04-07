import tensorflow as tf
import xarray as xr
import xbatcher
from xbatcher.loaders.keras import CustomTFDataset

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

    def open_xarray_data_array(
        self,
        bucket_name: str,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        endpoint_url: str = None,
    ) -> xr.DataArray:
        # Opens Zarr store in s3 bucket as Xarray Dataset and masks as needed
        try:
            # Define the path
            s3_path = f"s3://{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{cruise_name}.zarr"

            # Initialize S3FS
            s3fs_manager = S3FSManager(endpoint_url=endpoint_url)
            store_s3_map = s3fs_manager.s3_map(s3_zarr_store_path=s3_path)

            # Open Dataset
            ds = xr.open_dataset(
                filename_or_obj=store_s3_map,
                engine="zarr",
                chunks=None,
                cache=True,
            )

            # ds = ds.sel(
            #     # depth=slice(min_depth, max_depth),
            #     # time=slice(min_time, max_time),
            #     # frequency=ds.frequency.isin(frequencies)
            # )

            # Mask the sub-bottom dataset
            # TODO: verify that this returns the subset of interest without actually computing the entire store
            return ds.Sv.where(ds.depth < ds.bottom)

        except Exception as err:
            raise RuntimeError(f"Problem opening Zarr store from S3 with Xarray, {err}")

    def setup_xbatcher(
        self,
        bucket_name: str,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        endpoint_url: str = None,
    ) -> bool:
        sv_array = self.open_xarray_data_array(
            bucket_name=bucket_name,
            ship_name=ship_name,
            cruise_name=cruise_name,
            sensor_name=sensor_name,
            endpoint_url=endpoint_url,
        )

        patch_input_dims = dict(depth=8, time=8, frequency=4)
        patch_input_overlap = dict(depth=0, time=0, frequency=0)

        batch_generator = xbatcher.BatchGenerator(
            ds=sv_array,
            input_dims=patch_input_dims,
            input_overlap=patch_input_overlap,
            batch_dims={},
            concat_input_dims=False,
            preload_batch=False,  # Load each batch dynamically
            # cache=None, # TODO: figure this out
            # cache_preprocess=None, # TODO: figure this out
        )

        return batch_generator

    def create_keras_dataloader(
        self,
        bucket_name: str,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        endpoint_url: str = None,
    ):
        # get generator
        x_batch_generator = self.setup_xbatcher(
            bucket_name=bucket_name,
            ship_name=ship_name,
            cruise_name=cruise_name,
            sensor_name=sensor_name,
            endpoint_url=endpoint_url,
        )

        # create keras dataset
        # Use xbatcher's MapDataset to wrap the generators
        dataset = CustomTFDataset(x_batch_generator, x_batch_generator)

        # Create a DataLoader using tf.data.Dataset
        train_dataloader = tf.data.Dataset.from_generator(
            lambda: iter(dataset),
            output_signature=(
                tf.TensorSpec(
                    shape=(8, 8, 4), dtype=tf.float32
                ),  # shuffles the dims: {'depth': 8, 'frequency': 4, 'time': 8}
                tf.TensorSpec(shape=(8, 8, 4), dtype=tf.float32),
            ),
        ).prefetch(2)  # Prefetch N batches to improve performance
        # shape (8, 8, 4)
        return train_dataloader  # type(train_dataloader) == <class 'tensorflow.python.data.ops.prefetch_op._PrefetchDataset'>
