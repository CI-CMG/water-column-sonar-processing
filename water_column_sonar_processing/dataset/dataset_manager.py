from typing import Optional

import numpy as np
import xarray as xr

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
        bucket_name: str,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        endpoint_url: Optional[str] = None,
    ):
        self.bucket_name = bucket_name
        self.ship_name = ship_name
        self.cruise_name = cruise_name
        self.sensor_name = sensor_name
        self.endpoint_url = endpoint_url
        self.dtype = "float32"

    def open_xarray_dataset(
        self,
        mask: bool = True,
    ) -> xr.Dataset:
        # Opens Zarr store in s3 bucket as Xarray Dataset and masks as needed
        try:
            s3_path = f"s3://{self.bucket_name}/level_2/{self.ship_name}/{self.cruise_name}/{self.sensor_name}/{self.cruise_name}.zarr"

            s3fs_manager = S3FSManager(endpoint_url=self.endpoint_url)
            store_s3_map = s3fs_manager.s3_map(s3_zarr_store_path=s3_path)

            ds = xr.open_dataset(
                filename_or_obj=store_s3_map,
                engine="zarr",
                # backend_kwargs={'storage_options': {'anon': True}},
                chunks={},
                cache=False,
            )

            # Mask all sub-bottom dataset
            if mask:
                return ds.where(ds.depth < ds.bottom)

            return ds
        except Exception as err:
            raise RuntimeError(f"Problem opening Zarr store from S3 with Xarray, {err}")

    def vector_indices(
        self,
        first_index: int,
        last_index: int,
        step: int,
    ):
        starts = np.arange(first_index, last_index, step)
        ends = np.arange(step, last_index + 1, step)
        return list(zip(starts, ends))

    def dataset_batcher(
        self,
    ):
        """
        Opens a dataset and creates a generator that returns different chunks of data for processing.
        # TODO: get subset of cruise
        # TODO: if beneath bottom skip
        # TODO: preprocess? scale/normalize?
        # TODO: add in features
        # TODO: pass sv dataset
        """
        try:
            # open zarr store
            # sv_dataset = self.open_xarray_dataset(mask=True)

            # patch_input_dims = {"depth": 1, "time": 2, "frequency": 3}

            # define bounds
            outline_dims = {"depth": 7, "time": 4, "frequency": 2}

            bottom = np.array([5, np.nan, 3, 2])  # for nan should sample all depths

            for f in self.vector_indices(0, outline_dims["frequency"] + 1, 2):
                for t in self.vector_indices(0, outline_dims["time"] + 1, 2):
                    for d in self.vector_indices(0, outline_dims["depth"] + 1, 2):
                        indices = f"[d: {d}, t: {t}, f: {f}]"

                        if np.isnan(bottom[t]) or d > bottom[t]:
                            print("_+_+_+subbottom_+_+_+")
                            continue

                        yield indices
            # # generate
            # for f in np.arange(0, outline_dims['frequency'] + 1, 2):
            #     for t in np.arange(0, outline_dims['time'] + 1, 2):
            #         for d in np.arange(0, outline_dims['depth'] + 1, 2):
            #             indices = f"[d: {d}, t: {t}, f: {f}]"
            #             # TODO: get subset of cruise
            #             # TODO: if beneath bottom skip
            #             if np.isnan(bottom[t]) or d > bottom[t]:
            #                 print('_+_+_+subbottom_+_+_+')
            #                 continue
            #             # TODO: preprocess? scale/normalize?
            #             # TODO: add in features
            #             # TODO: pass sv dataset
            #             yield indices

        except Exception as err:
            raise RuntimeError(f"Problem defining dataset_batcher, {err}")

    # @deprecated("We cannot use xbatcher")
    # def setup_xbatcher(
    #     self,
    #     bucket_name: str,
    #     ship_name: str,
    #     cruise_name: str,
    #     sensor_name: str,
    #     endpoint_url: str = None,
    # ):#  -> xbatcher.generators.BatchGenerator:
    #     pass
    #     # try:
    #     #     sv_dataset = self.open_xarray_dataset(
    #     #         bucket_name=bucket_name,
    #     #         ship_name=ship_name,
    #     #         cruise_name=cruise_name,
    #     #         sensor_name=sensor_name,
    #     #         endpoint_url=endpoint_url,
    #     #     )
    #     #
    #     #     patch_input_dims = dict(
    #     #         depth=BatchShape.DEPTH.value,
    #     #         time=BatchShape.TIME.value,
    #     #         frequency=BatchShape.FREQUENCY.value,
    #     #     )
    #     #     patch_input_overlap = dict(depth=0, time=0, frequency=0)
    #     #
    #     #     batch_generator = xbatcher.generators.BatchGenerator(
    #     #         ds=sv_dataset.Sv,  # TODO: need to get the depth out of this somehow?
    #     #         input_dims=patch_input_dims,
    #     #         input_overlap=patch_input_overlap,
    #     #         # batch_dims={ "depth": 8, "time": 8, "frequency": 4 }, # no idea what this is doing
    #     #         concat_input_dims=False,
    #     #         preload_batch=False,  # Load each batch dynamically
    #     #         cache=None,  # TODO: figure this out
    #     #         # cache_preprocess=preprocess_batch,  # https://xbatcher.readthedocs.io/en/latest/user-guide/caching.html
    #     #     )
    #     #
    #     #     return batch_generator
    #     # except Exception as err:
    #     #     raise RuntimeError(f"Problem setting up xbatcher, {err}")

    # @deprecated("We cannot use xbatcher")
    # def create_keras_dataloader(
    #     self,
    #     bucket_name: str,
    #     ship_name: str,
    #     cruise_name: str,
    #     sensor_name: str,
    #     endpoint_url: str = None,
    #     batch_size: int = 3,
    # ):
    #     pass
    # x_batch_generator = self.setup_xbatcher(
    #     bucket_name=bucket_name,
    #     ship_name=ship_name,
    #     cruise_name=cruise_name,  # TODO: move all these to constructor
    #     sensor_name=sensor_name,
    #     endpoint_url=endpoint_url,
    # )
    #
    # def transform(
    #     x,
    # ):  # TODO: do clip and normalize here... [-100, 0] w mean at -65, clip?
    #     # return x + 1e-6  # (x + 50.) / 100.
    #     # return np.clip(x, -60, -50)
    #     return (x + 50.) / 100.
    #
    # keras_dataset = xbatcher.loaders.keras.CustomTFDataset(
    #     X_generator=x_batch_generator,
    #     y_generator=x_batch_generator,
    #     transform=transform,
    #     target_transform=transform,
    # )
    #
    # output_signature = tensorflow.TensorSpec(
    #     shape=(
    #         BatchShape.DEPTH.value, # 2
    #         BatchShape.TIME.value, # 3
    #         BatchShape.FREQUENCY.value, # 4
    #     ),
    #     dtype=tensorflow.float32,
    # )
    # train_dataloader = tensorflow.data.Dataset.from_generator(
    #     generator=lambda: iter(keras_dataset),
    #     output_signature=(output_signature, output_signature),
    # )
    #
    # return train_dataloader.batch(batch_size=BatchShape.BATCH_SIZE.value) # 5
