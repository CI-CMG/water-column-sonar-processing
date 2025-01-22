import xarray as xr
import numpy as np
import pandas as pd
import xbatcher
from typing import Optional
# s3fs.core.setup_logging("DEBUG")

class BatchDownloader:
    """
    Uses the xbatcher XbatchDownloader to download data from an xarray dataset. Connection
    is established
    """

    def __init__(
            self,
            bucket_name: Optional[str] = "noaa-wcsd-zarr-pds",
            ship_name: Optional[str] = "Henry_B._Bigelow",
            cruise_name: Optional[str] = "HB0707",
            sensor_name: Optional[str] = "EK60",
            patch_dims: Optional[int] = 64, # TODO: change to 64
            # input_steps: Optional[int] = 3,
    ):
        self.bucket_name = bucket_name
        self.ship_name = ship_name
        self.cruise_name = cruise_name
        self.sensor_name = sensor_name
        self.patch_dims = patch_dims

    # TODO: move this to the s3fs module
    def get_s3_zarr_store(self) -> xr.Dataset:
        """ Returns an Xarray Dataset """
        s3_zarr_store_path = f"{self.bucket_name}/level_2/{self.ship_name}/{self.cruise_name}/{self.sensor_name}/{self.cruise_name}.zarr"
        # Info about the HB0707 cruise:
        #   Time: ["2007-07-11T18:20:33.657573888", "2007-07-11T18:20:53.657573888", "2007-07-13T00:55:17.454448896"]
        #   Frequency: [ 18000.  38000. 120000. 200000.]
        #   Depth: [0.19, 999.74]

        # Needed to override credentials for github actions
        # s3_file_system = s3fs.S3FileSystem(anon=True)
        # store = s3fs.S3Map(root=s3_zarr_store_path, s3=s3_file_system, check=False)

        # return xr.open_zarr(store=f"s3://{s3_zarr_store_path}", consolidated=True, storage_options={'anon': True})
        return xr.open_dataset(f"s3://{s3_zarr_store_path}", engine="zarr", storage_options={'anon': True})
        # return xr.open_zarr(store, consolidated=True)

    def get_toy_batch_generator(self) -> xbatcher.BatchGenerator:
        """
        Returns a BatchGenerator with subsets of Sv data
        Note: this is synthetic data, for a smaller toy example
        """
        depth = np.arange(1, 21) # N meters
        time = pd.date_range(start="2025-01-01", end="2025-01-31", freq='D') # N days
        frequency = [1_000, 2_000, 3_000] # N frequencies
        Sv = np.random.rand(len(depth), len(time), len(frequency)) # synthetic data
        cruise = xr.Dataset(
            data_vars={
                "Sv": (["depth", "time", "frequency"], Sv)
            },
            coords={
                "depth": depth,
                "time": time,
                "frequency": frequency,
            },
            attrs=dict(description="Toy Example"),
        )
        batch_generator = xbatcher.BatchGenerator(
            ds=cruise,
            # get samples that are shaped 10x10x3
            input_dims={ 'depth': 10, 'time': 10, 'frequency': cruise.frequency.shape[0] }, # A dictionary specifying the size of the inputs in each dimension, e.g. ``{'lat': 30, 'lon': 30}`` These are the dimensions the ML library will see. All other dimensions will be stacked into one dimension called ``sample``.
            # no overlap between samples
            input_overlap={ 'depth': 0, 'time': 0, 'frequency': 0 }, # Zero means no overlap. A dictionary specifying the overlap along each dimension
        )
        return batch_generator

    def get_s3_batch_generator(self) -> xbatcher.BatchGenerator:
        """ Returns a BatchGenerator with subsets of Sv data from s3 Zarr store """
        cruise = self.get_s3_zarr_store()

        # TODO: temporarily limits to a smaller slice of the data
        cruise_select = (cruise
            .where(cruise.depth < 100., drop=True)
            .sel(time=slice("2007-07-11T18:20:33", "2007-07-11T18:20:53"))
            # .sel(time=slice("2007-07-11T18:20:00", "2007-07-11T19:20:00"))
        )
        print(cruise_select.Sv.shape) # (526 depth, 21 time, 4 freq)

        batch_generator = xbatcher.BatchGenerator(
            ds=cruise_select,
            input_dims={ 'depth': 10, 'time': 10, 'frequency': cruise.frequency.shape[0] }, # A dictionary specifying the size of the inputs in each dimension, e.g. ``{'lat': 30, 'lon': 30}`` These are the dimensions the ML library will see. All other dimensions will be stacked into one dimension called ``sample``.
            input_overlap={ 'depth': 0, 'time': 0, 'frequency': 0 }, # Zero means no overlap. A dictionary specifying the overlap along each dimension
            preload_batch=False,
        )

        # TODO: need to raise exception if all the data is nan

        return batch_generator
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator

    def get_s3_manual_batch_generator(self):
        """
        Using just xarray (no xbatcher), iterate through the data and generate batches.
        Returns a BatchGenerator with subsets of Sv data from s3 Zarr store.
        """
        cruise = self.get_s3_zarr_store()

        # TODO: temporarily limits to a smaller slice of the data
        cruise_select = (cruise
            .where(cruise.depth < 100., drop=True)
            .sel(time=slice("2007-07-11T18:20:33", "2007-07-11T18:20:53"))
        )
        print(cruise_select.Sv.shape) # (526 depth, 21 time, 4 freq)
        batch_generator = xbatcher.BatchGenerator(
            ds=cruise_select,
            input_dims={ 'depth': 10, 'time': 10, 'frequency': cruise.frequency.shape[0] }, # A dictionary specifying the size of the inputs in each dimension, e.g. ``{'lat': 30, 'lon': 30}`` These are the dimensions the ML library will see. All other dimensions will be stacked into one dimension called ``sample``.
            input_overlap={ 'depth': 0, 'time': 0, 'frequency': 0 }, # Zero means no overlap. A dictionary specifying the overlap along each dimension
            preload_batch=True,
        )

        # TODO: need to raise exception if all the data is nan

        return batch_generator
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator

"""
(105, 21, 4)

depth-start: 0.1899999976158142, depth-end: 1.899999976158142
time-start: 2007-07-11T18:20:33.657573888, time-end: 2007-07-11T18:20:42.657573888
frequency-start: 18000.0, frequency-end: 200000.0
(10, 10, 4)
np.nanmean: -53.70000076293945
"""