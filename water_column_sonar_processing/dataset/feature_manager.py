from typing import Optional

import xarray as xr


class DatasetManager:
    """
    Enrich the dataset with features
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

    def add_features(
        self,
    ) -> xr.Dataset:
        # Opens Zarr store in s3 bucket as Xarray Dataset and masks as needed
        try:
            pass
        except Exception as err:
            raise RuntimeError(f"Problem opening Zarr store from S3 with Xarray, {err}")
