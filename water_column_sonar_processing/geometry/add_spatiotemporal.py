import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from water_column_sonar_processing.model import ZarrManager


# # Convert "meters per second" to "knots"
# meters_per_second_to_knots = lambda mps_value: mps_value * 1.94384


class Spatiotemporal:
    #######################################################
    def __init__(
        self,
    ):
        self.NANOSECONDS_PER_SECOND = 1e9
        self.CUTOFF_DISTANCE_METERS = 50.0
        self.CUTOFF_TIME_SECONDS = 10.0

    #######################################################
    @staticmethod
    def meters_per_second_to_knots(
        mps_value,
    ):
        return mps_value * 1.94384

    #######################################################
    def compute_speed_and_distance(
        self,
        times: np.ndarray[tuple[int], np.dtype[np.datetime64]],
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> pd.DataFrame:
        try:
            geom = [Point(xy) for xy in zip(longitudes, latitudes)]
            points_df = gpd.GeoDataFrame({"geometry": geom}, crs="EPSG:4326")
            # Conversion to a rectilinear projection coordinate system where distance can be calculated with pythagorean theorem
            # EPSG:4087, WGS 84 / World Equidistant Cylindrical
            # https://epsg.io/4087
            points_df.to_crs(epsg=4087, inplace=True)
            distance_diffs = points_df.distance(points_df.geometry.shift())
            distance_diffs[0] = distance_diffs[1]  # missing first datapoint, backfill
            # Issue: np.max(distance_diffs) = 3397 meters
            time_diffs_ns = np.append(0, (times[1:] - times[:-1]).astype(int))
            time_diffs_ns[0] = time_diffs_ns[1]  # missing first datapoint, backfill
            time_diffs_seconds = time_diffs_ns / self.NANOSECONDS_PER_SECOND
            # Calculate the speed in knots
            speed_meters_per_second = np.array(
                (distance_diffs / time_diffs_ns * self.NANOSECONDS_PER_SECOND),
                dtype=np.float32,
            )
            knots = self.meters_per_second_to_knots(speed_meters_per_second)
            metrics_df = pd.DataFrame(
                {
                    "speed_knots": knots.astype(dtype=np.float32),
                    "distance_meters": distance_diffs.to_numpy(dtype=np.float32),
                    "diff_seconds": time_diffs_seconds.astype(np.float32),
                },
                index=times,
            )
            #
            return metrics_df
        except Exception as err:
            raise RuntimeError(f"Exception encountered, {err}")

    #######################################################
    def add_speed_and_distance(
        self,
        ship_name,
        cruise_name,
        sensor_name,
        bucket_name,
        endpoint_url=None,
    ) -> None:
        try:
            zarr_manager = ZarrManager()
            zarr_store = zarr_manager.open_s3_zarr_store_with_zarr(
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                output_bucket_name=bucket_name,
                endpoint_url=endpoint_url,
            )
            longitudes = zarr_store["longitude"].values
            latitudes = zarr_store["latitude"].values
            times = zarr_store["time"].values
            #
            metrics_df = self.compute_speed_and_distance(times, latitudes, longitudes)
            # Write the speed and distance to the output zarr store
            zarr_store["speed"][:] = metrics_df.speed_knots.values
            zarr_store["distance"][:] = metrics_df.distance_meters.values
        except Exception as err:
            raise RuntimeError(
                f"Exception encountered writing the speed and distance, {err}"
            )


###########################################################
