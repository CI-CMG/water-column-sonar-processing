import numcodecs
import numpy as np
import xarray as xr
import zarr
import importlib.metadata
from numcodecs import Blosc

from water_column_sonar_processing.aws import S3FSManager
from water_column_sonar_processing.utility import Constants
from water_column_sonar_processing.utility import Timestamp
from water_column_sonar_processing.utility import Coordinates

numcodecs.blosc.use_threads = False
numcodecs.blosc.set_nthreads(1)


# TODO: when ready switch to version 3 of model spec
# ZARR_V3_EXPERIMENTAL_API = 1


# creates the latlon data: foo = ep.consolidate.add_location(ds_Sv, echodata)
class ZarrManager:
    #######################################################
    def __init__(
        self,
    ):
        # TODO: revert to Blosc.BITSHUFFLE, troubleshooting misc error
        self.__compressor = Blosc(cname="zstd", clevel=2)  # shuffle=Blosc.NOSHUFFLE
        self.__overwrite = True
        self.__num_threads = numcodecs.blosc.get_nthreads()
        # self.input_bucket_name = os.environ.get("INPUT_BUCKET_NAME")
        # self.output_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")

    #######################################################
    def get_depth_values(
        self,
        min_echo_range: float = 1.0,  # minimum depth measured (zero non-inclusive) from whole cruise
        max_echo_range: float = 100.0,  # maximum depth measured from whole cruise
    ):
        # Gets the set of depth values that will be used when resampling and
        # regridding the data to a cruise level model store.
        # Note: returned values do not start at zero.
        print("Getting depth values.")
        all_cruise_depth_values = np.linspace(
            start=min_echo_range,
            stop=max_echo_range,
            num=int(max_echo_range / min_echo_range) + 1,
            endpoint=True,
        )

        if np.any(np.isnan(all_cruise_depth_values)):
            raise Exception('Problem depth values returned were NaN.')

        print("Done getting depth values.")
        return all_cruise_depth_values.round(decimals=2)

    #######################################################
    def create_zarr_store(
        self,
        path: str, # 'level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.model/tmp/HB0707.zarr/.zattrs'
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        frequencies: list,  # units in Hz
        width: int,  # TODO: needs better name... "ping_time"
        min_echo_range: float,  # smallest resolution in meters
        max_echo_range: float,
        calibration_status: bool = False,  # Assume uncalibrated
    ) -> str:
        print(
            f"Creating local zarr_manager store at {cruise_name}.zarr for ship {ship_name}"
        )
        # There can not currently be repeated frequencies
        # TODO: eventually switch coordinate to "channel" because frequencies can repeat
        if len(frequencies) != len(set(frequencies)):
            raise Exception("Number of frequencies does not match number of channels")

        print(f"Debugging number of threads: {self.__num_threads}")

        zarr_path = f"{path}/{cruise_name}.zarr"
        store = zarr.DirectoryStore(path=zarr_path, normalize_keys=False)
        root = zarr.group(store=store, overwrite=self.__overwrite, cache_attrs=True)

        #####################################################################
        # --- Coordinate: Time --- #
        # https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding
        root.create_dataset(
            name=Coordinates.TIME.value,
            data=np.repeat(0.0, width),
            shape=width,
            chunks=Constants.SPATIOTEMPORAL_CHUNK_SIZE.value,
            dtype=np.dtype(Coordinates.TIME_DTYPE.value),
            compressor=self.__compressor,
            fill_value=np.nan,  # TODO: do i want nan's?
            overwrite=self.__overwrite,
        )

        root.time.attrs["_ARRAY_DIMENSIONS"] = [Coordinates.TIME.value]

        root.time.attrs["calendar"] = Coordinates.TIME_CALENDAR.value
        root.time.attrs["units"] = Coordinates.TIME_UNITS.value
        root.time.attrs["long_name"] = Coordinates.TIME_LONG_NAME.value
        root.time.attrs["standard_name"] = Coordinates.TIME_STANDARD_NAME.value

        #####################################################################
        # --- Coordinate: Depth --- #
        depth_values = self.get_depth_values(
            min_echo_range=min_echo_range, max_echo_range=max_echo_range
        )

        root.create_dataset(
            name=Coordinates.DEPTH.value,
            # TODO: verify that these values are correct
            data=depth_values,
            shape=len(depth_values),
            chunks=Constants.SPATIOTEMPORAL_CHUNK_SIZE.value,
            dtype=np.dtype(
                Coordinates.DEPTH_DTYPE.value
            ),  # float16 == 2 significant digits would be ideal
            compressor=self.__compressor,
            fill_value=np.nan,
            overwrite=self.__overwrite,
        )

        if np.any(np.isnan(depth_values)):
            raise Exception('Some depth values returned were NaN.')

        root.depth.attrs["_ARRAY_DIMENSIONS"] = [Coordinates.DEPTH.value]

        root.depth.attrs["units"] = Coordinates.DEPTH_UNITS.value
        root.depth.attrs["long_name"] = Coordinates.DEPTH_LONG_NAME.value
        root.depth.attrs["standard_name"] = Coordinates.DEPTH_STANDARD_NAME.value

        #####################################################################
        # --- Coordinate: Latitude --- #
        root.create_dataset(
            name=Coordinates.LATITUDE.value,
            # data=np.repeat(0.0, width),  # root.longitude[:] = np.nan
            data=np.repeat(np.nan, width),
            shape=width,
            chunks=Constants.SPATIOTEMPORAL_CHUNK_SIZE.value,
            dtype=np.dtype(Coordinates.LATITUDE_DTYPE.value),
            compressor=self.__compressor,
            fill_value=np.nan,
            overwrite=self.__overwrite,
        )

        # Note: LATITUDE is indexed by TIME
        root.latitude.attrs["_ARRAY_DIMENSIONS"] = [Coordinates.TIME.value]

        root.latitude.attrs["units"] = Coordinates.LATITUDE_UNITS.value
        root.latitude.attrs["long_name"] = Coordinates.LATITUDE_LONG_NAME.value
        root.latitude.attrs["standard_name"] = Coordinates.LATITUDE_STANDARD_NAME.value

        #####################################################################
        # --- Coordinate: Longitude --- #
        root.create_dataset(
            name=Coordinates.LONGITUDE.value,
            # data=np.repeat(0.0, width),  # root.longitude[:] = np.nan
            data=np.repeat(np.nan, width),
            shape=width,
            chunks=Constants.SPATIOTEMPORAL_CHUNK_SIZE.value,
            dtype=np.dtype(Coordinates.LONGITUDE_DTYPE.value),
            compressor=self.__compressor,
            fill_value=np.nan,
            overwrite=self.__overwrite,
        )

        # Note: LONGITUDE is indexed by TIME
        root.longitude.attrs["_ARRAY_DIMENSIONS"] = [Coordinates.TIME.value]

        root.longitude.attrs["units"] = Coordinates.LONGITUDE_UNITS.value
        root.longitude.attrs["long_name"] = Coordinates.LONGITUDE_LONG_NAME.value
        root.longitude.attrs["standard_name"] = Coordinates.LONGITUDE_STANDARD_NAME.value

        #####################################################################
        # TODO: verify adding this variable for where the bottom was detected
        # --- Coordinate: Bottom --- #
        root.create_dataset(
            name=Coordinates.BOTTOM.value,
            data=np.repeat(0.0, width),  # root.longitude[:] = np.nan
            shape=width,
            chunks=Constants.SPATIOTEMPORAL_CHUNK_SIZE.value,
            dtype=np.dtype(Coordinates.BOTTOM_DTYPE.value),
            compressor=self.__compressor,
            fill_value=0.0,
            overwrite=self.__overwrite,
        )

        # BOTTOM is indexed by TIME
        root.bottom.attrs["_ARRAY_DIMENSIONS"] = [Coordinates.TIME.value]

        root.bottom.attrs["units"] = Coordinates.BOTTOM_UNITS.value
        root.bottom.attrs["long_name"] = Coordinates.BOTTOM_LONG_NAME.value
        root.bottom.attrs["standard_name"] = Coordinates.BOTTOM_STANDARD_NAME.value

        #####################################################################
        # --- Coordinate: Frequency --- #
        root.create_dataset(
            name=Coordinates.FREQUENCY.value,
            data=frequencies,
            shape=len(frequencies),
            chunks=len(frequencies),
            dtype=np.dtype(Coordinates.FREQUENCY_DTYPE.value),
            compressor=self.__compressor,
            fill_value=0.0,
            overwrite=self.__overwrite,
        )

        # TODO: best coordinate would be channel with str type
        root.frequency.attrs["_ARRAY_DIMENSIONS"] = [
            Coordinates.FREQUENCY.value
        ]  # TODO: is this correct

        root.frequency.attrs["units"] = Coordinates.FREQUENCY_UNITS.value
        root.frequency.attrs["long_name"] = Coordinates.FREQUENCY_LONG_NAME.value
        root.frequency.attrs["standard_name"] = (
            Coordinates.FREQUENCY_STANDARD_NAME.value
        )

        #####################################################################
        # --- Sv Data --- #
        root.create_dataset(
            name=Coordinates.SV.value,
            shape=(len(depth_values), width, len(frequencies)),
            # chunks=(Constants.TILE_SIZE.value, Constants.TILE_SIZE.value, len(frequencies)),
            chunks=(Constants.TILE_SIZE.value, Constants.TILE_SIZE.value, 1), # 256x256x1 <- speed up for alex
            dtype=np.dtype(
                Coordinates.SV_DTYPE.value
            ),  # TODO: try to experiment with 'float16'
            compressor=self.__compressor,
            fill_value=np.nan,
            overwrite=self.__overwrite,
        )

        root.Sv.attrs["_ARRAY_DIMENSIONS"] = [
            Coordinates.DEPTH.value,
            Coordinates.TIME.value,
            Coordinates.FREQUENCY.value,
        ]

        root.Sv.attrs["units"] = Coordinates.SV_UNITS.value
        root.Sv.attrs["long_name"] = Coordinates.SV_LONG_NAME.value
        root.Sv.attrs["tile_size"] = Constants.TILE_SIZE.value

        #####################################################################
        # --- Metadata --- #
        root.attrs["ship_name"] = ship_name
        root.attrs["cruise_name"] = cruise_name
        root.attrs["sensor_name"] = sensor_name
        #
        root.attrs["processing_software_name"] = Coordinates.PROJECT_NAME.value

        current_project_version = importlib.metadata.version('water_column_sonar_processing')
        root.attrs["processing_software_version"] = current_project_version
        root.attrs["processing_software_time"] = Timestamp.get_timestamp()
        #
        root.attrs["calibration_status"] = calibration_status
        root.attrs["tile_size"] = Constants.TILE_SIZE.value

        zarr.consolidate_metadata(store)
        #####################################################################
        """
        # zzz = zarr.open('https://echofish-dev-master-118234403147-echofish-zarr-store.s3.us-west-2.amazonaws.com/GU1002_resample.zarr')
        # zzz.time[0] = 1274979445.423
        # Initialize all to origin time, will be overwritten late
        """
        return zarr_path

    ############################################################################
    # def update_zarr_store(
    #         self,
    #         path: str,
    #         ship_name: str,
    #         cruise_name: str,  # TODO: just pass stem
    #         sensor_name: str,
    # ) -> None:
    #     """
    #     Opens an existing Zarr store living in a s3 bucket for the purpose
    #     of updating just a subset of the cruise-level Zarr store associated
    #     with a file-level Zarr store.
    #     """
    #     pass

    ############################################################################
    def open_s3_zarr_store_with_zarr(
        self,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        # zarr_synchronizer: Union[str, None] = None, # TODO:
        output_bucket_name: str,
        endpoint_url=None,
    ) -> zarr.hierarchy.Group:
        # Mounts a Zarr store using pythons Zarr implementation. The mounted store
        #  will have read/write privileges so that store can be updated.
        print("Opening L2 Zarr store with Zarr for writing.")
        try:
            s3fs_manager = S3FSManager(endpoint_url=endpoint_url)
            root = f"{output_bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{cruise_name}.zarr"
            store = s3fs_manager.s3_map(s3_zarr_store_path=root)
            # synchronizer = model.ProcessSynchronizer(f"/tmp/{ship_name}_{cruise_name}.sync")
            cruise_zarr = zarr.open(store=store, mode="r+")
        except Exception as err:  # Failure
            print(f"Exception encountered opening Zarr store with Zarr.: {err}")
            raise
        print("Done opening Zarr store with Zarr.")
        return cruise_zarr

    ############################################################################
    def open_s3_zarr_store_with_xarray(
        self,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        file_name_stem: str,
        input_bucket_name: str,
        endpoint_url=None,
    ) -> xr.Dataset:
        print("Opening L1 Zarr store in S3 with Xarray.") # TODO: Is this only used for reading from?
        try:
            zarr_path = f"s3://{input_bucket_name}/level_1/{ship_name}/{cruise_name}/{sensor_name}/{file_name_stem}.zarr"
            s3fs_manager = S3FSManager(endpoint_url=endpoint_url)
            store_s3_map = s3fs_manager.s3_map(s3_zarr_store_path=zarr_path)
            ds = xr.open_dataset(
                filename_or_obj=store_s3_map,
                engine="zarr",
                chunks={}
            )
        except Exception as err:
            print("Problem opening Zarr store in S3 as Xarray.")
            raise err
        finally:
            print("Exiting opening Zarr store in S3 as Xarray.")
        return ds

    def open_l2_zarr_store_with_xarray(
        self,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        bucket_name: str,
        endpoint_url=None,
    ) -> xr.Dataset:
        print("Opening L2 Zarr store in S3 with Xarray.")
        try:
            zarr_path = f"s3://{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{cruise_name}.zarr"
            s3fs_manager = S3FSManager(endpoint_url=endpoint_url)
            store_s3_map = s3fs_manager.s3_map(s3_zarr_store_path=zarr_path)
            ds = xr.open_zarr(store=store_s3_map, consolidated=None)
        except Exception as err:
            print("Problem opening Zarr store in S3 as Xarray.")
            raise err
        print("Done opening Zarr store in S3 as Xarray.")
        return ds
    ############################################################################

    #######################################################
    # def create_process_synchronizer(self):
    #     # TODO: explore aws redis options
    #     pass

    #######################################################
    # def verify_cruise_store_data(self):
    #     # TODO: run a check on a finished model store to ensure that
    #     #   none of the time, latitude, longitude, or depth values
    #     #   are NaN.
    #     pass

    #######################################################


###########################################################
