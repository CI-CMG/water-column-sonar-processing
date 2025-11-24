import importlib.metadata
import os
from typing import Optional

import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.core.group import Group

from water_column_sonar_processing.utility import Constants, Coordinates, Timestamp

compressor = BloscCodec(  # https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
    typesize=4,
    cname="zstd",
    clevel=5,
    shuffle=BloscShuffle.shuffle,
    blocksize=0,
)


# creates the latlon dataset: foo = ep.consolidate.add_location(ds_Sv, echodata)
class ZarrManager:
    #######################################################
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
    ):
        self.__overwrite = True
        # self.endpoint_url = endpoint_url
        self.key = os.environ.get("OUTPUT_BUCKET_ACCESS_KEY")
        self.secret = os.environ.get("OUTPUT_BUCKET_SECRET_ACCESS_KEY")

        # self.ship_name = ship_name

    #######################################################
    def get_depth_values(
        self,
        max_echo_range: float,  # maximum depth measured from whole cruise
        cruise_min_epsilon: float = 0.25,  # resolution between subsequent measurements
    ):  # TODO: define return type
        # Gets the set of depth values that will be used when resampling and
        # regridding the dataset to a cruise level model store.
        # Note: returned values start at zero!
        # For more info see here: https://echopype.readthedocs.io/en/stable/data-proc-additional.html
        print("Computing depth values.")
        all_cruise_depth_values = np.linspace(  # TODO: PROBLEM HERE
            start=0,  # just start it at zero
            stop=max_echo_range,
            num=int(max_echo_range / cruise_min_epsilon)
            + 1,  # int(np.ceil(max_echo_range / cruise_min_epsilon))?
            endpoint=True,
        )  # np.arange(min_echo_range, max_echo_range, step=min_echo_range) # this is worse

        if np.any(np.isnan(all_cruise_depth_values)):
            raise Exception("Problem depth values returned were NaN.")

        print("Done computing depth values.")
        return all_cruise_depth_values.round(decimals=2)

    #######################################################
    def create_zarr_store(
        self,
        path: str,  # 'level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.model/tmp/HB0707.zarr/.zattrs'
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        frequencies: list,  # units in Hz
        width: int,  # TODO: needs better name... "ping_time"
        max_echo_range: float,
        cruise_min_epsilon: float,  # smallest resolution in meters
        calibration_status: bool = False,  # Assume uncalibrated
    ) -> str:
        """
        Creates a new zarr store in a local temporary directory(?)
        """
        try:
            print(f"Creating local zarr store, {cruise_name}.zarr for ship {ship_name}")
            if len(frequencies) != len(set(frequencies)):
                raise Exception(
                    "Number of frequencies does not match number of channels"
                )

            zarr_path = f"{path}/{cruise_name}.zarr"
            #####################################################################
            # Colab notebook showing how DataSet is built for Zarr export
            # https://colab.research.google.com/drive/1r3R4DxPR791paFohnUXy_z10fdlfSJYu?usp=sharing
            #####################################################################

            ##### Depth #####
            depth_data_values = self.get_depth_values(
                max_echo_range=max_echo_range,
                cruise_min_epsilon=cruise_min_epsilon,
            )
            if np.any(np.isnan(depth_data_values)):
                raise Exception("Some depth values returned were NaN.")

            depth_data = np.array(
                depth_data_values, dtype=Coordinates.DEPTH_DTYPE.value
            )
            depth_da = xr.DataArray(
                data=depth_data,
                dims=Coordinates.DEPTH.value,
                name=Coordinates.DEPTH.value,
                attrs=dict(
                    units=Coordinates.DEPTH_UNITS.value,
                    long_name=Coordinates.DEPTH_LONG_NAME.value,
                    standard_name=Coordinates.DEPTH_STANDARD_NAME.value,
                ),
            )

            ##### Time #####
            # https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding
            time_data = np.array(
                np.repeat(np.datetime64(0, "ns"), width),
                dtype="datetime64[ns]",
            )
            time_da = xr.DataArray(
                data=time_data,
                dims=Coordinates.TIME.value,
                name=Coordinates.TIME.value,
                attrs=dict(
                    # Note: cal & units are written by xr
                    # calendar="proleptic_gregorian",
                    # units="seconds since 1970-01-01 00:00:00",
                    long_name=Coordinates.TIME_LONG_NAME.value,
                    standard_name=Coordinates.TIME_STANDARD_NAME.value,
                ),
            )

            ##### Frequency #####
            frequency_data = np.array(
                frequencies,
                dtype=np.dtype(Coordinates.FREQUENCY_DTYPE.value),
            )
            frequency_da = xr.DataArray(
                data=frequency_data,
                dims=Coordinates.FREQUENCY.value,
                name=Coordinates.FREQUENCY.value,
                attrs=dict(
                    units=Coordinates.FREQUENCY_UNITS.value,
                    long_name=Coordinates.FREQUENCY_LONG_NAME.value,
                    standard_name=Coordinates.FREQUENCY_STANDARD_NAME.value,
                ),
            )

            ##### Latitude #####
            gps_data = np.array(
                np.repeat(np.nan, width),
                dtype=np.dtype(Coordinates.LATITUDE_DTYPE.value),
            )
            latitude_da = xr.DataArray(
                data=gps_data,
                coords=dict(
                    time=time_da,
                ),
                dims=Coordinates.TIME.value,  # Note: "TIME"
                name=Coordinates.LATITUDE.value,
                attrs=dict(
                    units=Coordinates.LATITUDE_UNITS.value,
                    long_name=Coordinates.LATITUDE_LONG_NAME.value,
                    standard_name=Coordinates.LATITUDE_STANDARD_NAME.value,
                ),
            )  # Note: LATITUDE is indexed by TIME

            ##### Longitude #####
            longitude_da = xr.DataArray(
                data=gps_data,
                coords=dict(
                    time=time_da,
                ),
                dims=Coordinates.TIME.value,  # Note: "TIME"
                name=Coordinates.LONGITUDE.value,
                attrs=dict(
                    units=Coordinates.LONGITUDE_UNITS.value,
                    long_name=Coordinates.LONGITUDE_LONG_NAME.value,
                    standard_name=Coordinates.LONGITUDE_STANDARD_NAME.value,
                ),
            )  # Note: LONGITUDE is indexed by TIME

            ##### Bottom #####
            bottom_data = np.array(
                np.repeat(np.nan, width), dtype=np.dtype(Coordinates.BOTTOM_DTYPE.value)
            )
            bottom_da = xr.DataArray(
                data=bottom_data,
                coords=dict(
                    time=time_da,
                ),
                dims=Coordinates.TIME.value,  # Note: "TIME"
                name=Coordinates.BOTTOM.value,
                attrs=dict(
                    units=Coordinates.BOTTOM_UNITS.value,
                    long_name=Coordinates.BOTTOM_LONG_NAME.value,
                    standard_name=Coordinates.BOTTOM_STANDARD_NAME.value,
                ),
            )

            ##### Speed #####
            speed_data = np.array(
                np.repeat(np.nan, width), dtype=np.dtype(Coordinates.SPEED_DTYPE.value)
            )
            speed_da = xr.DataArray(
                data=speed_data,
                coords=dict(
                    time=time_da,
                ),
                dims=Coordinates.TIME.value,  # Note: "TIME"
                name=Coordinates.SPEED.value,
                attrs=dict(
                    units=Coordinates.SPEED_UNITS.value,
                    long_name=Coordinates.SPEED_LONG_NAME.value,
                    standard_name=Coordinates.SPEED_STANDARD_NAME.value,
                ),
            )

            ##### Distance #####
            distance_data = np.array(
                np.repeat(np.nan, width),
                dtype=np.dtype(Coordinates.DISTANCE_DTYPE.value),
            )
            distance_da = xr.DataArray(
                data=distance_data,
                coords=dict(
                    time=time_da,
                ),
                dims=Coordinates.TIME.value,  # Note: "TIME"
                name=Coordinates.DISTANCE.value,
                attrs=dict(
                    units=Coordinates.DISTANCE_UNITS.value,
                    long_name=Coordinates.DISTANCE_LONG_NAME.value,
                    standard_name=Coordinates.DISTANCE_STANDARD_NAME.value,
                ),
            )

            ##### Sv #####
            sv_data = np.empty(
                (len(depth_data), width, len(frequencies)),
                dtype=np.dtype(Coordinates.SV_DTYPE.value),
            )
            # print(sv_data.shape)
            sv_da = xr.DataArray(
                data=sv_data,
                coords=dict(
                    depth=depth_da,
                    time=time_da,
                    frequency=frequency_da,
                    #
                    latitude=latitude_da,
                    longitude=longitude_da,
                    bottom=bottom_da,
                    speed=speed_da,
                    distance=distance_da,
                ),
                dims=(  # Depth * Time * Frequency
                    Coordinates.DEPTH.value,
                    Coordinates.TIME.value,
                    Coordinates.FREQUENCY.value,
                ),
                name=Coordinates.SV.value,
                attrs=dict(
                    units=Coordinates.SV_UNITS.value,
                    long_name=Coordinates.SV_LONG_NAME.value,
                    standard_name=Coordinates.SV_STANDARD_NAME.value,
                    tiles_size=Constants.TILE_SIZE.value,
                ),
            )
            #####################################################################
            ### Now create the xarray.Dataset
            ds = xr.Dataset(
                data_vars=dict(
                    Sv=sv_da,
                ),
                coords=dict(
                    depth=depth_da,
                    time=time_da,
                    frequency=frequency_da,
                    #
                    latitude=latitude_da,
                    longitude=longitude_da,
                    bottom=bottom_da,
                    speed=speed_da,
                    distance=distance_da,
                ),
                attrs=dict(
                    # --- Metadata --- #
                    ship_name=ship_name,
                    cruise_name=cruise_name,
                    sensor_name=sensor_name,
                    processing_software_name=Coordinates.PROJECT_NAME.value,
                    # NOTE: for the version to be parsable you need to build the python package
                    #  locally first.
                    processing_software_version=importlib.metadata.version(
                        "water-column-sonar-processing"
                    ),
                    processing_software_time=Timestamp.get_timestamp(),
                    calibration_status=calibration_status,
                    tile_size=Constants.TILE_SIZE.value,
                ),
            )

            #####################################################################
            # Define the chunk sizes and the encoding
            spatiotemporal_chunk_size = int(
                1e6
            )  # 1_000_000 data points for quickest download
            depth_chunk_shape = (512,)  # TODO: parameterize
            time_chunk_shape = (spatiotemporal_chunk_size,)
            frequency_chunk_shape = (len(frequency_data),)
            latitude_chunk_shape = (spatiotemporal_chunk_size,)
            longitude_chunk_shape = (spatiotemporal_chunk_size,)
            bottom_chunk_shape = (spatiotemporal_chunk_size,)
            speed_chunk_shape = (spatiotemporal_chunk_size,)
            distance_chunk_shape = (spatiotemporal_chunk_size,)
            sv_chunk_shape = (512, 512, 1)  # TODO: move to constants

            encodings = dict(
                depth={
                    "compressors": [compressor],
                    "chunks": depth_chunk_shape,
                },
                time={
                    "compressors": [compressor],
                    "chunks": time_chunk_shape,
                    "units": Coordinates.TIME_UNITS.value,
                },
                frequency={
                    "compressors": [compressor],
                    "chunks": frequency_chunk_shape,
                },
                latitude={
                    "compressors": [compressor],
                    "chunks": latitude_chunk_shape,
                },
                longitude={
                    "compressors": [compressor],
                    "chunks": longitude_chunk_shape,
                },
                bottom={
                    "compressors": [compressor],
                    "chunks": bottom_chunk_shape,
                },
                speed={
                    "compressors": [compressor],
                    "chunks": speed_chunk_shape,
                },
                distance={
                    "compressors": [compressor],
                    "chunks": distance_chunk_shape,
                },
                Sv={
                    "compressors": [compressor],
                    "chunks": sv_chunk_shape,
                },
            )

            ds.to_zarr(
                store=zarr_path,
                mode="w",  # “w” means create (overwrite if exists)
                encoding=encodings,
                consolidated=False,
                safe_chunks=True,
                zarr_format=3,
                write_empty_chunks=False,  # Might need to change this
            )
            # print(ds)
            #####################################################################
            return zarr_path
        except Exception as err:
            raise RuntimeError(f"Problem trying to create zarr store, {err}")
        # finally:
        #     cleaner = Cleaner()
        #     cleaner.delete_local_files()
        # TODO: should delete zarr store in temp directory too?

    ############################################################################
    def open_s3_zarr_store_with_zarr(
        self,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        output_bucket_name: str,
        endpoint_url: Optional[str] = None,
    ) -> Group:
        # Mounts a Zarr store using pythons Zarr implementation. The mounted store
        #  will have read/write privileges so that store can be updated.
        print("Opening L2 Zarr store with Zarr for writing.")
        try:
            store = f"s3://{output_bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{cruise_name}.zarr"
            print(f"endpoint url: {endpoint_url}")
            cruise_zarr = zarr.open(
                store=store,
                mode="r+",
                zarr_format=3,
                storage_options={
                    "endpoint_url": endpoint_url,
                    "key": self.key,
                    "secret": self.secret,
                },
            )
            print("Done opening store with Zarr.")
            return cruise_zarr
        except Exception as err:  # Failure
            raise RuntimeError(f"Exception encountered opening store with Zarr, {err}")

    ###########################################################################
    def open_s3_zarr_store_with_xarray(
        self,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        file_name_stem: str,
        input_bucket_name: str,
        # level: str, # TODO: add level
        endpoint_url: Optional[str] = None,  # needed for moto testing
    ) -> xr.Dataset:
        print("Opening L1 Zarr store in S3 with Xarray.")
        try:
            zarr_path = f"s3://{input_bucket_name}/level_1/{ship_name}/{cruise_name}/{sensor_name}/{file_name_stem}.zarr"
            kwargs = {"consolidated": False}
            ds = xr.open_dataset(
                filename_or_obj=zarr_path,
                engine="zarr",
                backend_kwargs={
                    "storage_options": {
                        "endpoint_url": endpoint_url,
                        "anon": True,
                    },
                },
                **kwargs,
            )
            return ds
        except Exception as err:
            raise RuntimeError(f"Problem opening Zarr store in S3 as Xarray, {err}")
        # finally:
        #     print("Exiting opening Zarr store in S3 as Xarray.")

    ###########################################################################
    # TODO: can this be consolidated with above
    def open_l2_zarr_store_with_xarray(
        self,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        bucket_name: str,
        # level: str, # TODO: add level
        endpoint_url: Optional[str] = None,  # needed for moto testing
    ) -> xr.Dataset:
        print("Opening L2 Zarr store in S3 with Xarray.")
        try:
            zarr_path = f"s3://{bucket_name}/level_2/{ship_name}/{cruise_name}/{sensor_name}/{cruise_name}.zarr"
            kwargs = {"consolidated": False}
            ds = xr.open_dataset(
                filename_or_obj=zarr_path,
                engine="zarr",
                backend_kwargs={
                    "storage_options": {
                        "endpoint_url": endpoint_url,
                        "anon": True,
                    }
                },
                **kwargs,
            )
            return ds  # TODO: Assert that you open it anonymously
        except Exception as err:
            raise RuntimeError(f"Problem opening Zarr store in S3 as Xarray, {err}")

    ###########################################################################

    ###########################################################################
    # def create_process_synchronizer(self):
    #     # TODO: explore aws redis options
    #     pass

    ###########################################################################
    # def verify_cruise_store_data(self):
    #     # TODO: run a check on a finished model store to ensure that
    #     #   none of the time, latitude, longitude, or depth values
    #     #   are NaN.
    #     pass

    ###########################################################################


###########################################################
