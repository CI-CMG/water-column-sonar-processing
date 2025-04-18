import gc
import os
from datetime import datetime
from pathlib import Path  # , PurePath

import echopype as ep
import numcodecs
import numpy as np
from numcodecs import Blosc

from water_column_sonar_processing.aws import DynamoDBManager, S3Manager
from water_column_sonar_processing.geometry import GeometryManager
from water_column_sonar_processing.utility import Cleaner


# This code is getting copied from echofish-aws-raw-to-zarr-lambda
class RawToNetCDF:
    #######################################################
    def __init__(
        self,
        # output_bucket_access_key,
        # output_bucket_secret_access_key,
        # # overwrite_existing_zarr_store,
    ):
        # TODO: revert to Blosc.BITSHUFFLE, troubleshooting misc error
        self.__compressor = Blosc(cname="zstd", clevel=2)  # shuffle=Blosc.NOSHUFFLE
        self.__overwrite = True
        self.__num_threads = numcodecs.blosc.get_nthreads()
        # self.input_bucket_name = os.environ.get("INPUT_BUCKET_NAME")
        # self.output_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")
        # self.__table_name = table_name
        # # self.__overwrite_existing_zarr_store = overwrite_existing_zarr_store

    ############################################################################
    ############################################################################
    def __netcdf_info_to_table(
        self,
        # output_bucket_name,
        table_name,
        ship_name,
        cruise_name,
        sensor_name,
        file_name,
        # zarr_path,
        min_echo_range,
        max_echo_range,
        num_ping_time_dropna,
        start_time,
        end_time,
        frequencies,
        channels,
        water_level,
    ):
        print("Writing Zarr information to DynamoDB table.")
        dynamodb_manager = DynamoDBManager()
        dynamodb_manager.update_item(
            table_name=table_name,
            key={
                "FILE_NAME": {"S": file_name},  # Partition Key
                "CRUISE_NAME": {"S": cruise_name},  # Sort Key
            },
            expression_attribute_names={
                "#CH": "CHANNELS",
                "#ET": "END_TIME",
                # "#ED": "ERROR_DETAIL",
                "#FR": "FREQUENCIES",
                "#MA": "MAX_ECHO_RANGE",
                "#MI": "MIN_ECHO_RANGE",
                "#ND": "NUM_PING_TIME_DROPNA",
                # "#PS": "PIPELINE_STATUS",
                "#PT": "PIPELINE_TIME",
                "#SE": "SENSOR_NAME",
                "#SH": "SHIP_NAME",
                "#ST": "START_TIME",
                # "#ZB": "ZARR_BUCKET",
                # "#ZP": "ZARR_PATH",
                "#WL": "WATER_LEVEL",
            },
            expression_attribute_values={
                ":ch": {"L": [{"S": i} for i in channels]},
                ":et": {"S": end_time},
                # ":ed": {"S": ""},
                ":fr": {"L": [{"N": str(i)} for i in frequencies]},
                ":ma": {"N": str(np.round(max_echo_range, 4))},
                ":mi": {"N": str(np.round(min_echo_range, 4))},
                ":nd": {"N": str(num_ping_time_dropna)},
                # ":ps": {"S": "PROCESSING_RESAMPLE_AND_WRITE_TO_ZARR_STORE"},
                # ":ps": {"S": PipelineStatus.LEVEL_1_PROCESSING.name},
                ":pt": {"S": datetime.now().isoformat(timespec="seconds") + "Z"},
                ":se": {"S": sensor_name},
                ":sh": {"S": ship_name},
                ":st": {"S": start_time},
                ":wl": {"N": str(np.round(water_level, 2))},
                # ":zb": {"S": output_bucket_name},
                # ":zp": {"S": zarr_path},
            },
            update_expression=(
                "SET "
                "#CH = :ch, "
                "#ET = :et, "
                # "#ED = :ed, "
                "#FR = :fr, "
                "#MA = :ma, "
                "#MI = :mi, "
                "#ND = :nd, "
                # "#PS = :ps, "
                "#PT = :pt, "
                "#SE = :se, "
                "#SH = :sh, "
                "#ST = :st, "
                "#WL = :wl"
                # "#ZB = :zb, "
                # "#ZP = :zp"
            ),
        )
        print("Done writing Zarr information to DynamoDB table.")

    ############################################################################
    ############################################################################
    ############################################################################
    def __upload_files_to_output_bucket(
        self,
        output_bucket_name,
        local_directory,
        object_prefix,
        endpoint_url,
    ):
        # Note: this will be passed credentials if using NODD
        s3_manager = S3Manager(endpoint_url=endpoint_url)
        print("Uploading files using thread pool executor.")
        all_files = []
        for subdir, dirs, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(subdir, file)
                s3_key = os.path.join(object_prefix, local_path)
                all_files.append([local_path, s3_key])
        # all_files
        all_uploads = s3_manager.upload_files_with_thread_pool_executor(
            output_bucket_name=output_bucket_name,
            all_files=all_files,
        )
        return all_uploads

    def __upload_file_to_output_bucket(
        self,
        output_bucket_name,
        local_directory,
        object_prefix,
        endpoint_url,
    ):
        # Note: this will be passed credentials if using NODD
        s3_manager = S3Manager(endpoint_url=endpoint_url)
        print("Uploading files using thread pool executor.")
        all_files = [local_directory]
        all_uploads = s3_manager.upload_files_with_thread_pool_executor(
            output_bucket_name=output_bucket_name,
            all_files=all_files,
        )
        return all_uploads

    ############################################################################
    def raw_to_netcdf(
        self,
        table_name,
        input_bucket_name,
        output_bucket_name,
        ship_name,
        cruise_name,
        sensor_name,
        raw_file_name,
        endpoint_url=None,
        include_bot=True,
    ):
        """
        Downloads the raw files, processes them with echopype, and uploads files
        to the nodd bucket.

        Needs to create two files, one echopype opened file, one is Sv calibrated file
        """
        print(f"Opening raw: {raw_file_name} and creating netcdf.")
        try:
            geometry_manager = GeometryManager()
            cleaner = Cleaner()
            cleaner.delete_local_files(
                file_types=["*.nc", "*.json"]
            )  # TODO: include bot and raw?

            s3_manager = S3Manager(endpoint_url=endpoint_url)
            s3_file_path = (
                f"dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{raw_file_name}"
            )
            bottom_file_name = f"{Path(raw_file_name).stem}.bot"
            s3_bottom_file_path = f"dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{bottom_file_name}"
            s3_manager.download_file(
                bucket_name=input_bucket_name, key=s3_file_path, file_name=raw_file_name
            )
            # TODO: add the bottom file
            if include_bot:
                s3_manager.download_file(
                    bucket_name=input_bucket_name,
                    key=s3_bottom_file_path,
                    file_name=bottom_file_name,
                )

            gc.collect()
            print("Opening raw file with echopype.")
            # s3_file_path = f"s3://{bucket_name}/dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{file_name}"
            # s3_file_path = Path(f"s3://noaa-wcsd-pds/dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{file_name}")
            echodata = ep.open_raw(
                raw_file=raw_file_name,
                sonar_model=sensor_name,
                include_bot=include_bot,
            )

            netcdf_name = f"{Path(raw_file_name).stem}.nc"
            # Xarray Dataset to netcdf
            echodata.to_netcdf(
                save_path=netcdf_name,
                compress=True,
                overwrite=True,
            )

            print("Compute volume backscattering strength (Sv) from raw dataset.")
            ds_sv = ep.calibrate.compute_Sv(echodata)
            ds_sv = ep.consolidate.add_depth(
                ds_sv, echodata
            )  # TODO: consolidate with other depth values
            # water_level = ds_sv["water_level"].values
            gc.collect()
            print("Done computing volume backscatter strength (Sv) from raw dataset.")
            # Note: detected_seafloor_depth is located at echodata.vendor.detected_seafloor_depth
            # but is not written out with ds_sv
            if "detected_seafloor_depth" in list(echodata.vendor.variables):
                ds_sv["detected_seafloor_depth"] = (
                    echodata.vendor.detected_seafloor_depth
                )
            #
            # frequencies = echodata.environment.frequency_nominal.values
            #################################################################
            # Get GPS coordinates, just overwrite the lat lon values
            gps_data, lat, lon = geometry_manager.read_echodata_gps_data(
                echodata=echodata,
                output_bucket_name=output_bucket_name,
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                file_name=raw_file_name,
                endpoint_url=endpoint_url,
                write_geojson=False,
            )
            ds_sv = ep.consolidate.add_location(ds_sv, echodata)
            ds_sv.latitude.values = (
                lat  # overwriting echopype gps values to include missing values
            )
            ds_sv.longitude.values = lon
            # gps_data, lat, lon = self.__get_gps_data(echodata=echodata)

            # Create the netcdf
            netcdf_name_computed_Sv = f"{Path(raw_file_name).stem}_computed_Sv.nc"

            # Xarray Dataset to netcdf
            ds_sv.to_netcdf(
                path=netcdf_name_computed_Sv,
                mode="w",
            )
            gc.collect()
            #################################################################
            # output_netcdf_prefix = f"level_1/{ship_name}/{cruise_name}/{sensor_name}/"
            #################################################################
            # If netcdf already exists then delete
            s3_manager = S3Manager(endpoint_url=endpoint_url)
            child_objects = s3_manager.get_child_objects(
                bucket_name=output_bucket_name,
                sub_prefix=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/{Path(raw_file_name).stem}.nc",
            )
            if len(child_objects) > 0:
                print(
                    "NetCDF dataset already exists in s3, deleting existing and continuing."
                )
                s3_manager.delete_nodd_objects(
                    bucket_name=output_bucket_name,
                    objects=child_objects,
                )
            child_objects_computed_Sv = s3_manager.get_child_objects(
                bucket_name=output_bucket_name,
                sub_prefix=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/{Path(raw_file_name).stem}_computed_Sv.nc",
            )
            if len(child_objects_computed_Sv) > 0:
                print("data already exists in s3, deleting existing and continuing.")
                s3_manager.delete_nodd_objects(
                    bucket_name=output_bucket_name,
                    objects=child_objects_computed_Sv,
                )
            #################################################################
            s3_manager.upload_file(
                filename=netcdf_name,
                bucket_name=output_bucket_name,
                key=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/{Path(raw_file_name).stem}.nc",
            )
            s3_manager.upload_file(
                filename=netcdf_name_computed_Sv,
                bucket_name=output_bucket_name,
                key=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/{Path(raw_file_name).stem}_computed_Sv.nc",
            )
        except Exception as err:
            print(f"Exception encountered creating local netcdf with echopype: {err}")
            raise RuntimeError(f"Problem creating local netcdf, {err}")
        finally:
            gc.collect()
            cleaner.delete_local_files(
                file_types=["*.raw", "*.bot", "*.zarr", "*.nc", "*.json"]
            )
            print("Done creating local zarr store.")

    ############################################################################


################################################################################
############################################################################
