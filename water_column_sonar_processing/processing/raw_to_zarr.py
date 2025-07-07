import gc
import os
from datetime import datetime
from pathlib import Path

import echopype as ep
import numcodecs
import numpy as np
from numcodecs import Blosc

from water_column_sonar_processing.aws import DynamoDBManager, S3Manager
from water_column_sonar_processing.geometry import GeometryManager
from water_column_sonar_processing.utility import Cleaner


def get_water_level(ds):
    """
    needs to be mocked up so thats why this is broken out
    """
    if "water_level" in ds.keys():
        return ds.water_level.values
    else:
        return 0.0


# This code is getting copied from echofish-aws-raw-to-zarr-lambda
class RawToZarr:
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
    def __zarr_info_to_table(
        self,
        table_name,
        ship_name,
        cruise_name,
        sensor_name,  # : Constants, TODO: convert to enum
        file_name,
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
                "#PT": "PIPELINE_TIME",
                "#SE": "SENSOR_NAME",
                "#SH": "SHIP_NAME",
                "#ST": "START_TIME",
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
                ":pt": {"S": datetime.now().isoformat(timespec="seconds") + "Z"},
                ":se": {"S": sensor_name},
                ":sh": {"S": ship_name},
                ":st": {"S": start_time},
                ":wl": {"N": str(np.round(water_level, 2))},
            },
            update_expression=(
                "SET "
                "#CH = :ch, "
                "#ET = :et, "
                "#FR = :fr, "
                "#MA = :ma, "
                "#MI = :mi, "
                "#ND = :nd, "
                "#PT = :pt, "
                "#SE = :se, "
                "#SH = :sh, "
                "#ST = :st, "
                "#WL = :wl"
            ),
        )
        print("Done writing Zarr information to DynamoDB table.")

    ############################################################################
    ############################################################################
    ############################################################################
    def __upload_files_to_output_bucket(
        self,
        output_bucket_name: str,
        local_directory: str,  # e.g. 'D20070724-T042400.zarr'  # TODO: problem: if this is not in the current directory
        object_prefix: str,  # e.g. "level_1/Henry_B._Bigelow/HB0706/EK60/"
        endpoint_url,
    ):
        # Note: this will be passed credentials if using NODD
        # TODO: this will not work if the local_directory is anywhere other than the current folder
        # see test_s3_manager test_upload...pool_executor for solution
        s3_manager = S3Manager(endpoint_url=endpoint_url)
        print("Uploading files using thread pool executor.")
        all_files = []
        for subdir, dirs, files in os.walk(
            local_directory
        ):  # os.path.basename(s3_manager_test_path.joinpath("HB0707.zarr/"))
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

    ############################################################################

    ############################################################################
    def raw_to_zarr(
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
        Downloads the raw files, processes them with echopype, writes geojson, and uploads files
        to the nodd bucket.
        """
        print(f"Opening raw: {raw_file_name} and creating zarr store.")
        geometry_manager = GeometryManager()
        cleaner = Cleaner()
        cleaner.delete_local_files(
            file_types=["*.zarr", "*.json"]
        )  # TODO: include bot and raw?

        s3_manager = S3Manager(endpoint_url=endpoint_url)
        s3_file_path = (
            f"dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{raw_file_name}"
        )
        bottom_file_name = f"{Path(raw_file_name).stem}.bot"
        s3_bottom_file_path = (
            f"dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{bottom_file_name}"
        )
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

        try:
            gc.collect()
            print("Opening raw file with echopype.")
            # s3_file_path = f"s3://{bucket_name}/dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{file_name}"
            # s3_file_path = Path(f"s3://noaa-wcsd-pds/dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{file_name}")
            echodata = ep.open_raw(
                raw_file=raw_file_name,
                sonar_model=sensor_name,
                include_bot=include_bot,
                # use_swap=True,
                # max_chunk_size=300,
                # storage_options={'anon': True } # 'endpoint_url': self.endpoint_url} # this was creating problems
            )
            print("Compute volume backscattering strength (Sv) from raw dataset.")
            ds_sv = ep.calibrate.compute_Sv(echodata)
            ds_sv = ep.consolidate.add_depth(
                ds_sv, echodata
            )  # TODO: consolidate with other depth values

            water_level = get_water_level(ds_sv)

            gc.collect()
            print("Done computing volume backscatter strength (Sv) from raw dataset.")
            # Note: detected_seafloor_depth is located at echodata.vendor.detected_seafloor_depth
            # but is not written out with ds_sv
            if "detected_seafloor_depth" in list(echodata.vendor.variables):
                ds_sv["detected_seafloor_depth"] = (
                    echodata.vendor.detected_seafloor_depth
                )
            #
            frequencies = echodata.environment.frequency_nominal.values
            #################################################################
            # Get GPS coordinates
            gps_data, lat, lon = geometry_manager.read_echodata_gps_data(
                echodata=echodata,
                output_bucket_name=output_bucket_name,
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                file_name=raw_file_name,
                endpoint_url=endpoint_url,
                write_geojson=True,
            )
            ds_sv = ep.consolidate.add_location(ds_sv, echodata)
            ds_sv.latitude.values = (
                lat  # overwriting echopype gps values to include missing values
            )
            ds_sv.longitude.values = lon
            # gps_data, lat, lon = self.__get_gps_data(echodata=echodata)
            #################################################################
            # Technically the min_echo_range would be 0 m.
            # TODO: this var name is supposed to represent minimum resolution of depth measurements
            # TODO revert this so that smaller diffs can be used
            # The most minimum the resolution can be is as small as 0.25 meters
            min_echo_range = np.round(np.nanmin(np.diff(ds_sv.echo_range.values)), 2)
            # For the HB0710 cruise the depths vary from 499.7215 @19cm to 2999.4805 @ 1cm. Moving that back
            # inline with the
            min_echo_range = np.max(
                [0.20, min_echo_range]
            )  # TODO: experiment with 0.25 and 0.50

            max_echo_range = float(np.nanmax(ds_sv.echo_range))

            # This is the number of missing values found throughout the lat/lon
            num_ping_time_dropna = lat[~np.isnan(lat)].shape[0]  # symmetric to lon
            #
            start_time = (
                np.datetime_as_string(ds_sv.ping_time.values[0], unit="ms") + "Z"
            )
            end_time = (
                np.datetime_as_string(ds_sv.ping_time.values[-1], unit="ms") + "Z"
            )
            channels = list(ds_sv.channel.values)
            #
            #################################################################
            # Create the zarr store
            store_name = f"{Path(raw_file_name).stem}.zarr"
            # Sv = ds_sv.Sv
            # ds_sv['Sv'] = Sv.astype('int32', copy=False)
            ds_sv.to_zarr(
                store=store_name
            )  # ds_sv.Sv.sel(channel=ds_sv.channel.values[0]).shape
            gc.collect()
            #################################################################
            output_zarr_prefix = f"level_1/{ship_name}/{cruise_name}/{sensor_name}/"
            #################################################################
            # If zarr store already exists then delete
            s3_manager = S3Manager(endpoint_url=endpoint_url)
            child_objects = s3_manager.get_child_objects(
                bucket_name=output_bucket_name,
                sub_prefix=f"level_1/{ship_name}/{cruise_name}/{sensor_name}/{Path(raw_file_name).stem}.zarr",
            )
            if len(child_objects) > 0:
                print(
                    "Zarr store dataset already exists in s3, deleting existing and continuing."
                )
                s3_manager.delete_nodd_objects(
                    bucket_name=output_bucket_name,
                    objects=child_objects,
                )
            #################################################################
            self.__upload_files_to_output_bucket(
                output_bucket_name=output_bucket_name,
                local_directory=store_name,
                object_prefix=output_zarr_prefix,
                endpoint_url=endpoint_url,
            )
            #################################################################
            self.__zarr_info_to_table(
                table_name=table_name,
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                file_name=raw_file_name,
                min_echo_range=min_echo_range,
                max_echo_range=max_echo_range,
                num_ping_time_dropna=num_ping_time_dropna,
                start_time=start_time,
                end_time=end_time,
                frequencies=frequencies,
                channels=channels,
                water_level=water_level,
            )
            #######################################################################
            # TODO: verify count of objects matches, publish message, update status
            #######################################################################
            print("Finished raw-to-zarr conversion.")
        except Exception as err:
            print(
                f"Exception encountered creating local Zarr store with echopype: {err}"
            )
            raise RuntimeError(f"Problem creating local Zarr store, {err}")
        finally:
            gc.collect()
            print("Finally.")
            cleaner.delete_local_files(
                file_types=["*.raw", "*.bot", "*.zarr", "*.json"]
            )
        print("Done creating local zarr store.")

    ############################################################################
    # TODO: does this get called?
    # def execute(self, input_message):
    #     ship_name = input_message['shipName']
    #     cruise_name = input_message['cruiseName']
    #     sensor_name = input_message['sensorName']
    #     input_file_name = input_message['fileName']
    #     #
    #     try:
    #         self.__update_processing_status(
    #             file_name=input_file_name,
    #             cruise_name=cruise_name,
    #             pipeline_status="PROCESSING_RAW_TO_ZARR"
    #         )
    #         #######################################################################
    #         store_name = f"{os.path.splitext(input_file_name)[0]}.zarr"
    #         output_zarr_prefix = f"level_1/{ship_name}/{cruise_name}/{sensor_name}"
    #         bucket_key = f"dataset/raw/{ship_name}/{cruise_name}/{sensor_name}/{input_file_name}"
    #         zarr_prefix = os.path.join("level_1", ship_name, cruise_name, sensor_name)
    #         #
    #         os.chdir(TEMPDIR)  # Lambdas require use of temp directory
    #         #######################################################################
    #         #######################################################################
    #         # Check if zarr store already exists
    #         s3_objects = self.__s3.list_objects(
    #             bucket_name=self.__output_bucket,
    #             prefix=f"{zarr_prefix}/{os.path.splitext(input_file_name)[0]}.zarr/",
    #             access_key_id=self.__output_bucket_access_key,
    #             secret_access_key=self.__output_bucket_secret_access_key
    #         )
    #         if len(s3_objects) > 0:
    #             print('Zarr store dataset already exists in s3, deleting existing and continuing.')
    #             self.__s3.delete_objects(
    #                 bucket_name=self.__output_bucket,
    #                 objects=s3_objects,
    #                 access_key_id=self.__output_bucket_access_key,
    #                 secret_access_key=self.__output_bucket_secret_access_key
    #             )
    #         #######################################################################
    #         # self.__delete_all_local_raw_and_zarr_files()
    #         Cleaner.delete_local_files(file_types=["*.raw*", "*.zarr"])
    #         self.__s3.download_file(
    #             bucket_name=self.__input_bucket,
    #             key=bucket_key,
    #             file_name=input_file_name
    #         )
    #         self.__create_local_zarr_store(
    #             raw_file_name=input_file_name,
    #             cruise_name=cruise_name,
    #             sensor_name=sensor_name,
    #             output_zarr_prefix=output_zarr_prefix,
    #             store_name=store_name
    #         )
    #         #######################################################################
    #         self.__upload_files_to_output_bucket(store_name, output_zarr_prefix)
    #         #######################################################################
    #         # # TODO: verify count of objects matches
    #         # s3_objects = self.__s3.list_objects(
    #         #     bucket_name=self.__output_bucket,
    #         #     prefix=f"{zarr_prefix}/{os.path.splitext(input_file_name)[0]}.zarr/",
    #         #     access_key_id=self.__output_bucket_access_key,
    #         #     secret_access_key=self.__output_bucket_secret_access_key
    #         # )
    #         #######################################################################
    #         self.__update_processing_status(
    #             file_name=input_file_name,
    #             cruise_name=cruise_name,
    #             pipeline_status='SUCCESS_RAW_TO_ZARR'
    #         )
    #         #######################################################################
    #         self.__publish_done_message(input_message)
    #         #######################################################################
    #     # except Exception as err:
    #     #     print(f'Exception encountered: {err}')
    #     # self.__update_processing_status(
    #     #     file_name=input_file_name,
    #     #     cruise_name=cruise_name,
    #     #     pipeline_status='FAILURE_RAW_TO_ZARR',
    #     #     error_message=str(err),
    #     # )
    #     finally:
    #         self.__delete_all_local_raw_and_zarr_files()
    #######################################################################

    ############################################################################


################################################################################
############################################################################
