import os
import tempfile

import numcodecs
import numpy as np

from water_column_sonar_processing.aws import DynamoDBManager
from water_column_sonar_processing.aws import S3Manager
from water_column_sonar_processing.model import ZarrManager
from water_column_sonar_processing.utility import Cleaner

numcodecs.blosc.use_threads = False
numcodecs.blosc.set_nthreads(1)

# TODO: when ready switch to version 3 of model spec
# ZARR_V3_EXPERIMENTAL_API = 1
# creates the latlon data: foo = ep.consolidate.add_location(ds_Sv, echodata)


# TODO: change name to "CreateLocalEmptyZarrStore"
class CreateEmptyZarrStore:
    #######################################################
    def __init__(
        self,
    ):
        self.__overwrite = True
        # self.input_bucket_name = os.environ.get("INPUT_BUCKET_NAME")
        # self.output_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")

    #######################################################
    # TODO: move this to the s3_manager
    def upload_zarr_store_to_s3(
        self,
        output_bucket_name: str,
        local_directory: str,
        object_prefix: str,
        cruise_name: str,
    ) -> None:
        print("uploading model store to s3")
        s3_manager = S3Manager()
        #
        print("Starting upload with thread pool executor.")
        # # 'all_files' is passed a list of lists: [[local_path, s3_key], [...], ...]
        all_files = []
        for subdir, dirs, files in os.walk(
            f"{local_directory}/{cruise_name}.zarr"
        ):
            for file in files:
                local_path = os.path.join(subdir, file)
                # TODO: find a better method for splitting strings here:
                # 'level_2/Henry_B._Bigelow/HB0806/EK60/HB0806.zarr/.zattrs'
                s3_key = f'{object_prefix}/{cruise_name}.zarr{local_path.split(f"{cruise_name}.zarr")[-1]}'
                all_files.append([local_path, s3_key])
        #
        # print(all_files)
        s3_manager.upload_files_with_thread_pool_executor(
            output_bucket_name=output_bucket_name,
            all_files=all_files,
        )
        print("Done uploading with thread pool executor.")
        # TODO: move to common place

    #######################################################
    def create_cruise_level_zarr_store(
        self,
        output_bucket_name: str,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        table_name: str,
    ) -> None:
        tempdir = tempfile.TemporaryDirectory()
        try:
            # HB0806 - 123, HB0903 - 220
            dynamo_db_manager = DynamoDBManager()
            s3_manager = S3Manager()

            df = dynamo_db_manager.get_table_as_df(
                table_name=table_name,
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
            )

            # TODO: filter the dataframe just for enums >= LEVEL_1_PROCESSING
            # df[df['PIPELINE_STATUS'] < PipelineStatus.LEVEL_1_PROCESSING] = np.nan

            # TODO: VERIFY GEOJSON EXISTS as prerequisite!!!

            print(f"DataFrame shape: {df.shape}")
            cruise_channels = list(
                set([i for sublist in df["CHANNELS"].dropna() for i in sublist])
            )
            cruise_channels.sort()

            consolidated_zarr_width = np.sum(
                df["NUM_PING_TIME_DROPNA"].dropna().astype(int)
            )

            # [3] calculate the max/min measurement resolutions for the whole cruise
            cruise_min_echo_range = float(
                np.min(df["MIN_ECHO_RANGE"].dropna().astype(float))
            )

            # [4] calculate the maximum of the max depth values
            cruise_max_echo_range = float(
                np.max(df["MAX_ECHO_RANGE"].dropna().astype(float))
            )
            print(
                f"cruise_min_echo_range: {cruise_min_echo_range}, cruise_max_echo_range: {cruise_max_echo_range}"
            )

            # [5] get number of channels
            cruise_frequencies = [
                float(i) for i in df["FREQUENCIES"].dropna().values.flatten()[0]
            ]
            print(cruise_frequencies)

            new_width = int(consolidated_zarr_width)
            print(f"new_width: {new_width}")
            #################################################################
            store_name = f"{cruise_name}.zarr"
            print(store_name)
            ################################################################
            # Delete existing model store if it exists
            zarr_prefix = os.path.join("level_2", ship_name, cruise_name, sensor_name)
            child_objects = s3_manager.get_child_objects(
                bucket_name=output_bucket_name,
                sub_prefix=zarr_prefix,
            )
            if len(child_objects) > 0:
                s3_manager.delete_nodd_objects(
                    bucket_name=output_bucket_name,
                    objects=child_objects,
                )
            ################################################################
            # Create new model store
            zarr_manager = ZarrManager()
            new_height = len(
                zarr_manager.get_depth_values(
                    min_echo_range=cruise_min_echo_range,
                    max_echo_range=cruise_max_echo_range,
                )
            )
            print(f"new_height: {new_height}")

            zarr_manager.create_zarr_store(
                path=tempdir.name, # TODO: need to use .name or problem
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                frequencies=cruise_frequencies,
                width=new_width,
                min_echo_range=cruise_min_echo_range,
                max_echo_range=cruise_max_echo_range,
                calibration_status=True,
            )
            #################################################################
            self.upload_zarr_store_to_s3(
                output_bucket_name=output_bucket_name,
                local_directory=tempdir.name, # TODO: need to use .name or problem
                object_prefix=zarr_prefix,
                cruise_name=cruise_name,
            )
            # https://noaa-wcsd-zarr-pds.s3.amazonaws.com/index.html
            #################################################################
            # Verify count of the files uploaded
            # count = self.__get_file_count(store_name=store_name)
            # #
            # raw_zarr_files = self.__get_s3_files(  # TODO: just need count
            #     bucket_name=self.__output_bucket,
            #     sub_prefix=os.path.join(zarr_prefix, store_name),
            # )
            # if len(raw_zarr_files) != count:
            #     print(f'Problem writing {store_name} with proper count {count}.')
            #     raise Exception("File count doesnt equal number of s3 Zarr store files.")
            # else:
            #     print("File counts match.")
            #################################################################
            # Success
            # TODO: update enum in dynamodb
            print("Done creating cruise level zarr store.")
            #################################################################
        except Exception as err:
            print(f"Problem trying to create new cruise model store: {err}")
        finally:
            cleaner = Cleaner()
            cleaner.delete_local_files()
            # TODO: should delete zarr store in temp directory too?
        print("Done creating cruise level model store")


###########################################################
