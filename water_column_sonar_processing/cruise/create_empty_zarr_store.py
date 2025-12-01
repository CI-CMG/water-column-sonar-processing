import os
import tempfile

import numpy as np

from water_column_sonar_processing.utility import Constants
from water_column_sonar_processing.aws import DynamoDBManager, S3Manager
from water_column_sonar_processing.model import ZarrManager
from water_column_sonar_processing.utility import Cleaner


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
    @staticmethod
    def create_cruise_level_zarr_store(
        output_bucket_name: str,
        ship_name: str,
        cruise_name: str,
        sensor_name: str,
        table_name: str,
    ) -> None:
        """
        Initialize zarr store for the entire cruise which aggregates all the raw data.
        All cruises will be resampled at 20 cm depth.
        # tempdir="/tmp", # TODO: create better tmp directory for testing
        """
        tempdir = tempfile.TemporaryDirectory()
        try:
            dynamo_db_manager = DynamoDBManager()
            s3_manager = S3Manager()

            df = dynamo_db_manager.get_table_as_df(
                table_name=table_name,
                cruise_name=cruise_name,
            )

            # TODO: filter the dataframe just for enums >= LEVEL_1_PROCESSING
            # df[df['PIPELINE_STATUS'] < PipelineStatus.LEVEL_1_PROCESSING] = np.nan

            # TODO: VERIFY GEOJSON EXISTS as prerequisite!!! ...no more geojson needed

            print(f"DataFrame shape: {df.shape}")
            cruise_channels = list(
                set([i for sublist in df["CHANNELS"].dropna() for i in sublist])
            )
            cruise_channels.sort()

            consolidated_zarr_width = np.sum(
                df["NUM_PING_TIME_DROPNA"].dropna().astype(int)
            )

            # [4] max measurement resolution for the whole cruise
            cruise_max_echo_range = np.max(df["MAX_ECHO_RANGE"].dropna().astype(float))

            print(f"cruise_max_echo_range: {cruise_max_echo_range}")

            # [5] get number of channels
            cruise_frequencies = [
                float(i) for i in df["FREQUENCIES"].dropna().values.flatten()[0]
            ]

            new_width = int(consolidated_zarr_width)
            ################################################################
            # Delete any existing stores
            zarr_prefix = os.path.join(
                str(Constants.LEVEL_2.value), ship_name, cruise_name, sensor_name
            )
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
            zarr_manager.create_zarr_store(
                path=tempdir.name,
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                frequencies=cruise_frequencies,
                width=new_width,
                max_echo_range=cruise_max_echo_range,
                # cruise_min_epsilon=cruise_min_epsilon,
                calibration_status=True,
            )
            #################################################################
            # TODO: would be more elegant to create directly into s3 bucket
            s3_manager.upload_zarr_store_to_s3(
                output_bucket_name=output_bucket_name,
                local_directory=tempdir.name,
                object_prefix=zarr_prefix,
                cruise_name=cruise_name,
            )
            #################################################################
            # TODO: verify count of the files uploaded
            #################################################################
            # TODO: update enum in dynamodb
            print("Done creating cruise level zarr store.")
            #################################################################
        except Exception as err:
            raise RuntimeError(
                f"Problem trying to create new cruise model store, {err}"
            )
        finally:
            cleaner = Cleaner()
            cleaner.delete_local_files()
            # TODO: should delete zarr store in temp directory too?
        print("Done creating cruise level model store")


###########################################################
