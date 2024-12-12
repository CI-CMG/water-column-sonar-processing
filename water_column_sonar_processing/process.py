import json
import os

import numpy as np

from water_column_sonar_processing.aws import DynamoDBManager
from water_column_sonar_processing.aws import S3Manager
from water_column_sonar_processing.aws import S3FSManager
from water_column_sonar_processing.aws import SNSManager


###########################################################
class Process:
    #######################################################
    def __init__(
        self,
    ):
        self.input_bucket_name = os.environ["INPUT_BUCKET_NAME"]
        self.output_bucket_name = os.environ["OUTPUT_BUCKET_NAME"]
        self.table_name = os.environ["TABLE_NAME"]
        self.topic_arn = os.environ["TOPIC_ARN"]
        # self.output_bucket_access_key = ?
        # self.output_bucket_secret_access_key = ?

    def execute(self):
        input_s3_manager = (
            S3Manager()
        )  # TODO: Need to allow passing in of credentials when writing to protected bucket
        s3fs_manager = S3FSManager()  # TODO: delete this
        print(s3fs_manager)  # TODO: delete this
        output_s3_manager = S3Manager()
        # TODO: s3fs?
        sns_manager = SNSManager()
        ddb_manager = DynamoDBManager()

        # [1 of 5] Update Pipeline Status in DynamoDB
        # self.dynamodb.update_ status ()

        # [2 of 5] Download Object From Input Bucket
        # return_value = input_s3_manager.download_file(
        #    bucket_name=self.input_bucket_name,
        #    key="the_input_key",
        #    file_name="the_input_key",
        # )
        # print(return_value)

        # [3 of 5] Update Entry in DynamoDB
        ship_name = "David_Starr_Jordan"  # TODO: get this from input sns message
        cruise_name = "DS0604"
        sensor_name = "EK60"
        file_name = "DSJ0604-D20060406-T113407.raw"

        test_channels = [
            "GPT  38 kHz 009072055a7f 2 ES38B",
            "GPT  70 kHz 00907203400a 3 ES70-7C",
            "GPT 120 kHz 009072034d52 1 ES120-7",
            "GPT 200 kHz 0090720564e4 4 ES200-7C",
        ]
        test_frequencies = [38_000, 70_000, 120_000, 200_000]
        ddb_manager.update_item(
            table_name=self.table_name,
            key={
                "FILE_NAME": {"S": file_name},  # Partition Key
                "CRUISE_NAME": {"S": cruise_name},  # Sort Key
            },
            expression_attribute_names={
                "#CH": "CHANNELS",
                "#ET": "END_TIME",
                "#ED": "ERROR_DETAIL",
                "#FR": "FREQUENCIES",
                "#MA": "MAX_ECHO_RANGE",
                "#MI": "MIN_ECHO_RANGE",
                "#ND": "NUM_PING_TIME_DROPNA",
                "#PS": "PIPELINE_STATUS",  # testing this updated
                "#PT": "PIPELINE_TIME",  # testing this updated
                "#SE": "SENSOR_NAME",
                "#SH": "SHIP_NAME",
                "#ST": "START_TIME",
                "#ZB": "ZARR_BUCKET",
                "#ZP": "ZARR_PATH",
            },
            expression_attribute_values={
                ":ch": {"L": [{"S": i} for i in test_channels]},
                ":et": {"S": "2006-04-06T13:35:28.688Z"},
                ":ed": {"S": ""},
                ":fr": {"L": [{"N": str(i)} for i in test_frequencies]},
                ":ma": {"N": str(np.round(499.7653, 4))},
                ":mi": {"N": str(np.round(0.25, 4))},
                ":nd": {"N": str(2458)},
                ":ps": {"S": "SUCCESS_AGGREGATOR"},
                ":pt": {"S": "2023-10-02T08:54:43Z"},
                ":se": {"S": sensor_name},
                ":sh": {"S": ship_name},
                ":st": {"S": "2006-04-06T11:34:07.288Z"},
                ":zb": {"S": "r2d2-dev-echofish2-118234403147-echofish-dev-output"},
                ":zp": {
                    "S": "level_1/David_Starr_Jordan/DS0604/EK60/DSJ0604-D20060406-T113407.model"
                },
            },
            update_expression=(
                "SET "
                "#CH = :ch, "
                "#ET = :et, "
                "#ED = :ed, "
                "#FR = :fr, "
                "#MA = :ma, "
                "#MI = :mi, "
                "#ND = :nd, "
                "#PS = :ps, "
                "#PT = :pt, "
                "#SE = :se, "
                "#SH = :sh, "
                "#ST = :st, "
                "#ZB = :zb, "
                "#ZP = :zp"
            ),
        )

        # [4 of 5] Write Object to Output Bucket
        output_s3_manager.put(
            bucket_name=self.output_bucket_name, key="123", body="456"
        )

        # [_ of _] Read file-level Zarr store from bucket, Create GeoJSON, Write to bucket
        # [_ of _] Create empty cruise-level Zarr store
        # [_ of _] Resample and write to cruise-level Zarr Store

        # [5 of 5] Publish Done Message
        success_message = {
            "default": {
                "shipName": ship_name,
                "cruiseName": cruise_name,
                "sensorName": sensor_name,
                "fileName": file_name,
            }
        }
        sns_manager.publish(
            topic_arn=self.topic_arn,
            message=json.dumps(success_message),
        )
        print("done...")

    #######################################################


###########################################################
###########################################################
