import json
import os
import pytest
import numpy as np
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.aws.dynamodb_manager import DynamoDBManager
from water_column_sonar_processing.aws.s3_manager import S3Manager
from water_column_sonar_processing.aws.sns_manager import SNSManager
from water_column_sonar_processing.aws.sqs_manager import SQSManager
from water_column_sonar_processing.process import Process


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


#######################################################
@mock_aws
@pytest.mark.skip(reason="no way of currently testing this")
def test_model_happy_path():
    test_input_bucket_name = os.environ["INPUT_BUCKET_NAME"]

    test_output_bucket_name = os.environ["OUTPUT_BUCKET_NAME"]

    test_table_name = os.environ["TABLE_NAME"]

    test_topic_arn = os.environ["TOPIC_ARN"]
    test_topic_name = test_topic_arn.split(":")[-1]

    # [1 of 3] Create DynamoDB table
    ddbm = DynamoDBManager()
    ddbm.create_table(
        attribute_definitions=[
            {"AttributeName": "FILE_NAME", "AttributeType": "S"},
            {"AttributeName": "CRUISE_NAME", "AttributeType": "S"},
        ],
        table_name=test_table_name,
        key_schema=[
            {
                "AttributeName": "FILE_NAME",
                "KeyType": "HASH",
            },
            {
                "AttributeName": "CRUISE_NAME",
                "KeyType": "RANGE",
            },
        ],
    )
    ###################################################
    # tests data 0 - David_Starr_Jordan - DS0604
    # tests data 1 - Okeanos_Explorer - EX1404L2
    # tests data 2 - Henry_B._Bigelow - HB0707
    # tests data 3 - Miller_Freeman - MF0710
    ###################################################
    # tests data 0 - David_Starr_Jordan - DS0604
    test_channels = [
        "GPT  38 kHz 009072055a7f 2 ES38B",
        "GPT  70 kHz 00907203400a 3 ES70-7C",
        "GPT 120 kHz 009072034d52 1 ES120-7",
        "GPT 200 kHz 0090720564e4 4 ES200-7C",
    ]
    test_frequencies = [38_000, 70_000, 120_000, 200_000]
    # Create the first and third tests example files for the same cruise
    ddbm.update_item(
        table_name=test_table_name,
        key={
            "FILE_NAME": {"S": "DSJ0604-D20060406-T035914.raw"},  # Partition Key
            "CRUISE_NAME": {"S": "DS0604"},  # Sort Key
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
            ":et": {"S": "2006-04-06T03:59:15.587Z"},
            ":ed": {"S": ""},
            ":fr": {"L": [{"N": str(i)} for i in test_frequencies]},
            ":ma": {"N": str(np.round(499.5721, 4))},
            ":mi": {"N": str(np.round(0.25, 4))},
            ":nd": {"N": str(1)},
            ":ps": {"S": "SUCCESS_AGGREGATOR"},
            ":pt": {"S": "2023-10-02T08:54:41Z"},
            ":se": {"S": "EK60"},
            ":sh": {"S": "David_Starr_Jordan"},
            ":st": {"S": "2006-04-06T03:59:14.115Z"},
            ":zb": {"S": "r2d2-dev-echofish2-118234403147-echofish-dev-output"},
            ":zp": {
                "S": "level_1/David_Starr_Jordan/DS0604/EK60/DSJ0604-D20060406-T035914.model"
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
    ddbm.update_item(
        table_name=test_table_name,
        key={
            "FILE_NAME": {"S": "DSJ0604-D20060406-T133530.raw"},  # Partition Key
            "CRUISE_NAME": {"S": "DS0604"},  # Sort Key
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
            ":et": {"S": "2006-04-06T15:16:51.945Z"},
            ":ed": {"S": ""},
            ":fr": {"L": [{"N": str(i)} for i in test_frequencies]},
            ":ma": {"N": str(np.round(499.7653, 4))},
            ":mi": {"N": str(np.round(0.25, 4))},
            ":nd": {"N": str(2467)},
            ":ps": {"S": "SUCCESS_AGGREGATOR"},
            ":pt": {"S": "2023-10-02T08:54:43Z"},
            ":se": {"S": "EK60"},
            ":sh": {"S": "David_Starr_Jordan"},
            ":st": {"S": "2006-04-06T13:35:30.701Z"},
            ":zb": {"S": "r2d2-dev-echofish2-118234403147-echofish-dev-output"},
            ":zp": {
                "S": "level_1/David_Starr_Jordan/DS0604/EK60/DSJ0604-D20060406-T133530.model"
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
    ###################################################
    # tests data 1 - Okeanos_Explorer - EX1404L2
    test_channels = ["GPT  18 kHz 009072066c0e 1-1 ES18-11"]
    test_frequencies = [18_000]
    ddbm.update_item(
        table_name=test_table_name,
        key={
            "FILE_NAME": {"S": "EX1404L2_EK60_-D20140908-T173907.raw"},  # Partition Key
            "CRUISE_NAME": {"S": "EX1404L2"},  # Sort Key
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
            ":et": {"S": "2014-09-08T17:56:49.024Z"},
            ":ed": {"S": ""},
            ":fr": {"L": [{"N": str(i)} for i in test_frequencies]},
            ":ma": {"N": str(np.round(2499.7573, 4))},
            ":mi": {"N": str(np.round(0.25, 4))},
            ":nd": {"N": str(324)},
            ":ps": {"S": "SUCCESS_AGGREGATOR"},
            ":pt": {"S": "2023-10-02T18:19:44Z"},
            ":se": {"S": "EK60"},
            ":sh": {"S": "Okeanos_Explorer"},
            ":st": {"S": "2014-09-08T17:39:07.660Z"},
            ":zb": {"S": "r2d2-dev-echofish2-118234403147-echofish-dev-output"},
            ":zp": {
                "S": "level_1/Okeanos_Explorer/EX1404L2/EK60/EX1404L2_EK60_-D20140908-T173907.model"
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
    ###################################################
    # tests data 2 - Henry_B._Bigelow - HB0707
    test_channels = [
        "GPT  18 kHz 009072056b0e 2 ES18-11",
        "GPT  38 kHz 0090720346bc 1 ES38B",
        "GPT 120 kHz 0090720580f1 3 ES120-7C",
        "GPT 200 kHz 009072034261 4 ES200-7C",
    ]
    test_frequencies = [18_000, 38_000, 120_000, 200_000]
    ddbm.update_item(
        table_name=test_table_name,
        key={
            "FILE_NAME": {"S": "D20070712-T061745.raw"},  # Partition Key
            "CRUISE_NAME": {"S": "HB0707"},  # Sort Key
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
            ":et": {"S": "2007-07-12T10:05:02.579Z"},
            ":ed": {"S": ""},
            ":fr": {"L": [{"N": str(i)} for i in test_frequencies]},
            ":ma": {"N": str(np.round(249.792, 4))},
            ":mi": {"N": str(np.round(0.25, 4))},
            ":nd": {"N": str(9733)},
            ":ps": {"S": "SUCCESS_AGGREGATOR"},
            ":pt": {"S": "2023-10-01T20:13:58Z"},
            ":se": {"S": "EK60"},
            ":sh": {"S": "Henry_B._Bigelow"},
            ":st": {"S": "2007-07-12T06:17:45.579Z"},
            ":zb": {"S": "r2d2-dev-echofish2-118234403147-echofish-dev-output"},
            ":zp": {
                "S": "level_1/Henry_B._Bigelow/HB0707/EK60/D20070712-T061745.model"
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
    ###################################################
    # tests data 3 - Miller_Freeman - MF0710
    test_channels = [
        "GPT  18 kHz 009072034d55 3 ES18-11",
        "GPT  38 kHz 009072016e01 4 ES38B",
        "GPT 120 kHz 009072016a73 1 ES120-7C",
        "GPT 200 kHz 009072033fcc 2 ES200-7C",
    ]
    test_frequencies = [18_000, 38_000, 120_000, 200_000]
    ddbm.update_item(
        table_name=test_table_name,
        key={
            "FILE_NAME": {"S": "HAKE2007-D20070708-T200449.raw"},  # Partition Key
            "CRUISE_NAME": {"S": "MF0710"},  # Sort Key
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
            ":et": {"S": "2007-07-08T20:44:55.598Z"},
            ":ed": {"S": ""},
            ":fr": {"L": [{"N": str(i)} for i in test_frequencies]},
            ":ma": {"N": str(np.round(749.7416, 4))},
            ":mi": {"N": str(np.round(0.25, 4))},
            ":nd": {"N": str(801)},
            ":ps": {"S": "SUCCESS_AGGREGATOR"},
            ":pt": {"S": "2023-10-02T08:41:50Z"},
            ":se": {"S": "EK60"},
            ":sh": {"S": "Miller_Freeman"},
            ":st": {"S": "2007-07-08T20:04:49.552Z"},
            ":zb": {"S": "r2d2-dev-echofish2-118234403147-echofish-dev-output"},
            ":zp": {
                "S": "level_1/Miller_Freeman/MF0710/EK60/HAKE2007-D20070708-T200449.model"
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
    ###################################################

    # [2 of 3 - Part I] Create S3 bucket
    input_s3m = S3Manager()
    input_s3m.create_bucket(bucket_name=test_input_bucket_name)
    output_s3m = S3Manager()  # TODO: requires different credentials
    output_s3m.create_bucket(bucket_name=test_output_bucket_name)
    # TODO: create two buckets with two sets of credentials required
    all_buckets = input_s3m.list_buckets()
    print(all_buckets)

    # [2 of 3 - Part II] Add Object to Input Bucket
    input_s3m.put(
        bucket_name=test_input_bucket_name, key="the_input_key", body="the_input_body"
    )

    # [3 of 3] Set up SNS and SQS
    snsm = SNSManager()
    sqsm = SQSManager()

    sqs_queue_name = "test-queue"
    create_queue_response = sqsm.create_queue(queue_name=sqs_queue_name)
    print(create_queue_response["QueueUrl"])
    assert create_queue_response["ResponseMetadata"]["HTTPStatusCode"] == 200

    create_topic_response = snsm.create_topic(topic_name=test_topic_name)
    sns_topic_arn = create_topic_response["TopicArn"]
    sqs_queue = sqsm.get_queue_by_name(queue_name=sqs_queue_name)
    sqs_queue_arn = sqs_queue.attributes["QueueArn"]
    snsm.subscribe(topic_arn=sns_topic_arn, endpoint=sqs_queue_arn)
    ###troubleshooting
    # snsm.list_topics()
    # snsm.publish(
    #     topic_arn=sns_topic_arn,
    #     message=json.dumps("abc"),
    #     # MessageStructure='json'
    # )
    ###### end setup ######

    #############################################################
    model_instance = Process()
    # run the src
    model_instance.execute()
    #############################################################

    # tests all the outcomes
    # (1) file is in bucket
    # (2) sns messages are in queue
    # (3) dynamodb was updated

    # [1 of 3] Check that file is in the Output Bucket
    # TODO: change to writing file to s3 bucket using s3fs
    s3_object = input_s3m.get(bucket_name=test_input_bucket_name, key="the_input_key")
    body = s3_object.get()["Body"].read().decode("utf-8")
    assert body == "the_input_body"

    # [2 of 3] Validate SNS Message was Dispatched
    sqs_msgs = sqs_queue.receive_messages(
        AttributeNames=["All"],
        MessageAttributeNames=["All"],
        VisibilityTimeout=15,
        WaitTimeSeconds=20,
        MaxNumberOfMessages=10,
    )
    assert len(sqs_msgs) == 1
    test_success_message = {
        "default": {
            "shipName": "David_Starr_Jordan",
            "cruiseName": "DS0604",
            "sensorName": "EK60",
            "fileName": "DSJ0604-D20060406-T113407.raw",
        }
    }
    assert json.loads(sqs_msgs[0].body)["Message"] == json.dumps(test_success_message)

    # [3 of 3] Check that DynamoDB has been updated
    # TODO: get the table as a dataframe
    df = ddbm.get_table_as_df(
        table_name=test_table_name,
        ship_name="David_Starr_Jordan",
        cruise_name="DS0604",
        sensor_name="EK60",
    )

    # 2 files were processed previously, creating new total of 3
    assert df.shape[0] == 3

    # 16 columns of data are captured
    assert df.shape[1] == 16

    # check that new file name is included
    assert "DSJ0604-D20060406-T113407.raw" in list(df["FILE_NAME"])

    # make sure that other filenames aren't included
    assert "HAKE2007-D20070708-T200449.raw" not in list(df["FILE_NAME"])

    # assert df[PIPELINE_STATUS'] == __?__


# def test_model_file_already_exists(self):
#     pass

#######################################################
