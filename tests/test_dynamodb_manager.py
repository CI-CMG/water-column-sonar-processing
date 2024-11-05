import numpy as np
from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.aws.dynamodb_manager import DynamoDBManager
from water_column_sonar_processing.utility.pipeline_status import PipelineStatus


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


#######################################################
def test_serializer_deserializer():
    # ---Initialize--- #
    low_level_data = {
        "ACTIVE": {"BOOL": True},
        "CRC": {"N": "-1600155180"},
        "ID": {"S": "bewfv43843b"},
        "params": {"M": {"customer": {"S": "TEST"}, "index": {"N": "1"}}},
        "THIS_STATUS": {"N": "10"},
        "TYPE": {"N": "22"},
    }

    # Lazy-eval the dynamodb attribute (boto3 is dynamic!)
    dynamo_db_manager = DynamoDBManager()

    # To go from low-level format to python
    deserializer = dynamo_db_manager.type_deserializer
    python_data = {k: deserializer.deserialize(v) for k, v in low_level_data.items()}

    assert python_data["ACTIVE"]
    assert python_data["CRC"] == -1600155180
    assert python_data["ID"]
    assert python_data["params"] == {"customer": "TEST", "index": 1}
    assert python_data["THIS_STATUS"] == 10
    assert python_data["TYPE"] == 22

    # To go from python to low-level format
    serializer = dynamo_db_manager.type_serializer
    low_level_copy = {k: serializer.serialize(v) for k, v in python_data.items()}

    assert low_level_data == low_level_copy


#######################################################
@mock_aws
def test_dynamodb_manager():
    # ---Initialize--- #
    table_name = "test_table"
    dynamo_db_manager = DynamoDBManager()

    # ---Create Table--- #
    # TODO: move create tabel into DynamoDBManager
    dynamo_db_manager.create_table(
        table_name=table_name,
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
        attribute_definitions=[
            {"AttributeName": "FILE_NAME", "AttributeType": "S"},
            {"AttributeName": "CRUISE_NAME", "AttributeType": "S"},
        ],
    )

    # ---Add Items to Table--- #
    test_channels = [
        "GPT  38 kHz 009072055a7f 2 ES38B",
        "GPT  70 kHz 00907203400a 3 ES70-7C",
        "GPT 120 kHz 009072034d52 1 ES120-7",
        "GPT 200 kHz 0090720564e4 4 ES200-7C",
    ]
    test_frequencies = [38_000, 70_000, 120_000, 200_000]
    ship_name = "David_Starr_Jordan"
    cruise_name = "DS0604"
    sensor_name = "EK60"
    file_name = "DSJ0604-D20060419-T184612.raw"

    dynamo_db_manager.update_item(
        table_name=table_name,
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
            ":et": {"S": "2006-04-19T20:07:33.623Z"},
            ":ed": {"S": ""},
            ":fr": {"L": [{"N": str(i)} for i in test_frequencies]},
            ":ma": {"N": str(np.round(500.785201, 4))},
            ":mi": {"N": str(np.round(0.25001, 4))},
            ":nd": {"N": str(2428)},
            ":ps": {"S": "PROCESSING_RESAMPLE_AND_WRITE_TO_ZARR_STORE"},
            ":pt": {"S": "2023-10-02T08:08:08Z"},
            ":se": {"S": sensor_name},
            ":sh": {"S": ship_name},
            ":st": {"S": "2006-04-19T18:46:12.564Z"},
            ":zb": {"S": "r2d2-dev-echofish2-118234403147-echofish-dev-output"},
            ":zp": {
                "S": f"level_1/{ship_name}/{cruise_name}/{sensor_name}/DSJ0604-D20060419-T184612.zarr"
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

    # ---Read From Table--- #
    response = dynamo_db_manager.get_table_item(
        table_name=table_name,
        key={"FILE_NAME": "DSJ0604-D20060419-T184612.raw", "CRUISE_NAME": "DS0604"},
    )
    assert (
        response["Item"]["PIPELINE_STATUS"]
        == "PROCESSING_RESAMPLE_AND_WRITE_TO_ZARR_STORE"
    )

    # ---Change Items in Table (using src)--- #
    dynamo_db_manager.update_item(
        table_name=table_name,
        key={
            "FILE_NAME": {
                "S": "DSJ0604-D20060419-T184612.raw",
            },
            "CRUISE_NAME": {
                "S": "DS0604",
            },
        },
        expression_attribute_names={
            "#PS": "PIPELINE_STATUS",
            "#PT": "PIPELINE_TIME",
        },
        expression_attribute_values={
            ":ps": {"S": PipelineStatus.SUCCESS_CRUISE_PROCESSOR.name},
            ":pt": {"S": "2023-10-02T09:09:09Z"},
        },
        update_expression=("SET " "#PS = :ps, " "#PT = :pt"),
    )

    # ---Read From Table Again--- #
    response = dynamo_db_manager.get_table_item(
        table_name=table_name,
        key={"FILE_NAME": "DSJ0604-D20060419-T184612.raw", "CRUISE_NAME": "DS0604"},
    )

    assert response["Item"]["PIPELINE_STATUS"] == "SUCCESS_CRUISE_PROCESSOR"

    # TODO: get the table as a dataframe
    df = dynamo_db_manager.get_table_as_df(
        table_name=table_name,
        ship_name="David_Starr_Jordan",
        cruise_name="DS0604",
        sensor_name="EK60",
    )

    assert df.shape[1] == 16
    # TODO: check fields


#######################################################
