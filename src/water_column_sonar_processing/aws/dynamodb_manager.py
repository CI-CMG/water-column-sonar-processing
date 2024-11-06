import os

import boto3
import pandas as pd
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer


#########################################################################
class DynamoDBManager:
    #####################################################################
    def __init__(self):
        self.__dynamodb_session = boto3.Session(
            aws_access_key_id=os.environ.get("ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", default="us-east-1"),
        )
        self.__dynamodb_resource = self.__dynamodb_session.resource(
            service_name="dynamodb",
        )
        self.__dynamodb_client = self.__dynamodb_session.client(
            service_name="dynamodb",
        )
        self.type_serializer = TypeSerializer()  # https://stackoverflow.com/a/46738251
        self.type_deserializer = TypeDeserializer()

    #####################################################################
    ### defined in raw-to-model, not used
    # def put_item(
    #         self,
    #         table_name,
    #         item
    # ):
    #     response = boto3.Session().client(service_name='dynamodb').put_item(TableName=table_name, Item=item)
    #     status_code = response['ResponseMetadata']['HTTPStatusCode']
    #     assert (status_code == 200), "Problem, unable to update dynamodb table."

    #####################################################################
    def create_table(
        self,
        table_name,
        key_schema,
        attribute_definitions,
    ):
        self.__dynamodb_client.create_table(
            AttributeDefinitions=attribute_definitions,
            TableName=table_name,
            KeySchema=key_schema,
            BillingMode="PAY_PER_REQUEST",  # "PROVISIONED",
            # ProvisionedThroughput={
            #     'ReadCapacityUnits': 1_000,
            #     'WriteCapacityUnits': 1_000
            # }
        )

    #####################################################################
    def create_water_column_sonar_table(
        self,
        table_name,
    ):
        self.create_table(
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

    #####################################################################
    def get_item(self, table_name, key):
        response = self.__dynamodb_client.get_item(TableName=table_name, Key=key)
        item = None
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            if "Item" in response:
                item = response["Item"]
        return item

    #####################################################################
    def update_item(
        self,
        table_name,
        key,
        expression_attribute_names,
        expression_attribute_values,
        update_expression,
    ):
        response = self.__dynamodb_client.update_item(
            TableName=table_name,
            Key=key,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values,
            UpdateExpression=update_expression,
        )
        status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        # TODO: change to exception
        assert status_code == 200, "Problem, unable to update dynamodb table."

    #####################################################################
    def get_table_as_df(
        self,
        ship_name,
        cruise_name,
        sensor_name,
        table_name,
    ):
        expression_attribute_values = {
            ":cr": {"S": cruise_name},
            ":se": {"S": sensor_name},
            ":sh": {"S": ship_name},
        }

        filter_expression = (
            "CRUISE_NAME = :cr and SENSOR_NAME = :se and SHIP_NAME = :sh"
        )
        response = self.__dynamodb_client.scan(
            TableName=table_name,
            Select="ALL_ATTRIBUTES",
            ExpressionAttributeValues=expression_attribute_values,
            FilterExpression=filter_expression,
        )
        # Note: table.scan() has 1 MB limit on results so pagination is used
        data = response["Items"]

        while "LastEvaluatedKey" in response:
            response = self.__dynamodb_client.scan(
                TableName=table_name,
                Select="ALL_ATTRIBUTES",
                ExpressionAttributeValues=expression_attribute_values,
                FilterExpression=filter_expression,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            data.extend(response["Items"])

        deserializer = self.type_deserializer
        df = pd.DataFrame([deserializer.deserialize({"M": i}) for i in data])

        return df.sort_values(by="START_TIME", ignore_index=True)

    #####################################################################
    # is this used?
    def get_table_item(
        self,
        table_name,
        key,
    ):
        # a bit more high level, uses resource to get table item
        table = self.__dynamodb_resource.Table(table_name)
        response = table.get_item(Key=key)
        return response

    #####################################################################
    # TODO: add helper method to delete the data
    def delete_cruise(
        self,
        table_name,
        cruise_name,
    ):
        pass


#########################################################################
