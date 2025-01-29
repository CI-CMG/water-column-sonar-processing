import os

import boto3
import pandas as pd
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer


#########################################################################
class DynamoDBManager:
    #####################################################################
    def __init__(
            self,
            # endpoint_url
    ):
        # self.endpoint_url = endpoint_url
        self.dynamodb_session = boto3.Session(
            aws_access_key_id=os.environ.get("ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", default="us-east-1"),
        )
        self.dynamodb_resource = self.dynamodb_session.resource(
            service_name="dynamodb",
            # endpoint_url=self.endpoint_url
        )
        self.dynamodb_client = self.dynamodb_session.client(
            service_name="dynamodb",
            # endpoint_url=self.endpoint_url
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
    #####################################################################
    def create_water_column_sonar_table(
        self,
        table_name,
    ):
        self.dynamodb_client.create_table(
            TableName=table_name,
            KeySchema=[
                {
                    "AttributeName": "FILE_NAME",
                    "KeyType": "HASH",
                },
                {
                    "AttributeName": "CRUISE_NAME",
                    "KeyType": "RANGE",
                },
            ],
            AttributeDefinitions=[
                {"AttributeName": "FILE_NAME", "AttributeType": "S"},
                {"AttributeName": "CRUISE_NAME", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST"
            # ProvisionedThroughput={
            #     'ReadCapacityUnits': 1_000,
            #     'WriteCapacityUnits': 1_000
            # }
        )
        # TODO: after creating status is 'CREATING', wait until 'ACTIVE'
        response = self.dynamodb_client.describe_table(TableName=table_name)
        print(response) # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb/client/describe_table.html
        # sleep then response['Table']['TableStatus'] == 'ACTIVE'

    #####################################################################
    # don't think this is used?
    # def get_item(
    #         self,
    #         table_name,
    #         key
    # ):
    #     response = self.dynamodb_client.get_item(TableName=table_name, Key=key)
    #     item = None
    #     if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
    #         if "Item" in response:
    #             item = response["Item"]
    #     return item

    #####################################################################
    def get_table_item(
        self,
        table_name,
        key,
    ):
        """
        Gets a single row from the db.
        """
        table = self.dynamodb_resource.Table(table_name)
        response = table.get_item(Key=key)
        # TODO:
        # if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        #     throw error
        return response

    #####################################################################
    def update_item(
        self,
        table_name,
        key,
        expression_attribute_names,
        expression_attribute_values,
        update_expression,
    ): # TODO: convert to boolean
        try:
            response = self.dynamodb_client.update_item(
                TableName=table_name,
                Key=key,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values,
                UpdateExpression=update_expression,
            )
            status_code = response["ResponseMetadata"]["HTTPStatusCode"]
            # print(f"HTTPStatusCode: {status_code}")
            # assert status_code == 200, "Problem, unable to update dynamodb table."
            # assert response['ConsumedCapacity']['TableName'] == table_name
        except Exception as err:
            print(f"Problem was encountered while updating item: {err}")

    #####################################################################
    # TODO: change to "get_cruise_as_df"
    def get_table_as_df(
        self,
        ship_name,
        cruise_name,
        sensor_name,
        table_name,
    ) -> pd.DataFrame:
        """
        To be used to initialize a cruise, deletes all entries associated with that cruise
        in the database.
        """
        filter_expression = "CRUISE_NAME = :cr"
        response = self.dynamodb_client.scan(
            TableName=table_name,
            # Limit=1000,
            Select='ALL_ATTRIBUTES', # or 'SPECIFIC_ATTRIBUTES',
            # ExclusiveStartKey=where to pick up
            #ReturnConsumedCapacity='INDEXES' | 'TOTAL' | 'NONE', ...not sure
            # ProjectionExpression='#SH, #CR, #FN', # what to specifically return — from expression_attribute_names
            FilterExpression=filter_expression,
            # ExpressionAttributeNames={
            #     '#SH': 'SHIP_NAME',
            #     '#CR': 'CRUISE_NAME',
            #     '#FN': 'FILE_NAME',
            # },
            ExpressionAttributeValues={ # criteria
                ':cr': {
                    'S': cruise_name,
                },
            },
            ConsistentRead=True
            # ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        # Note: table.scan() has 1 MB limit on results so pagination is used

        if len(response["Items"]) == 0 and "LastEvaluatedKey" not in response:
            return pd.DataFrame() # If no results, return empty dataframe

        data = response["Items"]

        while response.get('LastEvaluatedKey'): #"LastEvaluatedKey" in response:
            response = self.dynamodb_client.scan(
                TableName=table_name,
                ### Either 'Select' or 'ExpressionAttributeNames'/'ProjectionExpression'
                Select='ALL_ATTRIBUTES', # or 'SPECIFIC_ATTRIBUTES',
                FilterExpression=filter_expression,
                #ProjectionExpression='#SH, #CR, #FN',  # what to specifically return — from expression_attribute_names
                # ExpressionAttributeNames={ # would need to specify all cols in df
                #     '#SH': 'SHIP_NAME',
                #     '#CR': 'CRUISE_NAME',
                #     '#FN': 'FILE_NAME',
                # },
                ExpressionAttributeValues={  # criteria
                    ':cr': {
                        'S': cruise_name,
                    },
                },
                ConsistentRead=True,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            data.extend(response["Items"])

        deserializer = self.type_deserializer
        df = pd.DataFrame([deserializer.deserialize({"M": i}) for i in data])

        return df.sort_values(by="START_TIME", ignore_index=True)

    #####################################################################
    # def get_cruise_list(
    #     self,
    #     table_name,
    # ) -> list:
    #     """
    #     Experimental, gets all cruise names as list
    #     """
    #     filter_expression = "CRUISE_NAME = :cr"
    #     response = self.dynamodb_client.scan(
    #         TableName=table_name,
    #         Select='SPECIFIC_ATTRIBUTES',
    #         #ReturnConsumedCapacity='INDEXES' | 'TOTAL' | 'NONE', ...not sure
    #         # ProjectionExpression='#SH, #CR, #FN', # what to specifically return — from expression_attribute_names
    #         FilterExpression=filter_expression,
    #         # ExpressionAttributeNames={
    #         #     '#SH': 'SHIP_NAME',
    #         #     '#CR': 'CRUISE_NAME',
    #         #     '#FN': 'FILE_NAME',
    #         # },
    #         # ExpressionAttributeValues={ # criteria
    #         #     ':cr': {
    #         #         'S': cruise_name,
    #         #     },
    #         # },
    #     )
    #     # Note: table.scan() has 1 MB limit on results so pagination is used
    #
    #     if len(response["Items"]) == 0 and "LastEvaluatedKey" not in response:
    #         return pd.DataFrame() # If no results, return empty dataframe
    #
    #     data = response["Items"]
    #
    #     while response.get('LastEvaluatedKey'): #"LastEvaluatedKey" in response:
    #         response = self.dynamodb_client.scan(
    #             TableName=table_name,
    #             ### Either 'Select' or 'ExpressionAttributeNames'/'ProjectionExpression'
    #             Select='ALL_ATTRIBUTES', # or 'SPECIFIC_ATTRIBUTES',
    #             FilterExpression=filter_expression,
    #             #ProjectionExpression='#SH, #CR, #FN',  # what to specifically return — from expression_attribute_names
    #             # ExpressionAttributeNames={ # would need to specify all cols in df
    #             #     '#SH': 'SHIP_NAME',
    #             #     '#CR': 'CRUISE_NAME',
    #             #     '#FN': 'FILE_NAME',
    #             # },
    #             ExpressionAttributeValues={  # criteria
    #                 ':cr': {
    #                     'S': cruise_name,
    #                 },
    #             },
    #             ConsistentRead=True,
    #             ExclusiveStartKey=response["LastEvaluatedKey"],
    #         )
    #         data.extend(response["Items"])
    #
    #     deserializer = self.type_deserializer
    #     df = pd.DataFrame([deserializer.deserialize({"M": i}) for i in data])
    #
    #     return df.sort_values(by="START_TIME", ignore_index=True)

    #####################################################################
    # TODO: WIP
    def delete_item(
        self,
        table_name,
        cruise_name,
        file_name,
    ):
        """
        Finds all rows associated with a cruise and deletes them.
        """
        response = self.dynamodb_client.delete_item(
            Key={
                "CRUISE_NAME": {
                    "S": cruise_name
                },
                "FILE_NAME": {
                    "S": file_name
                }
            },
            TableName=table_name,
            ReturnConsumedCapacity="TOTALS",
        )
        # TODO: there should be attributes included in response but they are missing
        # if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        #     throw error
        return response

    #####################################################################
    def describe_table(
            self,
            table_name,
    ):
        """
        Get a description of the table. Used to verify that records were added/removed.
        """
        response = self.dynamodb_client.describe_table(TableName=table_name)
        print(response)
        return response



    #####################################################################
    # TODO: from test_raw_to_zarr get enum and use here
    # def __update_processing_status(
    #         self,
    #         file_name: str,
    #         cruise_name: str,
    #         pipeline_status: str,
    #         error_message: str = None,
    # ):
    #     print(f"Updating processing status to {pipeline_status}.")
    #     if error_message:
    #         print(f"Error message: {error_message}")
    #         self.dynamo.update_item(
    #             table_name=self.__table_name,
    #             key={
    #                 'FILE_NAME': {'S': file_name},  # Partition Key
    #                 'CRUISE_NAME': {'S': cruise_name},  # Sort Key
    #             },
    #             attribute_names={
    #                 '#PT': 'PIPELINE_TIME',
    #                 '#PS': 'PIPELINE_STATUS',
    #                 '#EM': 'ERROR_MESSAGE',
    #             },
    #             expression='SET #PT = :pt, #PS = :ps, #EM = :em',
    #             attribute_values={
    #                 ':pt': {
    #                     'S': datetime.now().isoformat(timespec="seconds") + "Z"
    #                 },
    #                 ':ps': {
    #                     'S': pipeline_status
    #                 },
    #                 ':em': {
    #                     'S': error_message
    #                 }
    #             }
    #         )
    #     else:
    #         self.dynamo.update_item(
    #             table_name=self.__table_name,
    #             key={
    #                 'FILE_NAME': {'S': file_name},  # Partition Key
    #                 'CRUISE_NAME': {'S': cruise_name},  # Sort Key
    #             },
    #             attribute_names={
    #                 '#PT': 'PIPELINE_TIME',
    #                 '#PS': 'PIPELINE_STATUS',
    #             },
    #             expression='SET #PT = :pt, #PS = :ps',
    #             attribute_values={
    #                 ':pt': {
    #                     'S': datetime.now().isoformat(timespec="seconds") + "Z"
    #                 },
    #                 ':ps': {
    #                     'S': pipeline_status
    #                 }
    #             }
    #         )
    #     print("Done updating processing status.")

#########################################################################
