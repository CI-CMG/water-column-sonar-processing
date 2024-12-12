import os

import boto3


###########################################################
class SQSManager:
    #######################################################
    def __init__(
        self,
    ):
        self.__sqs_region = os.environ.get("AWS_REGION", default="us-east-1")
        self.__sqs_session = boto3.Session(
            aws_access_key_id=os.environ.get("ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("SECRET_ACCESS_KEY"),
            region_name=self.__sqs_region,
        )
        self.__sqs_resource = self.__sqs_session.resource(
            service_name="sqs", region_name=self.__sqs_region
        )
        self.__sqs_client = self.__sqs_session.client(
            service_name="sqs", region_name=self.__sqs_region
        )

    #######################################################
    def create_queue(self, queue_name):
        response = self.__sqs_client.create_queue(QueueName=queue_name)
        return response

    #######################################################
    def get_queue_by_name(self, queue_name):
        sqs_queue = self.__sqs_resource.get_queue_by_name(QueueName=queue_name)
        return sqs_queue

    #######################################################
    def list_queues(self, queue_name_prefix):
        # Note: SQS control plane is eventually consistent, meaning that it
        # takes a while to propagate the data accross the systems.
        response = self.__sqs_client.list_queues(QueueNamePrefix=queue_name_prefix)
        print(response)

    #######################################################
