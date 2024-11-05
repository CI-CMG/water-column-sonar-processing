import os

import boto3


###########################################################
class SNSManager:
    #######################################################
    def __init__(
        self,
    ):
        self.__sns_region = os.environ.get("AWS_REGION", default="us-east-1")
        self.__sns_session = boto3.Session(
            aws_access_key_id=os.environ.get("ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("SECRET_ACCESS_KEY"),
            region_name=self.__sns_region,
        )
        self.__sns_resource = self.__sns_session.resource(
            service_name="sns", region_name=self.__sns_region
        )
        self.__sns_client = self.__sns_session.client(
            service_name="sns", region_name=self.__sns_region
        )

    #######################################################
    # TODO: pick one
    def publish(self, topic_arn, message):
        response = self.__sns_client.publish(
            TopicArn=topic_arn,
            Message=message,
            # MessageStructure='json'
        )
        print(response)

    # TODO:
    # def publish_done_message(self, message):
    #     print("Sending done message")
    #     self.__sns_operations.publish(self.__done_topic_arn, json.dumps(message))

    #######################################################
    def create_topic(self, topic_name):
        response = self.__sns_client.create_topic(Name=topic_name)
        return response

    #######################################################
    def subscribe(self, topic_arn, endpoint):
        self.__sns_client.subscribe(
            TopicArn=topic_arn, Protocol="sqs", Endpoint=endpoint
        )

    #######################################################
    def list_topics(self):
        print(self.__sns_client.list_topics())


###########################################################
