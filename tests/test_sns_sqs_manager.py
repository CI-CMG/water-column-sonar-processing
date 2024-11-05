import json
import os

from dotenv import find_dotenv, load_dotenv
from moto import mock_aws

from water_column_sonar_processing.aws.sns_manager import SNSManager
from water_column_sonar_processing.aws.sqs_manager import SQSManager


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


#######################################################
@mock_aws
def test_sns_manager():
    print(os.environ.get("AWS_REGION"))
    # --- Initialize --- #
    sqs_queue_name = "test_queue"
    sns_topic_name = "test_topic"

    # --- Initialize SQS --- #
    sns = SNSManager()
    sqs = SQSManager()
    queue_response = sqs.create_queue(queue_name=sqs_queue_name)

    assert queue_response["ResponseMetadata"]["HTTPStatusCode"] == 200

    # --- Initialize SNS --- #
    topic_response = sns.create_topic(topic_name=sns_topic_name)
    sns_topic_arn = topic_response["TopicArn"]

    sqs_queue = sqs.get_queue_by_name(queue_name=sqs_queue_name)
    sqs_queue_arn = sqs_queue.attributes["QueueArn"]

    sns.subscribe(topic_arn=sns_topic_arn, endpoint=sqs_queue_arn)

    # --- Publish Three SNS Test Messages --- #
    test_msg_111 = {"default": {"x": "foo111", "y": "bar111"}}
    sns.publish(topic_arn=sns_topic_arn, message=json.dumps(test_msg_111))

    test_msg_222 = {"default": {"x": "foo222", "y": "bar222"}}
    sns.publish(topic_arn=sns_topic_arn, message=json.dumps(test_msg_222))

    test_msg_333 = {"default": {"x": "foo333", "y": "bar333"}}
    sns.publish(topic_arn=sns_topic_arn, message=json.dumps(test_msg_333))

    # --- Validate Messages --- #
    sqs_msgs = sqs_queue.receive_messages(
        AttributeNames=["All"],
        MessageAttributeNames=["All"],
        VisibilityTimeout=15,
        WaitTimeSeconds=20,
        MaxNumberOfMessages=10,
    )

    assert len(sqs_msgs) == 3

    assert json.loads(sqs_msgs[0].body)["Message"] == json.dumps(test_msg_111)
    assert json.loads(sqs_msgs[1].body)["Message"] == json.dumps(test_msg_222)
    assert json.loads(sqs_msgs[2].body)["Message"] == json.dumps(test_msg_333)

    print(sqs_msgs[0].body)

    #######################################################
