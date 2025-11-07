import datetime


###########################################################
class Timestamp:
    @staticmethod
    def get_timestamp():
        # return timestamp in form: '2024-03-29T19:36:52.433Z'
        return f"{datetime.datetime.now(datetime.UTC).isoformat()[:23]}Z"


###########################################################
