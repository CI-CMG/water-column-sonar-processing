from datetime import datetime


###########################################################
class Timestamp:
    @staticmethod
    def get_timestamp():
        # return timestamp in form:
        #   PIPELINE_TIME = '2024-03-29T19:36:52.433Z'
        return f"{datetime.utcnow().isoformat()[:23]}Z"


###########################################################
