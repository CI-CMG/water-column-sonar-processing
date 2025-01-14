import os
import re
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from water_column_sonar_processing.aws import S3Manager


MAX_POOL_CONNECTIONS = 64
MAX_CONCURRENCY = 64
MAX_WORKERS = 64
GB = 1024**3


class IndexManager:
    # TODO: index into dynamodb instead of csv files

    def __init__(self, input_bucket_name, calibration_bucket, calibration_key):
        self.input_bucket_name = input_bucket_name
        self.calibration_bucket = calibration_bucket
        self.calibration_key = calibration_key
        self.s3_manager = S3Manager() # TODO: make anonymous?

    #################################################################
    def list_ships(
        self,
        prefix="data/raw/",
    ):
        page_iterator = self.s3_manager.paginator.paginate(
            Bucket=self.input_bucket_name, Prefix=prefix, Delimiter="/"
        )
        # common_prefixes = s3_client.list_objects(Bucket=self.input_bucket_name, Prefix=prefix, Delimiter='/')
        # print(common_prefixes)
        ships = []
        for page in page_iterator:
            if "Contents" in page.keys():
                ships.extend([k["Prefix"] for k in page["CommonPrefixes"]])
        return ships  # ~76 ships

    #################################################################
    def list_cruises(
        self,
        ship_prefixes,  # e.g. 'data/raw/Alaska_Knight/'
    ):
        cruises = []
        for ship_prefix in ship_prefixes:
            page_iterator = self.s3_manager.paginator.paginate(
                Bucket=self.input_bucket_name, Prefix=ship_prefix, Delimiter="/"
            )
            for page in page_iterator:
                cruises.extend([k["Prefix"] for k in page["CommonPrefixes"]])
        return cruises  # ~1204 cruises

    #################################################################
    def list_ek60_cruises(
        self,
        cruise_prefixes,
    ):
        """
        This returns a list of ek60 prefixed cruises.
        """
        cruise_sensors = []  # includes all sensor types
        for cruise_prefix in cruise_prefixes:
            page_iterator = self.s3_manager.paginator.paginate(
                Bucket=self.input_bucket_name, Prefix=cruise_prefix, Delimiter="/"
            )
            for page in page_iterator:
                cruise_sensors.extend([k["Prefix"] for k in page["CommonPrefixes"]])
        # Note: these are "EK60" by prefix. They still need to be verified by scanning the datagram.
        return [i for i in cruise_sensors if "/EK60/" in i]  # ~447 different cruises

    #################################################################
    def get_raw_files(
        self,
        ship_name,
        cruise_name,
        sensor_name,
    ):
        # Gets all raw files for a cruise under the given prefix
        prefix = f"data/raw/{ship_name}/{cruise_name}/{sensor_name}/"  # Note no forward slash at beginning
        page_iterator = self.s3_manager.paginator.paginate(
            Bucket=self.input_bucket_name,
            Prefix=prefix,
            Delimiter="/"
        )
        all_files = []
        for page in page_iterator:
            if "Contents" in page.keys():
                all_files.extend([i["Key"] for i in page["Contents"]])
        return [i for i in all_files if i.endswith(".raw")]

    def get_first_raw_file(
        self,
        ship_name,
        cruise_name,
        sensor_name,
    ):
        # Same as above but only needs to get the first raw file
        # because we are only interested in the first datagram of one file
        prefix = f"data/raw/{ship_name}/{cruise_name}/{sensor_name}/"  # Note no forward slash at beginning
        # page_iterator = self.s3_manager.paginator.paginate(
        #     Bucket=self.input_bucket_name,
        #     Prefix=prefix,
        #     Delimiter="/",
        #     PaginationConfig={ 'MaxItems': 5 }
        # ) # TODO: this can create a problem if there is a non raw file returned first
        ### filter with JMESPath expressions ###
        page_iterator = self.s3_manager.paginator.paginate(
            Bucket=self.input_bucket_name,
            Prefix=prefix,
            Delimiter="/",
        )
        # page_iterator = page_iterator.search("Contents[?Size < `2200`][]")
        page_iterator = page_iterator.search(expression="Contents[?contains(Key, '.raw')] ")
        for res in page_iterator:
            if "Key" in res:
                return res["Key"]
        # else raise exception?

        # DSJ0604-D20060406-T050022.bot 2kB == 2152 'Size'

    def get_files_under_size(
        self,
        ship_name,
        cruise_name,
        sensor_name,
    ):
        # THIS isn't used, just playing with JMES paths spec
        prefix = f"data/raw/{ship_name}/{cruise_name}/{sensor_name}/"
        ### filter with JMESPath expressions ###
        page_iterator = self.s3_manager.paginator.paginate(
            Bucket=self.input_bucket_name,
            Prefix=prefix,
            Delimiter="/",
        )
        page_iterator = page_iterator.search("Contents[?Size < `2200`][]")
        all_files = []
        for page in page_iterator:
            if "Contents" in page.keys():
                all_files.extend([i["Key"] for i in page["Contents"]])
        return [i for i in all_files if i.endswith(".raw")]

    #################################################################
    def get_raw_files_csv(
        self,
        ship_name,
        cruise_name,
        sensor_name,
    ):
        raw_files = self.get_raw_files(
            ship_name=ship_name,
            cruise_name=cruise_name,
            sensor_name=sensor_name
        )
        files_list = [
            {
                "ship_name": ship_name,
                "cruise_name": cruise_name,
                "sensor_name": sensor_name,
                "file_name": os.path.basename(raw_file),
            }
            for raw_file in raw_files
        ]
        df = pd.DataFrame(files_list)
        df.to_csv(f"{ship_name}_{cruise_name}.csv", index=False, header=False, sep=" ")
        print("done")

    def get_raw_files_list(
        self,
        ship_name,
        cruise_name,
        sensor_name,
    ):
        # gets all raw files in cruise and returns a list of dicts
        raw_files = self.get_raw_files(
            ship_name=ship_name,
            cruise_name=cruise_name,
            sensor_name=sensor_name
        )
        files_list = [
            {
                "ship_name": ship_name,
                "cruise_name": cruise_name,
                "sensor_name": sensor_name,
                "file_name": os.path.basename(raw_file),
            }
            for raw_file in raw_files
        ]
        return files_list

    #################################################################
    def get_subset_ek60_prefix( # TODO: is this used?
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Returns all objects with 'EK60' in prefix of file path
        # Note that this can include 'EK80' data that are false-positives
        # in dataframe with ['key', 'filename', 'ship', 'cruise', 'sensor', 'size', 'date', 'datagram']
        print("getting subset of ek60 data by prefix")
        objects = []
        for row in df.itertuples():
            row_split = row[1].split(os.sep)
            if len(row_split) == 6:
                filename = os.path.basename(
                    row[1]
                )  # 'EX1608_EK60-D20161205-T040300.raw'
                if filename.endswith(".raw"):
                    ship_name, cruise_name, sensor_name = row_split[
                        2:5
                    ]  # 'Okeanos_Explorer', 'EX1608', 'EK60'
                    if (
                        re.search("[D](\\d{8})", filename) is not None
                        and re.search("[T](\\d{6})", filename) is not None
                    ):
                        # Parse date if possible e.g.: 'data/raw/Henry_B._Bigelow/HB1006/EK60/HBB-D20100723-T025105.raw'
                        # and 'data/raw/Henry_B._Bigelow/HB1802/EK60/D20180513-T150250.raw'
                        date_substring = re.search("[D](\\d{8})", filename).group(1)
                        time_substring = re.search("[T](\\d{6})", filename).group(1)
                        date_string = datetime.strptime(
                            f"{date_substring}{time_substring}", "%Y%m%d%H%M%S"
                        )
                    else:  # otherwise use current date
                        date_string = f"{datetime.utcnow().isoformat()[:19]}Z"
                    objects.append(
                        {
                            "KEY": row[1],
                            "FILENAME": filename,
                            "SHIP": ship_name,
                            "CRUISE": cruise_name,
                            "SENSOR": sensor_name,
                            "SIZE": row[2],
                            "DATE": date_string,
                            "DATAGRAM": None,
                        }
                    )
        return pd.DataFrame(objects)

    #################################################################
    def scan_datagram(
        self,
        select_key: str
    ) -> list:
        # Reads the first 8 bytes of S3 file. Used to determine if ek60 or ek80
        # Note: uses boto3 session instead of boto3 client: https://github.com/boto/boto3/issues/801
        # select_key = 'data/raw/Albatross_Iv/AL0403/EK60/L0005-D20040302-T200108-EK60.raw'
        s3_resource = self.s3_manager.s3_resource
        obj = s3_resource.Object(
            bucket_name=self.input_bucket_name, key=select_key
        )  # XML0
        first_datagram = (
            obj.get(Range="bytes=3-7")["Body"].read().decode().strip("\x00")
        )
        # return [{'KEY': select_key, 'DATAGRAM': first_datagram}]
        ### EK60 data are denoted by 'CON0' ###
        return first_datagram

    #################################################################
    def get_subset_datagrams( # TODO: is this getting used
        self,
        df: pd.DataFrame
    ) -> list:
        print("getting subset of datagrams")
        select_keys = df[["KEY", "CRUISE"]].drop_duplicates(subset="CRUISE")["KEY"].values.tolist()
        all_datagrams = []
        with ThreadPoolExecutor(max_workers=MAX_POOL_CONNECTIONS) as executor:
            futures = [
                executor.submit(self.scan_datagram, select_key)
                for select_key in select_keys
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_datagrams.extend(result)
        return all_datagrams

    #################################################################
    def get_ek60_objects(
        self,
        df: pd.DataFrame,
        subset_datagrams: list
    ) -> pd.DataFrame:
        # for each key write datagram value to all other files in same cruise
        for subset_datagram in subset_datagrams:
            if subset_datagram["DATAGRAM"] == "CON0":
                select_cruise = df.loc[df["KEY"] == subset_datagram["KEY"]][
                    "CRUISE"
                ].iloc[0]
                df.loc[df["CRUISE"] == select_cruise, ["DATAGRAM"]] = subset_datagram[
                    "DATAGRAM"
                ]
        return df.loc[df["DATAGRAM"] == "CON0"]

    #################################################################
    def get_calibration_information(
        self,
    ) -> pd.DataFrame:
        # Calibration data generated by data manager currently located here:
        #      https://noaa-wcsd-pds-index.s3.amazonaws.com/calibrated_crusies.csv
        # Note: Data are either:
        #      [1] Calibrated w/ calibration data
        #      [2] Calibrated w/o calibration data
        #      [3] uncalibrated
        response = self.s3_manager.get_object(
            bucket_name=self.calibration_bucket, key_name=self.calibration_key
        )
        calibration_statuses = pd.read_csv(response.get("Body"))
        calibration_statuses["DATASET_NAME"] = calibration_statuses[
            "DATASET_NAME"
        ].apply(lambda x: x.split("_EK60")[0])
        calibration_statuses["CAL_STATE"] = calibration_statuses["CAL_STATE"].apply(
            lambda x: x.find("Calibrated") >= 0
        )
        return calibration_statuses

    #################################################################
    # def index(  # TODO: get rid of this?
    #         self
    # ):
    #     start_time = datetime.now()  # used for benchmarking
    #     # Get all object in public dataset bucket
    #     all_objects = self.get_all_objects()
    #     #
    #     subset_ek60_by_prefix = self.get_subset_ek60_prefix(
    #         df=all_objects[all_objects['Key'].str.contains('EK60')][['Key', 'Size']]
    #     )
    #     #
    #     subset_datagrams = self.get_subset_datagrams(df=subset_ek60_by_prefix)
    #     print("done getting subset of datagrams")
    #     ek60_objects = self.get_ek60_objects(subset_ek60_by_prefix, subset_datagrams)
    #     print("done getting ek60_objects")
    #     print(start_time)
    #     calibration_status = self.get_calibration_information(s3)
    #     cruise_names = list(set(ek60_objects['CRUISE']))
    #     cruise_names.sort()
    #     for cruise_name in cruise_names:  # ~322 cruises
    #         cruise_data = ek60_objects.groupby('CRUISE').get_group(cruise_name)
    #         ship = cruise_data['SHIP'].iloc[0]
    #         sensor = cruise_data['SENSOR'].iloc[0]
    #         datagram = cruise_data['DATAGRAM'].iloc[0]
    #         file_count = cruise_data.shape[0]
    #         total_size = np.sum(cruise_data['SIZE'])
    #         calibrated = cruise_name in calibration_status['DATASET_NAME'].unique()  # ~276 entries
    #         start_date = np.min(cruise_data['DATE']).isoformat(timespec="seconds") + "Z"
    #         end_date = np.max(cruise_data['DATE']).isoformat(timespec="seconds") + "Z"
    #     end_time = datetime.now()  # used for benchmarking
    #     print(start_time)
    #     print(end_time)
