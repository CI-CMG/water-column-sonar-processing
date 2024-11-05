from dotenv import find_dotenv, load_dotenv

from water_column_sonar_processing.index.index_manager import IndexManager


#######################################################
def setup_module(module):
    print("setup")
    env_file = find_dotenv(".env-prod")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module(module):
    print("teardown")


#######################################################


def test_get_calibration_information():
    input_bucket_name = "noaa-wcsd-pds"
    calibration_bucket = "noaa-wcsd-pds-index"
    calibration_key = "calibrated_cruises.csv"

    index_manager = IndexManager(input_bucket_name, calibration_bucket, calibration_key)

    calibration_information = index_manager.get_calibration_information()
    assert "DP06_EK80" in list(calibration_information["DATASET_NAME"])
    assert "DY1906" in list(calibration_information["DATASET_NAME"])


# @mock_s3
# def test_index_manager(tmp_path):
#     input_bucket_name = 'noaa-wcsd-pds'
#     calibration_bucket = 'noaa-wcsd-pds-index'
#     calibration_key = 'calibrated_crusies.csv'
#
#     index = IndexManager(
#         input_bucket_name,
#         calibration_bucket,
#         calibration_key
#     )
#
#     all_ek60_data = index.index()
#     print(all_ek60_data)


def test_get_all_cruise_raw_files(tmp_path):
    input_bucket_name = "noaa-wcsd-pds"
    calibration_bucket = "noaa-wcsd-pds-index"
    calibration_key = "calibrated_crusies.csv"

    index_manager = IndexManager(input_bucket_name, calibration_bucket, calibration_key)

    ship_prefixes = index_manager.list_ships(prefix="data/raw/")
    cruise_prefixes = index_manager.list_cruises(ship_prefixes=ship_prefixes)
    ek60_cruise_prefixes = index_manager.list_ek60_cruises(
        cruise_prefixes=cruise_prefixes
    )
    print(len(ek60_cruise_prefixes))

    # TODO: process all these cruises
    bigelow_cruises = [
        "data/raw/Henry_B._Bigelow/HB0706/EK60/",
        "data/raw/Henry_B._Bigelow/HB0707/EK60/",
        "data/raw/Henry_B._Bigelow/HB0710/EK60/",
        "data/raw/Henry_B._Bigelow/HB0711/EK60/",
        "data/raw/Henry_B._Bigelow/HB0802/EK60/",
        "data/raw/Henry_B._Bigelow/HB0803/EK60/",
        "data/raw/Henry_B._Bigelow/HB0805/EK60/",
        "data/raw/Henry_B._Bigelow/HB0806/EK60/",
        "data/raw/Henry_B._Bigelow/HB0807/EK60/",
        "data/raw/Henry_B._Bigelow/HB0901/EK60/",
        "data/raw/Henry_B._Bigelow/HB0902/EK60/",
        "data/raw/Henry_B._Bigelow/HB0903/EK60/",
        "data/raw/Henry_B._Bigelow/HB0904/EK60/",
        "data/raw/Henry_B._Bigelow/HB0905/EK60/",
        "data/raw/Henry_B._Bigelow/HB1002/EK60/",
        "data/raw/Henry_B._Bigelow/HB1006/EK60/",
        "data/raw/Henry_B._Bigelow/HB1102/EK60/",
        "data/raw/Henry_B._Bigelow/HB1103/EK60/",
        "data/raw/Henry_B._Bigelow/HB1105/EK60/",
        "data/raw/Henry_B._Bigelow/HB1201/EK60/",
        "data/raw/Henry_B._Bigelow/HB1206/EK60/",
        "data/raw/Henry_B._Bigelow/HB1301/EK60/",
        "data/raw/Henry_B._Bigelow/HB1303/EK60/",
        "data/raw/Henry_B._Bigelow/HB1304/EK60/",
        "data/raw/Henry_B._Bigelow/HB1401/EK60/",
        "data/raw/Henry_B._Bigelow/HB1402/EK60/",
        "data/raw/Henry_B._Bigelow/HB1403/EK60/",
        "data/raw/Henry_B._Bigelow/HB1405/EK60/",
        "data/raw/Henry_B._Bigelow/HB1501/EK60/",
        "data/raw/Henry_B._Bigelow/HB1502/EK60/",
        "data/raw/Henry_B._Bigelow/HB1503/EK60/",
        "data/raw/Henry_B._Bigelow/HB1506/EK60/",
        "data/raw/Henry_B._Bigelow/HB1507/EK60/",
        "data/raw/Henry_B._Bigelow/HB1601/EK60/",
        "data/raw/Henry_B._Bigelow/HB1603/EK60/",
        "data/raw/Henry_B._Bigelow/HB1604/EK60/",
        "data/raw/Henry_B._Bigelow/HB1701/EK60/",
        "data/raw/Henry_B._Bigelow/HB1702/EK60/",
        "data/raw/Henry_B._Bigelow/HB1801/EK60/",
        "data/raw/Henry_B._Bigelow/HB1802/EK60/",
        "data/raw/Henry_B._Bigelow/HB1803/EK60/",
        "data/raw/Henry_B._Bigelow/HB1804/EK60/",
        "data/raw/Henry_B._Bigelow/HB1805/EK60/",
        "data/raw/Henry_B._Bigelow/HB1806/EK60/",
        "data/raw/Henry_B._Bigelow/HB1901/EK60/",
        "data/raw/Henry_B._Bigelow/HB1902/EK60/",
        "data/raw/Henry_B._Bigelow/HB1903/EK60/",
        "data/raw/Henry_B._Bigelow/HB1904/EK60/",
        "data/raw/Henry_B._Bigelow/HB1906/EK60/",
        "data/raw/Henry_B._Bigelow/HB1907/EK60/",
        "data/raw/Henry_B._Bigelow/HB2001/EK60/",
        "data/raw/Henry_B._Bigelow/HB2006/EK60/",
        "data/raw/Henry_B._Bigelow/HB2007/EK60/",
        "data/raw/Henry_B._Bigelow/HB20ORT/EK60/",
        "data/raw/Henry_B._Bigelow/HB20TR/EK60/",
        "data/raw/Henry_B._Bigelow/HB2101/EK60/",
        "data/raw/Henry_B._Bigelow/HB2102/EK60/",
        "data/raw/Henry_B._Bigelow/HB2103/EK60/",
        "data/raw/Henry_B._Bigelow/HB2201/EK60/",
        "data/raw/Henry_B._Bigelow/HB2202/EK60/",
        "data/raw/Henry_B._Bigelow/HB2203/EK60/",
        "data/raw/Henry_B._Bigelow/HB2204/EK60/",
        "data/raw/Henry_B._Bigelow/HB2205/EK60/",
        "data/raw/Henry_B._Bigelow/HB2206/EK60/",
    ]

    # TODO: for each verify ek60 datagram
    # index.get_raw_files_csv(
    #     ship_name='Henry_B._Bigelow',
    #     cruise_name='HB2206',
    #     sensor_name='EK60'
    # )
    # for iii in bigelow_cruises:
    for iii in ek60_cruise_prefixes:
        print(iii)
        s_n = iii.split("/")[2]
        c_n = iii.split("/")[3]
        ### get raw file to scan datagram ###
        select_key = index_manager.get_raw_files(
            ship_name=s_n, cruise_name=c_n, sensor_name="EK60"
        )[0]
        ### check if datagram is ek60 ###
        datagram = index_manager.scan_datagram(select_key=select_key)
        if datagram == "CON0":  # if ek60
            print(f"{c_n} is ek60")
            ### create csv file with all raw file paths ###
            index_manager.get_raw_files_csv(
                ship_name=s_n, cruise_name=c_n, sensor_name="EK60"
            )
        else:
            print(f"{c_n} is not ek60")
    # all_raw_files = index.get_raw_files(ship_name='Bell_M._Shimada', cruise_name='SH1906', sensor_name='EK60')
    # 'data/raw/Bell_M._Shimada/SH1204/EK60/'
    # all_ek60_data = index.index()
    # print(all_ek60_data)


#######################################################

# TODO: for post analysis of coverage
#  need to check each cruise has same number of files in noaa-wcsd-pds and noaa-wcsd-model-pds buckets
