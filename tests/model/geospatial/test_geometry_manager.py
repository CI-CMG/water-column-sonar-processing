from pathlib import Path
import geopandas
import numpy as np
import shapely
import xarray as xr
import pandas as pd
import echopype as ep
# from s3fs import S3Map, S3FileSystem
# from moto import mock_s3
from dotenv import load_dotenv, find_dotenv
from src.model.aws.s3_manager import S3Manager
from src.model.geospatial.geometry_manager import GeoManager


#######################################################
def setup_module(module):
    print('setup')
    # env_file = find_dotenv('.env-test')
    env_file = find_dotenv('.env-prod')
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module(module):
    print('teardown')

#######################################################

# @mock_s3
def test_geometry_manager(tmp_path):
    bucket_name = 'noaa-wcsd-pds'
    # file_name = 'D20070719-T232718.raw'  # too big
    # file_name = 'D20070720-T224031.raw'  # has >4 points in dataset
    file_name = 'D20070724-T042400.raw'
    # file_name_stem = Path(file_name).stem
    ship_name = 'Henry_B._Bigelow'
    cruise_name = 'HB0706'
    sensor_name = 'EK60'

    s3_path = f"s3://{bucket_name}/data/raw/{ship_name}/{cruise_name}/{sensor_name}/{file_name}"
    # s3_path = f"r2d2-testing-level-2-data/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr"

    print(s3_path)

    echodata = ep.open_raw(
        raw_file=s3_path,
        sonar_model=sensor_name,
        use_swap=True,
        storage_options={'anon': True}
    )

    geo_manager = GeoManager()

    time, lat, lon = geo_manager.read_echodata_gps_data(
        echodata=echodata,
        ship_name=ship_name,
        cruise_name=cruise_name,
        sensor_name=sensor_name,
        file_name=file_name,
        write_geojson=False,
    )
    # NOTE CHECK FOR NULL ISLAND ON RETURN
    null_island_indices = list(
        set.intersection(set(np.where(np.abs(lat) < 1e-3)[0]), set(np.where(np.abs(lon) < 1e-3)[0]))
    )
    lat[null_island_indices] = np.nan
    lon[null_island_indices] = np.nan

    assert len(time) == 36
    assert len(lat) == 36
    assert len(lat[~np.isnan(lat)]) == 35
    assert len(lon) == 36

#######################################################
