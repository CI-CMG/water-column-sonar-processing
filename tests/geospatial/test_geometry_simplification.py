import pytest
import unittest
# import geopandas
# import shapely.geometry as geom
# import xarray as xr
# import pandas as pd
# from s3fs import S3Map, S3FileSystem
# from moto import mock_s3
from dotenv import load_dotenv, find_dotenv

# import os

# s3fs.core.setup_logging("DEBUG")


#######################################################
class TestGeometrySimplification(unittest.TestCase):
    def setup_module(module):
        print('setup')
        pass

    @pytest.fixture(scope='session', autouse=True)
    def load_env(self):
        # env_file = find_dotenv('.env-test')
        env_file = find_dotenv('.env-test')
        load_dotenv(dotenv_path=env_file, override=True)

    def teardown_module(module):
        print('teardown')

    # @mock_s3
    def test_geometry_simplification(self):
        bucket_name = 'noaa-wcsd-zarr_manager-pds'
        pass

#######################################################
