import pytest
from dotenv import find_dotenv, load_dotenv

from water_column_sonar_processing.geometry import PMTileGeneration


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


# @mock_aws
@pytest.mark.skip(
    reason="no way of currently testing this without accessing actual zarr stores"
)
def test_pmtile_generator():
    pmtile_generation = PMTileGeneration()
    pmtile_generation.create_collection_geojson()
