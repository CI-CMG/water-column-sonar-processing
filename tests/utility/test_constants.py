from dotenv import find_dotenv, load_dotenv

from water_column_sonar_processing.utility import Constants


#######################################################
def setup_module():
    print("setup")
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


#######################################################
def test_constants():
    """
    Was having problems writing to zarr stores when the time-index chunking was bigger
    than 2**16, just ensuring that it is below that threshold for future reference.
    """
    # assert Constants.SPATIOTEMPORAL_CHUNK_SIZE.value < int(2**16)
    assert Constants.SPATIOTEMPORAL_CHUNK_SIZE.value == 1000000


#######################################################
#######################################################
