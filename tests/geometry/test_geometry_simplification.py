from dotenv import find_dotenv, load_dotenv

# s3fs.core.setup_logging("DEBUG")


#######################################################
def setup_module():
    print("setup")
    # env_file = find_dotenv('.env-test')
    env_file = find_dotenv(".env-test")
    load_dotenv(dotenv_path=env_file, override=True)


def teardown_module():
    print("teardown")


# class TestGeometrySimplification(unittest.TestCase):
#     def setup_module(module):
#         print('setup')
#         pass
#
#     @pytest.fixture(scope='session', autouse=True)
#     def load_env(self):
#         # env_file = find_dotenv('.env-test')
#         env_file = find_dotenv('.env-test')
#         load_dotenv(dotenv_path=env_file, override=True)
#
#     def teardown_module(module):
#         print('teardown')


# @mock_s3
def test_geometry_simplification():
    # bucket_name = 'noaa-wcsd-model-pds'
    pass


#######################################################
