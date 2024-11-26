"""``pytest`` configuration."""

import pytest
from pathlib import Path

# from echopype.testing import TEST_DATA_FOLDER
HERE = Path(__file__).parent.absolute()
TEST_DATA_FOLDER = HERE / "test_data"


@pytest.fixture(scope="session")
def dump_output_dir():
    return TEST_DATA_FOLDER / "dump"


@pytest.fixture(scope="session")
def test_path():
    return {
        'ROOT': TEST_DATA_FOLDER,
        'EA640': TEST_DATA_FOLDER / "ea640",
        'EK60': TEST_DATA_FOLDER / "ek60",
        'EK80': TEST_DATA_FOLDER / "ek80",
        'EK80_NEW': TEST_DATA_FOLDER / "ek80_new",
        'ES60': TEST_DATA_FOLDER / "es60",
        'ES70': TEST_DATA_FOLDER / "es70",
        'ES80': TEST_DATA_FOLDER / "es80",
        'AZFP': TEST_DATA_FOLDER / "azfp",
        'AZFP6': TEST_DATA_FOLDER / "azfp6",
        'AD2CP': TEST_DATA_FOLDER / "ad2cp",
        'EK80_CAL': TEST_DATA_FOLDER / "ek80_bb_with_calibration",
        'EK80_EXT': TEST_DATA_FOLDER / "ek80_ext",
        'ECS': TEST_DATA_FOLDER / "ecs",
    }