"""``pytest`` configuration."""

import pytest
from pathlib import Path

# from echopype.testing import TEST_DATA_FOLDER
HERE = Path(__file__).parent.absolute()
TEST_DATA_FOLDER = HERE / "test_resources"


@pytest.fixture(scope="session")
def dump_output_dir():
    return TEST_DATA_FOLDER / "dump"


@pytest.fixture(scope="session")
def test_path():
    return {
        'RAW_TO_ZARR_TEST_PATH': TEST_DATA_FOLDER / "raw_to_zarr",
        'INDEX_TEST_PATH': TEST_DATA_FOLDER / "index",
    }
