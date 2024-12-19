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
        'ZARR_MANAGER_TEST_PATH': TEST_DATA_FOLDER / "zarr_manager",
        'PMTILE_GENERATION_TEST_PATH': TEST_DATA_FOLDER / "pmtile",
        'CREATE_EMPTY_ZARR_TEST_PATH': TEST_DATA_FOLDER / "create_empty_zarr",
        'RESAMPLE_REGRID_TEST_PATH': TEST_DATA_FOLDER / "resample_regrid",
        'S3FS_MANAGER_TEST_PATH': TEST_DATA_FOLDER / "s3fs_manager",
        'S3_MANAGER_TEST_PATH': TEST_DATA_FOLDER / "s3_manager",
    }