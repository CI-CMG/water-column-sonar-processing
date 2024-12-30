from pathlib import Path

import pooch
import pytest

HERE = Path(__file__).parent.absolute()
TEST_DATA_FOLDER = HERE / "test_resources"


HB0707_RAW = pooch.create(
    path=pooch.os_cache("water-column-sonar-processing"),
    base_url="https://noaa-wcsd-pds.s3.amazonaws.com/data/raw/Henry_B._Bigelow/HB0707/EK60/",
    retry_if_failed=1,
    registry={
        # "ad2cp.zip": "sha256:8c0e45451eca31b478e7ba9d265fc1bb5257045d30dc50fc5299d2df2abe8430",
        # "azfp.zip": "sha256:6b2067e18e71e7752b768cb84284fef3e0d82b5b775ea2499d52df8202936415",
        #https://noaa-wcsd-zarr-pds.s3.amazonaws.com/level_1/Henry_B._Bigelow/HB0707/EK60/D20070711-T182032.zarr/
        #https://noaa-wcsd-pds.s3.amazonaws.com/data/raw/Henry_B._Bigelow/HB0707/EK60/D20070711-T182032.raw
        #"D20070711-T182032.raw": "sha256:2db05a529053d79e564aa1e85481c61de389f9d48889760080eaed1b7608dec1",
        #"D20070712-T061745.raw": "sha256:89b0069c760ffda68bb2539cfea56521ae537e5b830fd1924dfff410883854ef",
        #"D20070712-T004447.raw": "sha256:4b8ddac65c236f86e65884f638d5a0df3184fd534c8e27f8bd1c7a8e56586cf8",
        ### https://noaa-wcsd-pds.s3.amazonaws.com/data/raw/Henry_B._Bigelow/HB0707/EK60/D20070712-T100505.raw
        #"D20070712-T100505.raw": "sha256:9851012f7753d449bf58b4492f37860cf8c6cf870b9d12df9301ab5fc091cb64", # 250 m
        # TODO: add bottom files
        "D20070712-T124906.raw": "sha256:44f9b2402a8d6d51c69235d1e33c3e4ab570fc541e9f269009924378bf4d97a2", # 250 m, 158 MB
        "D20070712-T152416.raw": "sha256:94a937eefd6ae5763c27c9ba1e4769b2b76fcc2d840e7db6c2e0edd925d6f70f", # 1000 m, 200 MB
        #"D20070712-T201647.raw": "sha256:e5fa948877dcf123c28cdaba8a9e83c1fed881602232add044fa464297da322b", # 250 m
        #"D20070712-T231759.raw": "sha256:0a5f83992d101056656895d6444907b350784d846941728a00d63aecd92d3c02",
        #"D20070712-T033431.raw": "sha256:059546953293bb80ff24096d8ea504140db97e265fa777d31679404010223d67",

    },
)

def fetch_raw_files():
    # Only need to test two for successful across two depths
    fname1 = HB0707_RAW.fetch(fname="D20070712-T124906.raw", progressbar=True)
    fname2 = HB0707_RAW.fetch(fname="D20070712-T152416.raw", progressbar=True)
    #fname3 = HB0707_RAW.fetch(fname="D20070712-T201647.raw", progressbar=True)
    return Path(fname2).parent #joinpath(Path(file_path).stem)

@pytest.fixture(scope="session")
def test_path():
    return {
        'RAW_TO_ZARR_TEST_PATH': TEST_DATA_FOLDER / "raw_to_zarr",
        'INDEX_TEST_PATH': TEST_DATA_FOLDER / "index",
        'ZARR_MANAGER_TEST_PATH': TEST_DATA_FOLDER / "zarr_manager",
        'PMTILE_GENERATION_TEST_PATH': TEST_DATA_FOLDER / "pmtile",
        'CREATE_EMPTY_ZARR_TEST_PATH': TEST_DATA_FOLDER / "create_empty_zarr",
        # 'RESAMPLE_REGRID_TEST_PATH': TEST_DATA_FOLDER / "resample_regrid",
        'RESAMPLE_REGRID_TEST_PATH': fetch_raw_files(),
        'S3FS_MANAGER_TEST_PATH': TEST_DATA_FOLDER / "s3fs_manager",
        'S3_MANAGER_TEST_PATH': TEST_DATA_FOLDER / "s3_manager",
    }

# """
# Windows
# C:\Users\<user>\AppData\Local\echopype\Cache\2024.12.23.10.10
# MacOS
# /Users//Library/Caches/echopype/2024.12.23.10.10
# """
