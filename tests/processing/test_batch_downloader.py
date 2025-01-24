import numpy as np
import pytest

from water_column_sonar_processing.processing.batch_downloader import BatchDownloader


#######################################################
def setup_module():
    print("setup")


def teardown_module():
    print("teardown")


#######################################################
@pytest.mark.skip(reason="Running very slow for some reason")
def test_get_toy_batch_generator():
    np.random.seed(0)

    batch_downloader = BatchDownloader()
    batch_generator = batch_downloader.get_toy_batch_generator()
    for da_batch in batch_generator:
        assert da_batch.Sv.shape == (10, 10, 3) # depth, time, freq

        print(f"depth-start: {da_batch.depth.values[0]}, depth-end: {da_batch.depth.values[-1]}")
        assert np.isclose(da_batch.depth.values[0], 1)
        assert np.isclose(da_batch.depth.values[-1], 10)

        print(f"time-start: {da_batch.time.values[0]}, time-end: {da_batch.time.values[-1]}")
        assert da_batch.time.values[0] == np.datetime64("2025-01-01T00:00:00")

        print(f"frequency-start: {da_batch.frequency.values[0]}, frequency-end: {da_batch.frequency.values[-1]}")
        assert da_batch.frequency.values[0] == 1_000

        sample_sv_mean = np.round(np.nanmean(da_batch.Sv.values), 2)
        print(f"Batch Sv mean: {sample_sv_mean}")

        assert np.isclose(sample_sv_mean, 0.52)

        break # Only testing the first batch


@pytest.mark.skip(reason="Running very slow for some reason")
def test_get_s3_batch_generator():
    """
    This is a functional test getting real data from s3 and checking batch values.
    """
    batch_downloader = BatchDownloader(
        bucket_name="noaa-wcsd-zarr-pds",
        ship_name="Henry_B._Bigelow",
        cruise_name="HB0707",
        sensor_name="EK60",
    )

    batch_generator = batch_downloader.get_s3_batch_generator()
    print(f"number_of_batches_in_cruise: {list(batch_generator._batch_selectors.selectors)}")
    for da_batch in batch_generator:
        assert da_batch.Sv.shape == (10, 10, 4)

        print(f"depth-start: {da_batch.depth.values[0]}, depth-end: {da_batch.depth.values[-1]}")
        assert np.isclose(da_batch.depth.values[0], 0.19)
        assert np.isclose(da_batch.depth.values[-1], 1.90)

        print(f"time-start: {da_batch.time.values[0]}, time-end: {da_batch.time.values[-1]}")
        # Rounded to the nearest second
        assert da_batch.time.values[0].astype('datetime64[s]') == np.datetime64("2007-07-11T18:20:33.657573888").astype('datetime64[s]')

        print(f"frequency-start: {da_batch.frequency.values[0]}, frequency-end: {da_batch.frequency.values[-1]}")
        assert da_batch.frequency.values[0] == 18_000.
        assert da_batch.frequency.values[-1] == 200_000.

        sample_sv_mean = np.round(np.nanmean(da_batch.Sv.values), 2)
        print(f"mean: {sample_sv_mean}")
        assert np.isclose(sample_sv_mean, -53.7)

        print(' ______ ')
        # break # Only testing the first batch
        break

@pytest.mark.skip(reason="Running very slow for some reason")
def testget_s3_manual_batch_generator():
    """
    This is a functional test getting real data from s3 and checking batch values.
    """
    batch_downloader = BatchDownloader(
        bucket_name="noaa-wcsd-zarr-pds",
        ship_name="Henry_B._Bigelow",
        cruise_name="HB0707",
        sensor_name="EK60",
    )

    batch_generator = batch_downloader.get_s3_manual_batch_generator()
    for da_batch in batch_generator:
        assert da_batch.Sv.shape == (10, 10, 4)

        print(f"depth-start: {da_batch.depth.values[0]}, depth-end: {da_batch.depth.values[-1]}")
        assert np.isclose(da_batch.depth.values[0], 0.19)
        assert np.isclose(da_batch.depth.values[-1], 1.90)

        print(f"time-start: {da_batch.time.values[0]}, time-end: {da_batch.time.values[-1]}")
        # Rounded to the nearest second
        assert da_batch.time.values[0].astype('datetime64[s]') == np.datetime64("2007-07-11T18:20:33.657573888").astype('datetime64[s]')

        print(f"frequency-start: {da_batch.frequency.values[0]}, frequency-end: {da_batch.frequency.values[-1]}")
        assert da_batch.frequency.values[0] == 18_000.
        assert da_batch.frequency.values[-1] == 200_000.

        sample_sv_mean = np.round(np.nanmean(da_batch.Sv.values), 2)
        print(f"mean: {sample_sv_mean}")
        assert np.isclose(sample_sv_mean, -53.7)

        break # Only testing the first batch

#######################################################
