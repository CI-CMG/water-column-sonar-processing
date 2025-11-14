import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from water_column_sonar_processing.aws import DynamoDBManager
from water_column_sonar_processing.geometry import GeometryManager
from water_column_sonar_processing.model import ZarrManager

warnings.simplefilter("ignore", category=RuntimeWarning)


class ResampleRegrid:
    #######################################################
    def __init__(
        self,
    ):
        self.__overwrite = True
        # self.input_bucket_name = os.environ.get("INPUT_BUCKET_NAME")
        # self.output_bucket_name = os.environ.get("OUTPUT_BUCKET_NAME")
        self.dtype = "float32"

    #################################################################
    def interpolate_data(
        self,
        input_xr,
        ping_times,
        all_cruise_depth_values,  # includes water_level offset
        water_level,  # this is the offset that will be added to each respective file
    ) -> np.ndarray:
        """
        What gets passed into interpolate data
        """
        print("Interpolating dataset.")
        try:
            data = np.empty(
                (
                    len(all_cruise_depth_values),
                    len(ping_times),
                    len(input_xr.frequency_nominal),
                ),
                dtype=self.dtype,
            )

            data[:] = np.nan

            regrid_resample = xr.DataArray(  # where data will be written to
                data=data,
                dims=("depth", "time", "frequency"),
                coords={
                    "depth": all_cruise_depth_values,
                    "time": ping_times,
                    "frequency": input_xr.frequency_nominal.values,
                },
            )

            # shift the input data by water_level
            input_xr.echo_range.values = (
                input_xr.echo_range.values + water_level
            )  # water_level # TODO: change

            channels = input_xr.channel.values
            for channel in range(
                len(channels)
            ):  # ?TODO: leaving off here, need to subset for just indices in time axis
                gc.collect()
                max_depths = np.nanmax(
                    a=input_xr.echo_range.sel(channel=input_xr.channel[channel]).values,
                    # + water_level,
                    axis=1,
                )
                superset_of_max_depths = set(
                    max_depths
                )  # HB1501, D20150503-T102035.raw, TypeError: unhashable type: 'numpy.ndarray'
                set_of_max_depths = list(
                    {x for x in superset_of_max_depths if x == x}
                )  # removes nan's
                # iterate through partitions of dataset with similar depths and resample
                for select_max_depth in set_of_max_depths:
                    # TODO: for nan just skip and leave all nan's
                    select_indices = [
                        i
                        for i in range(0, len(max_depths))
                        if max_depths[i] == select_max_depth
                    ]

                    # now create new DataArray with proper dimension and indices
                    # data_select = input_xr.Sv.sel(
                    #     channel=input_xr.channel[channel]
                    # ).values[select_indices, :].T  # TODO: dont like this transpose
                    data_select = input_xr.Sv.sel(channel=input_xr.channel[channel])[
                        select_indices, :
                    ].T.values
                    # change from ".values[select_indices, :].T" to "[select_indices, :].values.T"

                    times_select = input_xr.ping_time.values[select_indices]
                    depths_select = input_xr.echo_range.sel(
                        channel=input_xr.channel[channel]
                    ).values[
                        select_indices[0], :
                    ]  # '0' because all others in group should be same

                    da_select = xr.DataArray(
                        data=data_select,
                        dims=("depth", "time"),
                        coords={
                            "depth": depths_select,
                            "time": times_select,
                        },
                    ).dropna(dim="depth")
                    resampled = da_select.interp(
                        depth=all_cruise_depth_values, method="nearest"
                    )
                    # write to the resample array
                    regrid_resample.loc[
                        dict(
                            time=times_select,
                            frequency=input_xr.frequency_nominal.values[channel],
                        )
                    ] = resampled
                    print(f"updated {len(times_select)} ping times")
                    gc.collect()
        except Exception as err:
            raise RuntimeError(f"Problem finding the dynamodb table, {err}")
        print("Done interpolating dataset.")
        return regrid_resample.values.copy()

    #################################################################
    def resample_regrid(
        self,
        ship_name,
        cruise_name,
        sensor_name,
        table_name,
        bucket_name,
        override_select_files=None,
        # override_cruise_min_epsilon=None,
        endpoint_url=None,
    ) -> None:
        """
        The goal here is to interpolate the dataset against the depth values already populated
        in the existing file level model stores. We open the cruise-level store with model for
        read/write operations. We open the file-level store with Xarray to leverage tools for
        resampling and subsetting the dataset.
        """
        print("Resample Regrid, Interpolating dataset.")
        try:
            zarr_manager = ZarrManager()
            geo_manager = GeometryManager()

            output_zarr_store = zarr_manager.open_s3_zarr_store_with_zarr(
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                output_bucket_name=bucket_name,
                endpoint_url=endpoint_url,
            )

            # get dynamo stuff
            dynamo_db_manager = DynamoDBManager()
            cruise_df = dynamo_db_manager.get_table_as_df(
                # ship_name=ship_name,
                cruise_name=cruise_name,
                # sensor_name=sensor_name,
                table_name=table_name,
            )

            #########################################################
            #########################################################
            all_file_names = cruise_df["FILE_NAME"]

            if override_select_files is not None:
                all_file_names = override_select_files

            # Iterate files
            for file_name in all_file_names:
                gc.collect()
                file_name_stem = Path(file_name).stem
                print(f"Processing file: {file_name_stem}.")

                if f"{file_name_stem}.raw" not in list(cruise_df["FILE_NAME"]):
                    raise Exception("Raw file file_stem not found in dynamodb.")

                # status = PipelineStatus['LEVEL_1_PROCESSING']
                # TODO: filter rows by enum success, filter the dataframe just for enums >= LEVEL_1_PROCESSING
                #  df[df['PIPELINE_STATUS'] < PipelineStatus.LEVEL_1_PROCESSING] = np.nan

                # Get index from all cruise files. Note: should be based on which are included in cruise.
                index = int(
                    cruise_df.index[cruise_df["FILE_NAME"] == f"{file_name_stem}.raw"][
                        0
                    ]
                )

                # Get input store — this is unadjusted for water_level
                input_xr_zarr_store = zarr_manager.open_s3_zarr_store_with_xarray(
                    ship_name=ship_name,
                    cruise_name=cruise_name,
                    sensor_name=sensor_name,
                    file_name_stem=file_name_stem,
                    input_bucket_name=bucket_name,
                    endpoint_url=endpoint_url,
                )

                # This is the vertical offset of the sensor related to the ocean surface
                # See https://echopype.readthedocs.io/en/stable/data-proc-additional.html
                if "water_level" in input_xr_zarr_store.keys():
                    water_level = input_xr_zarr_store.water_level.values
                else:
                    water_level = 0.0
                #########################################################################
                # [3] Get needed time indices — along the x-axis
                # Offset from start index to insert new dataset. Note that missing values are excluded.
                ping_time_cumsum = np.insert(
                    np.cumsum(
                        cruise_df["NUM_PING_TIME_DROPNA"].dropna().to_numpy(dtype=int)
                    ),
                    obj=0,
                    values=0,
                )
                start_ping_time_index = ping_time_cumsum[index]
                end_ping_time_index = ping_time_cumsum[index + 1]

                max_echo_range = np.max(
                    (cruise_df["MAX_ECHO_RANGE"] + cruise_df["WATER_LEVEL"])
                    .dropna()
                    .astype(float)
                )
                cruise_min_epsilon = np.min(
                    cruise_df["MIN_ECHO_RANGE"].dropna().astype(float)
                )

                # Note: cruise dims (depth, time, frequency)
                all_cruise_depth_values = zarr_manager.get_depth_values(  # needs to integrate water_level
                    # min_echo_range=min_echo_range,
                    max_echo_range=max_echo_range,  # does it here
                    cruise_min_epsilon=cruise_min_epsilon,  # remove this & integrate into min_echo_range
                )  # with offset of 7.5 meters, 0 meter measurement should now start at 7.5 meters

                print(" ".join(list(input_xr_zarr_store.Sv.dims)))
                if set(input_xr_zarr_store.Sv.dims) != {
                    "channel",
                    "ping_time",
                    "range_sample",
                }:
                    raise Exception("Xarray dimensions are not as expected.")

                indices, geospatial = geo_manager.read_s3_geo_json(
                    ship_name=ship_name,
                    cruise_name=cruise_name,
                    sensor_name=sensor_name,
                    file_name_stem=file_name_stem,
                    input_xr_zarr_store=input_xr_zarr_store,
                    endpoint_url=endpoint_url,
                    output_bucket_name=bucket_name,
                )

                input_xr = input_xr_zarr_store.isel(
                    ping_time=indices
                )  # Problem with HB200802-D20080310-T174959.zarr/

                ping_times = input_xr.ping_time.values
                # Date format: numpy.datetime64('2007-07-20T02:10:25.845073920') converts to "1184897425.845074"
                epoch_seconds = [
                    (pd.Timestamp(i) - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
                    for i in ping_times
                ]
                output_zarr_store["time"][start_ping_time_index:end_ping_time_index] = (
                    epoch_seconds
                )

                # --- UPDATING --- #
                regrid_resample = self.interpolate_data(
                    input_xr=input_xr,
                    ping_times=ping_times,
                    all_cruise_depth_values=all_cruise_depth_values,  # should accommodate the water_level already
                    water_level=water_level,  # not applied to anything yet
                )

                print(
                    f"start_ping_time_index: {start_ping_time_index}, end_ping_time_index: {end_ping_time_index}"
                )
                #########################################################################
                # write Sv values to cruise-level-model-store

                for fff in range(regrid_resample.shape[-1]):
                    output_zarr_store["Sv"][
                        :, start_ping_time_index:end_ping_time_index, fff
                    ] = regrid_resample[:, :, fff]
                #########################################################################
                # TODO: add the "detected_seafloor_depth/" to the
                #  L2 cruise dataarrays
                # TODO: make bottom optional
                # TODO: Only checking the first channel for now. Need to average across all channels
                #  in the future. See https://github.com/CI-CMG/water-column-sonar-processing/issues/11
                if "detected_seafloor_depth" in input_xr.variables:
                    print(
                        "Found detected_seafloor_depth, adding dataset to output store."
                    )
                    detected_seafloor_depth = input_xr.detected_seafloor_depth.values
                    detected_seafloor_depth[detected_seafloor_depth == 0.0] = np.nan
                    # TODO: problem here: Processing file: D20070711-T210709.

                    # Use the lowest frequencies to determine bottom
                    detected_seafloor_depths = detected_seafloor_depth[0, :]

                    detected_seafloor_depths[detected_seafloor_depths == 0.0] = np.nan
                    print(f"min depth measured: {np.nanmin(detected_seafloor_depths)}")
                    print(f"max depth measured: {np.nanmax(detected_seafloor_depths)}")
                    # available_indices = np.argwhere(np.isnan(geospatial['latitude'].values))
                    output_zarr_store["bottom"][
                        start_ping_time_index:end_ping_time_index
                    ] = detected_seafloor_depths
                #
                #########################################################################
                # [5] write subset of latitude/longitude
                output_zarr_store["latitude"][
                    start_ping_time_index:end_ping_time_index
                ] = geospatial.dropna()[
                    "latitude"
                ].values  # TODO: get from ds_sv directly, dont need geojson anymore
                output_zarr_store["longitude"][
                    start_ping_time_index:end_ping_time_index
                ] = geospatial.dropna()["longitude"].values
                #########################################################################
                #########################################################################
        except Exception as err:
            raise RuntimeError(f"Problem with resample_regrid, {err}")
        finally:
            print("Exiting resample_regrid.")
            # TODO: read across times and verify dataset was written?

    #######################################################


###########################################################
