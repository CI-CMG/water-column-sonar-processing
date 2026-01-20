import gc
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from water_column_sonar_processing.aws import DynamoDBManager
from water_column_sonar_processing.model import ZarrManager

warnings.simplefilter("ignore", category=RuntimeWarning)


class ResampleRegrid:
    #######################################################
    def __init__(
        self,
    ):
        self.__overwrite = True
        self.dtype = "float32"

    #################################################################
    def interpolate_data(
        self,
        input_xr: xr.Dataset,
        ping_times: np.ndarray,
        all_cruise_depth_values: np.ndarray,  # includes water_level offset
        water_level: float = 0.0,
    ) -> np.ndarray:
        """
        Input dataset is passed in along with times and depth values to regrid to.
        """
        print("Interpolating dataset.")
        try:
            # add offset for the water level to the whole input xarray
            input_xr.depth.values = input_xr.depth.values + water_level

            data = np.empty(
                (  # Depth / Time / Frequency
                    len(all_cruise_depth_values),
                    len(ping_times),
                    len(input_xr.frequency_nominal.values),
                ),
                dtype=self.dtype,
            )

            data[:] = np.nan

            regrid_resample = xr.DataArray(  # where data will be written to
                data=data,
                coords={
                    "depth": all_cruise_depth_values,
                    "time": ping_times,
                    "frequency": input_xr.frequency_nominal.values,
                },
                dims=("depth", "time", "frequency"),
                name="Sv",
            )

            channels = input_xr.channel.values
            for channel in range(len(channels)):
                gc.collect()
                max_depths = np.nanmax(
                    a=input_xr.depth.sel(channel=input_xr.channel[channel]).values,
                    # + water_level,
                    axis=1,
                )
                superset_of_max_depths = set(max_depths)
                set_of_max_depths = list(
                    {x for x in superset_of_max_depths if x == x}
                )  # To speed things up resample in groups denoted by max_depth -- so samples might no longer be adjacent
                for select_max_depth in set_of_max_depths:
                    # TODO: for nan just skip and leave all nan's
                    select_indices = [
                        i
                        for i in range(0, len(max_depths))
                        if max_depths[i] == select_max_depth
                    ]

                    data_select = input_xr.Sv.sel(channel=input_xr.channel[channel])[
                        select_indices, :
                    ].T.values

                    times_select = input_xr.ping_time.values[select_indices]
                    # input_xr.depth[0][0] -> [0., 499.9] before
                    # input_xr.depth.values = input_xr.depth.values + water_level  # issue here!! overwritting all the data
                    # input_xr.depth[0][0] -> [7.5, 507.40] after
                    depths_all = input_xr.depth.sel(
                        channel=input_xr.channel[channel],
                        ping_time=input_xr.ping_time[select_indices[0]],
                    ).values
                    depths_select = depths_all[~np.isnan(depths_all)]
                    #
                    da_select = xr.DataArray(
                        data=data_select[: len(depths_select), :],
                        dims=("depth", "time"),
                        coords={
                            "depth": depths_select,
                            "time": times_select,
                        },
                    )
                    # 'resampled' is now the interpolated superset of new dimensions
                    resampled = da_select.interp(  # need to define the data with water level (domain)
                        depth=all_cruise_depth_values,  # and need to interpolate over the (range)
                        method="nearest",
                        assume_sorted=True,
                    )  # good through here, @27 is -3.11 which is 5.4 m depth

                    ### write to outptut ###
                    regrid_resample.loc[  # ~150 MB for 5001x7706x4
                        dict(
                            time=times_select,
                            frequency=input_xr.frequency_nominal.values[channel],
                        )
                    ] = resampled
                    # print(f"updated {len(times_select)} ping times")
                    gc.collect()
            return regrid_resample.values.copy()
        except Exception as err:
            raise RuntimeError(f"Problem finding the dynamodb table, {err}")
        finally:
            gc.collect()
            print("Done interpolating dataset.")

    #################################################################
    def resample_regrid(
        self,
        ship_name,
        cruise_name,
        sensor_name,
        table_name,
        bucket_name,
        override_select_files=None,
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

            output_zarr_store = zarr_manager.open_s3_zarr_store_with_zarr(
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                output_bucket_name=bucket_name,
                endpoint_url=endpoint_url,
            )

            dynamo_db_manager = DynamoDBManager()
            cruise_df = dynamo_db_manager.get_table_as_df(
                cruise_name=cruise_name,
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
                    print("Raw file file_stem not found in dynamodb.")
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

                # Get input store
                input_xr_zarr_store = zarr_manager.open_s3_zarr_store_with_xarray(
                    ship_name=ship_name,
                    cruise_name=cruise_name,
                    sensor_name=sensor_name,
                    file_name_stem=file_name_stem,
                    bucket_name=bucket_name,
                    endpoint_url=endpoint_url,
                )

                #########################################################################
                # This is the vertical offset of the sensor related to the ocean surface
                # See https://echopype.readthedocs.io/en/stable/data-proc-additional.html
                if "water_level" in input_xr_zarr_store.keys():
                    water_level = float(input_xr_zarr_store.water_level.values)
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

                max_echo_range = np.max(  # Should water level go in here?
                    (cruise_df["MAX_ECHO_RANGE"] + cruise_df["WATER_LEVEL"])
                    .dropna()
                    .astype(np.float32)
                )
                # cruise_min_epsilon = np.min(
                #     cruise_df["MIN_ECHO_RANGE"].dropna().astype(float)
                # ) # TODO: currently overwriting to 0.25 m

                all_cruise_depth_values = zarr_manager.get_depth_values(
                    max_echo_range=max_echo_range,
                    # cruise_min_epsilon=cruise_min_epsilon,
                )

                if set(
                    input_xr_zarr_store.Sv.dims
                ) != {  # Cruise dimensions are: (depth, time, frequency)
                    "channel",
                    "ping_time",
                    "range_sample",
                }:
                    raise Exception("Xarray dimensions are not as expected.")

                # indices, geospatial = geo_manager.read_s3_geo_json(  # TODO: remove this!!!!
                #     ship_name=ship_name,
                #     cruise_name=cruise_name,
                #     sensor_name=sensor_name,
                #     file_name_stem=file_name_stem,
                #     input_xr_zarr_store=input_xr_zarr_store,
                #     endpoint_url=endpoint_url,
                #     output_bucket_name=bucket_name,
                # )

                input_xr = input_xr_zarr_store  # .isel(ping_time=indices)

                ping_times = input_xr.ping_time.values
                output_zarr_store["time"][start_ping_time_index:end_ping_time_index] = (
                    input_xr.ping_time.data
                )

                # --- UPDATING --- # # TODO: problem, this returns dimensionless array
                regrid_resample = self.interpolate_data(
                    input_xr=input_xr,
                    ping_times=ping_times,
                    all_cruise_depth_values=all_cruise_depth_values,  # should accommodate the water_level already
                    water_level=water_level,
                )

                print(
                    f"start_ping_time_index: {start_ping_time_index}, end_ping_time_index: {end_ping_time_index}"
                )
                #########################################################################
                # write Sv values to cruise-level-model-store

                for fff in range(regrid_resample.shape[-1]):
                    output_zarr_store["Sv"][
                        : regrid_resample[:, :, fff].shape[0],
                        start_ping_time_index:end_ping_time_index,
                        fff,
                    ] = regrid_resample[:, :, fff]
                #########################################################################
                #  in the future. See https://github.com/CI-CMG/water-column-sonar-processing/issues/11
                if "detected_seafloor_depth" in list(input_xr.variables):
                    print("Adding detected_seafloor_depth to output")
                    detected_seafloor_depth = input_xr.detected_seafloor_depth.values
                    detected_seafloor_depth[detected_seafloor_depth == 0.0] = np.nan

                    # As requested, use the lowest frequencies to determine bottom
                    detected_seafloor_depths = detected_seafloor_depth[0, :]

                    detected_seafloor_depths[detected_seafloor_depths == 0.0] = np.nan
                    print(f"min depth measured: {np.nanmin(detected_seafloor_depths)}")
                    print(f"max depth measured: {np.nanmax(detected_seafloor_depths)}")
                    output_zarr_store["bottom"][
                        start_ping_time_index:end_ping_time_index
                    ] = detected_seafloor_depths
                #
                #########################################################################
                # [5] write subset of latitude/longitude
                # output_zarr_store["latitude"][
                #     start_ping_time_index:end_ping_time_index
                # ] = geospatial.dropna()[
                #     "latitude"
                # ].values  # TODO: get from ds_sv directly, dont need geojson anymore
                # output_zarr_store["longitude"][
                #     start_ping_time_index:end_ping_time_index
                # ] = geospatial.dropna()["longitude"].values
                #########################################################################
                output_zarr_store["latitude"][
                    start_ping_time_index:end_ping_time_index
                ] = input_xr_zarr_store.latitude.dropna(dim="ping_time").values
                output_zarr_store["longitude"][
                    start_ping_time_index:end_ping_time_index
                ] = input_xr_zarr_store.longitude.dropna(dim="ping_time").values
                #########################################################################
        except Exception as err:
            raise RuntimeError(f"Problem with resample_regrid, {err}")
        finally:
            print("Exiting resample_regrid.")
            # TODO: read across times and verify dataset was written?
            gc.collect()

    #######################################################


###########################################################
