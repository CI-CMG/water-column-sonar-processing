import gc
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import xarray as xr

from water_column_sonar_processing.aws import DynamoDBManager
from water_column_sonar_processing.geometry import GeometryManager
from water_column_sonar_processing.model import ZarrManager

numcodecs.blosc.use_threads = False
numcodecs.blosc.set_nthreads(1)


# TODO: when ready switch to version 3 of model spec
#  ZARR_V3_EXPERIMENTAL_API = 1
#  creates the latlon data: foo = ep.consolidate.add_location(ds_Sv, echodata)


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
        all_cruise_depth_values,
    ) -> np.ndarray:
        print("Interpolating data.")
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

            regrid_resample = xr.DataArray(
                data=data,
                dims=("depth", "time", "frequency"),
                coords={
                    "depth": all_cruise_depth_values,
                    "time": ping_times,
                    "frequency": input_xr.frequency_nominal.values,
                },
            )

            channels = input_xr.channel.values
            for channel in range(
                len(channels)
            ):  # TODO: leaving off here, need to subset for just indices in time axis
                print(
                    np.nanmax(
                        input_xr.echo_range.sel(
                            channel=input_xr.channel[channel]
                        ).values
                    )
                )
                #
                max_depths = np.nanmax(
                    a=input_xr.echo_range.sel(channel=input_xr.channel[channel]).values,
                    axis=1,
                )
                superset_of_max_depths = set(
                    np.nanmax(
                        input_xr.echo_range.sel(
                            channel=input_xr.channel[channel]
                        ).values,
                        1,
                    )
                )
                set_of_max_depths = list(
                    {x for x in superset_of_max_depths if x == x}
                )  # removes nan's
                # iterate through partitions of data with similar depths and resample
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
        except Exception as err:
            print(f"Problem finding the dynamodb table: {err}")
            raise err
        print("Done interpolating data.")
        return regrid_resample

    #################################################################
    def resample_regrid(
        self,
        ship_name,
        cruise_name,
        sensor_name,
        table_name,
        # TODO: file_name?,
        bucket_name, # TODO: this is the same bucket
        override_select_files=None,
        endpoint_url=None
    ) -> None:
        """
        The goal here is to interpolate the data against the depth values already populated
        in the existing file level model stores. We open the cruise-level store with model for
        read/write operations. We open the file-level store with Xarray to leverage tools for
        resampling and subsetting the data.
        """
        print("Resample Regrid, Interpolating data.")
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
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
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

                if f"{file_name_stem}.raw" not in list(cruise_df['FILE_NAME']):
                    raise Exception(f"Raw file file_stem not found in dynamodb.")

                # status = PipelineStatus['LEVEL_1_PROCESSING']
                # TODO: filter rows by enum success, filter the dataframe just for enums >= LEVEL_1_PROCESSING
                #  df[df['PIPELINE_STATUS'] < PipelineStatus.LEVEL_1_PROCESSING] = np.nan

                # Get index from all cruise files. Note: should be based on which are included in cruise.
                index = int(cruise_df.index[
                    cruise_df["FILE_NAME"] == f"{file_name_stem}.raw"
                ][0])

                # get input store
                input_xr_zarr_store = zarr_manager.open_s3_zarr_store_with_xarray(
                    ship_name=ship_name,
                    cruise_name=cruise_name,
                    sensor_name=sensor_name,
                    file_name_stem=file_name_stem,
                    input_bucket_name=bucket_name,
                    endpoint_url=endpoint_url,
                )
                #########################################################################
                # [3] Get needed indices
                # Offset from start index to insert new data. Note that missing values are excluded.
                ping_time_cumsum = np.insert(
                    np.cumsum(
                        cruise_df["NUM_PING_TIME_DROPNA"].dropna().to_numpy(dtype=int)
                    ),
                    obj=0,
                    values=0,
                )
                start_ping_time_index = ping_time_cumsum[index]
                end_ping_time_index = ping_time_cumsum[index + 1]

                min_echo_range = np.nanmin(np.float32(cruise_df["MIN_ECHO_RANGE"]))
                max_echo_range = np.nanmax(np.float32(cruise_df["MAX_ECHO_RANGE"]))

                print("Creating empty ndarray for Sv data.")  # Note: cruise dims (depth, time, frequency)
                output_zarr_store_shape = output_zarr_store.Sv.shape
                end_ping_time_index - start_ping_time_index
                output_zarr_store_height = output_zarr_store_shape[0]
                output_zarr_store_width = end_ping_time_index - start_ping_time_index
                output_zarr_store_depth = output_zarr_store_shape[2]
                cruise_sv_subset = np.empty(
                    shape=(output_zarr_store_height, output_zarr_store_width, output_zarr_store_depth)
                )
                cruise_sv_subset[:, :, :] = np.nan

                all_cruise_depth_values = zarr_manager.get_depth_values(
                    min_echo_range=min_echo_range,
                    max_echo_range=max_echo_range
                ) # (5262,) and

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

                input_xr = input_xr_zarr_store.isel(ping_time=indices)

                ping_times = input_xr.ping_time.values
                # Date format: numpy.datetime64('2007-07-20T02:10:25.845073920') converts to "1184897425.845074"
                epoch_seconds = [
                    (pd.Timestamp(i) - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
                    for i in ping_times
                ]
                output_zarr_store.time[start_ping_time_index:end_ping_time_index] = (
                    epoch_seconds
                )

                # --- UPDATING --- #
                regrid_resample = self.interpolate_data(
                    input_xr=input_xr,
                    ping_times=ping_times,
                    all_cruise_depth_values=all_cruise_depth_values,
                )

                print(f"start_ping_time_index: {start_ping_time_index}, end_ping_time_index: {end_ping_time_index}")
                #########################################################################
                # write Sv values to cruise-level-model-store
                output_zarr_store.Sv[:, start_ping_time_index:end_ping_time_index, :] = regrid_resample.values
                #########################################################################
                # TODO: add the "detected_seafloor_depth/" to the
                #  L2 cruise dataarrays
                # TODO: make bottom optional
                # TODO: Only checking the first channel for now. Need to average across all channels
                #  in the future. See https://github.com/CI-CMG/water-column-sonar-processing/issues/11
                if 'detected_seafloor_depth' in input_xr.variables:
                    print('Found detected_seafloor_depth, adding data to output store.')
                    detected_seafloor_depth = input_xr.detected_seafloor_depth.values
                    detected_seafloor_depth[detected_seafloor_depth == 0.] = np.nan
                    # TODO: problem here: Processing file: D20070711-T210709.
                    detected_seafloor_depths = np.nanmean(detected_seafloor_depth, 0) # RuntimeWarning: Mean of empty slice detected_seafloor_depths = np.nanmean(detected_seafloor_depth, 0)
                    detected_seafloor_depths[detected_seafloor_depths == 0.] = np.nan
                    print(f"min depth measured: {np.nanmin(detected_seafloor_depths)}")
                    print(f"max depth measured: {np.nanmax(detected_seafloor_depths)}")
                    #available_indices = np.argwhere(np.isnan(geospatial['latitude'].values))
                    output_zarr_store.bottom[
                        start_ping_time_index:end_ping_time_index
                    ] = detected_seafloor_depths
                #
                #########################################################################
                # [5] write subset of latitude/longitude
                output_zarr_store.latitude[
                    start_ping_time_index:end_ping_time_index
                ] = geospatial.dropna()["latitude"].values # TODO: get from ds_sv directly, dont need geojson anymore
                output_zarr_store.longitude[
                    start_ping_time_index:end_ping_time_index
                ] = geospatial.dropna()["longitude"].values
                #########################################################################
                #########################################################################
        except Exception as err:
            print(f"Problem interpolating the data: {err}")
            raise err
        finally:
            print("Done interpolating data.")
            # TODO: read across times and verify data was written?

    #######################################################


###########################################################
