import os
from pathlib import Path

import geopandas
import numpy as np
import pandas as pd

from water_column_sonar_processing.aws import S3Manager
from water_column_sonar_processing.utility import Cleaner

"""
//  [Decimal / Places / Degrees	/ Object that can be recognized at scale / N/S or E/W at equator, E/W at 23N/S, E/W at 45N/S, E/W at 67N/S]
  //  0   1.0	        1° 00′ 0″	        country or large region                             111.32 km	  102.47 km	  78.71 km	43.496 km
  //  1	  0.1	        0° 06′ 0″         large city or district                              11.132 km	  10.247 km	  7.871 km	4.3496 km
  //  2	  0.01	      0° 00′ 36″        town or village                                     1.1132 km	  1.0247 km	  787.1 m	  434.96 m
  //  3	  0.001	      0° 00′ 3.6″       neighborhood, street                                111.32 m	  102.47 m	  78.71 m	  43.496 m
  //  4	  0.0001	    0° 00′ 0.36″      individual street, land parcel                      11.132 m	  10.247 m	  7.871 m	  4.3496 m
  //  5	  0.00001	    0° 00′ 0.036″     individual trees, door entrance	                    1.1132 m	  1.0247 m	  787.1 mm	434.96 mm
  //  6	  0.000001	  0° 00′ 0.0036″    individual humans                                   111.32 mm	  102.47 mm	  78.71 mm	43.496 mm
  //  7	  0.0000001	  0° 00′ 0.00036″   practical limit of commercial surveying	            11.132 mm	  10.247 mm	  7.871 mm	4.3496 mm
"""


class GeometryManager:
    #######################################################
    def __init__(
        self,
    ):
        self.DECIMAL_PRECISION = 5  # precision for GPS coordinates
        self.SIMPLIFICATION_TOLERANCE = 0.0001  # RDP simplification to "street level"

    #######################################################
    def read_echodata_gps_data(
        self,
        echodata,
        output_bucket_name,
        ship_name,
        cruise_name,
        sensor_name,
        file_name,
        endpoint_url=None,
        write_geojson=True,
    ) -> tuple:
        file_name_stem = Path(file_name).stem
        geo_json_name = f"{file_name_stem}.json"

        print("Getting GPS data from echopype object.")
        try:
            latitude = np.round(
                echodata.platform.latitude.values, self.DECIMAL_PRECISION
            )
            longitude = np.round(
                echodata.platform.longitude.values, self.DECIMAL_PRECISION
            )

            # RE: time coordinates: https://github.com/OSOceanAcoustics/echopype/issues/656#issue-1219104771
            # 'nmea_times' are times from the nmea datalogger associated with GPS
            #   note that nmea_times, unlike time1, can be sorted
            nmea_times = np.sort(echodata.platform.time1.values)

            # 'time1' are times from the echosounder associated with the data of the transducer measurement
            time1 = echodata.environment.time1.values

            if len(nmea_times) < len(time1):
                raise Exception( # TODO: explore this logic further...
                    "Problem: Not enough NMEA times available to extrapolate time1."
                )

            # Align 'sv_times' to 'nmea_times'
            if not (
                np.all(time1[:-1] <= time1[1:])
                and np.all(nmea_times[:-1] <= nmea_times[1:])
            ):
                raise Exception("Problem: NMEA times are not sorted.")

            # Finds the indices where 'v' can be inserted just to the right of 'a'
            indices = np.searchsorted(a=nmea_times, v=time1, side="right") - 1
            lat = latitude[indices]
            lat[indices < 0] = np.nan  # values recorded before indexing are set to nan
            lon = longitude[indices]
            lon[indices < 0] = np.nan

            if not (
                np.all(lat[~np.isnan(lat)] >= -90.0)
                and np.all(lat[~np.isnan(lat)] <= 90.0)
                and np.all(lon[~np.isnan(lon)] >= -180.0)
                and np.all(lon[~np.isnan(lon)] <= 180.0)
            ):
                raise Exception("Problem: GPS Data falls outside allowed bounds.")

            # check for visits to null island
            null_island_indices = list(
                set.intersection(
                    set(np.where(np.abs(lat) < 1e-3)[0]),
                    set(np.where(np.abs(lon) < 1e-3)[0]),
                )
            )
            lat[null_island_indices] = np.nan
            lon[null_island_indices] = np.nan

            # create requirement for minimum linestring size
            MIN_ALLOWED_SIZE = (
                4  # don't want to process files with less than 4 data points
            )
            if (
                len(lat[~np.isnan(lat)]) < MIN_ALLOWED_SIZE
                or len(lon[~np.isnan(lon)]) < MIN_ALLOWED_SIZE
            ):
                raise Exception(
                    f"There was not enough data in lat or lon to create geojson, {len(lat[~np.isnan(lat)])} found, less than {MIN_ALLOWED_SIZE}."
                )

            # https://osoceanacoustics.github.io/echopype-examples/echopype_tour.html
            gps_df = (
                pd.DataFrame({"latitude": lat, "longitude": lon, "time": time1})
                .set_index(["time"])
                .fillna(0)
            )

            # Note: We set np.nan to 0,0 so downstream missing values can be omitted
            gps_gdf = geopandas.GeoDataFrame(
                gps_df,
                geometry=geopandas.points_from_xy(
                    gps_df["longitude"], gps_df["latitude"]
                ),
                crs="epsg:4326",
            )
            # Note: We set np.nan to 0,0 so downstream missing values can be omitted
            # TODO: so what ends up here is data with corruption at null island!!!
            geo_json_line = gps_gdf.to_json()
            if write_geojson:
                print("Creating local copy of geojson file.")
                with open(geo_json_name, "w") as write_file:
                    write_file.write(geo_json_line) # NOTE: this file can include zeros for lat lon

                geo_json_prefix = (
                    f"spatial/geojson/{ship_name}/{cruise_name}/{sensor_name}"
                )

                print("Checking s3 and deleting any existing GeoJSON file.")
                s3_manager = S3Manager(endpoint_url=endpoint_url)
                geojson_object_exists = s3_manager.check_if_object_exists(
                    bucket_name=output_bucket_name,
                    key_name=f"{geo_json_prefix}/{geo_json_name}"
                )
                if geojson_object_exists:
                    print("GeoJSON already exists in s3, deleting existing and continuing.")
                    s3_manager.delete_nodd_object(bucket_name=output_bucket_name, key_name=f"{geo_json_prefix}/{geo_json_name}")

                print("Upload GeoJSON to s3.")
                s3_manager.upload_nodd_file(
                    file_name=geo_json_name,  # file_name
                    key=f"{geo_json_prefix}/{geo_json_name}",  # key
                    output_bucket_name=output_bucket_name,
                )

                # TODO: delete geo_json file
                cleaner = Cleaner()
                cleaner.delete_local_files(file_types=["*.json"])

            #################################################################
            # TODO: simplify with shapely
            # linestring = shapely.geometry.LineString(
            #     [xy for xy in zip(gps_gdf.longitude, gps_gdf.latitude)]
            # )
            # len(linestring.coords)
            # line_simplified = linestring.simplify(
            #     tolerance=self.SIMPLIFICATION_TOLERANCE,
            #     preserve_topology=True
            # )
            # print(f"Total number of points for original linestring: {len(linestring.coords)}")
            # print(f"Total number of points needed for the simplified linestring: {len(line_simplified.coords)}")
            # print(line_simplified)
            # geo_json_line_simplified = shapely.to_geojson(line_simplified)
            #################################################################
            # GeoJSON FeatureCollection with IDs as "time"
        except Exception as err:
            print(
                f"Exception encountered extracting gps coordinates creating geojson: {err}"
            )
            raise
        # Note: returned lat/lon values can include np.nan because they need to be aligned with
        # the Sv data! GeoJSON needs simplification but has been filtered.
        # return gps_df.index.values, gps_df.latitude.values, gps_df.longitude.values
        return gps_df.index.values, lat, lon
        # TODO: if geojson is already returned with 0,0, the return here
        #  can include np.nan values?

    #######################################################
    def read_s3_geo_json(
        self,
        ship_name,
        cruise_name,
        sensor_name,
        file_name_stem,
        input_xr_zarr_store,
        endpoint_url,
        output_bucket_name,
    ):
        try:
            s3_manager = S3Manager(endpoint_url=endpoint_url)
            geo_json = s3_manager.read_s3_json(
                ship_name=ship_name,
                cruise_name=cruise_name,
                sensor_name=sensor_name,
                file_name_stem=file_name_stem,
                output_bucket_name=output_bucket_name,

            )
            ###
            geospatial = geopandas.GeoDataFrame.from_features(
                geo_json["features"]
            ).set_index(pd.json_normalize(geo_json["features"])["id"].values)
            null_island_indices = list(
                set.intersection(
                    set(np.where(np.abs(geospatial.latitude.values) < 1e-3)[0]),
                    set(np.where(np.abs(geospatial.longitude.values) < 1e-3)[0]),
                )
            )
            geospatial.iloc[null_island_indices] = np.nan
            ###
            geospatial_index = geospatial.dropna().index.values.astype("datetime64[ns]")
            aa = input_xr_zarr_store.ping_time.values.tolist()
            vv = geospatial_index.tolist()
            indices = np.searchsorted(a=aa, v=vv)

            return indices, geospatial
        except Exception as err:  # Failure
            print(f"Exception encountered reading s3 GeoJSON: {err}")
            raise

    ############################################################################
    # COMES from the raw-to-zarr conversion
    def __write_geojson_to_file(
            self,
            store_name,
            data
    ) -> None:
        print('Writing GeoJSON to file.')
        with open(os.path.join(store_name, 'geo.json'), "w") as outfile:
            outfile.write(data)


###########################################################
