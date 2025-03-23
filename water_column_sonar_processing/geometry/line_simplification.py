# import json
import geopandas as gpd
import numpy as np
from pykalman import KalmanFilter
from shapely.geometry import Point

# import matplotlib.pyplot as plt


# lambda for timestamp in form "yyyy-MM-ddTHH:mm:ssZ"
# dt = lambda: datetime.now().isoformat(timespec="seconds") + "Z"

# TODO: get line for example HB1906 ...save linestring to array for testing

MAX_SPEED_KNOTS = 50


# Lambert's formula ==> better accuracy than haversinte
# Lambert's formula (the formula used by the calculators above) is the method used to calculate the shortest distance along the surface of an ellipsoid. When used to approximate the Earth and calculate the distance on the Earth surface, it has an accuracy on the order of 10 meters over thousands of kilometers, which is more precise than the haversine formula.


def mph_to_knots(mph_value):
    # 1 mile per hour === 0.868976 Knots
    return mph_value * 0.868976


# https://shapely.readthedocs.io/en/stable/reference/shapely.MultiLineString.html#shapely.MultiLineString
class LineSimplification:
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
        private static final int SRID = 8307;
        private static final double simplificationTolerance = 0.0001;
        private static final long splitGeometryMs = 900000L;
        private static final int batchSize = 10000;
        private static final int geoJsonPrecision = 5;
        final int geoJsonPrecision = 5;
        final double simplificationTolerance = 0.0001;
        final int simplifierBatchSize = 3000;
        final long maxCount = 0;
        private static final double maxAllowedSpeedKnts = 60D;
    """

    # TODO: in the future move to standalone library
    #######################################################
    def __init__(
        self,
    ):
        pass

    #######################################################
    def kalman_filter(
        self,
        longitudes,
        latitudes,
    ) -> (np.ndarray, np.ndarray):
        """
        # TODO: need to use masked array to get the right number of values
        """
        ### https://github.com/pykalman/pykalman
        # https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data
        measurements = np.asarray([list(elem) for elem in zip(longitudes, latitudes)])
        initial_state_mean = [measurements[0, 0], 0, measurements[0, 1], 0]
        transition_matrix = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
        observation_matrix = [[1, 0, 0, 0], [0, 0, 1, 0]]

        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=initial_state_mean,
        )
        kf = kf.em(measurements, n_iter=2)  # TODO: 5
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

        # plt.plot(longitudes, latitudes, label="original")
        # plt.plot(smoothed_state_means[:, 0], smoothed_state_means[:, 2], label="smoothed")
        # plt.legend()
        # plt.show()

        return smoothed_state_means[:, [0, 2]]

    #######################################################
    def get_speeds(
        self,
        times: np.ndarray,  # don't really need time, do need to segment the data first
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        print(MAX_SPEED_KNOTS)  # TODO: too high
        print(times[0], latitudes[0], longitudes[0])
        # TODO: distance/time ==> need to take position2 - position1 to get speed

        # get distance difference
        geom = [Point(xy) for xy in zip(longitudes, latitudes)]
        points_df = gpd.GeoDataFrame({"geometry": geom}, crs="EPSG:4326")
        # Conversion to UTM, a rectilinear projection coordinate system where distance can be calculated with pythagorean theorem
        # an alternative could be to use EPSG 32663
        points_df.to_crs(
            epsg=3310, inplace=True
        )  # https://gis.stackexchange.com/questions/293310/finding-distance-between-two-points-with-geoseries-distance
        distance_diffs = points_df.distance(points_df.shift())
        # distance_diffs_sorted = distance_diffs.sort_values(
        #     ascending=False
        # )  # TODO: get avg cutoff time
        #
        time_diffs_ns = np.append(0, (times[1:] - times[:-1]).astype(int))
        # time_diffs_ns_sorted = np.sort(time_diffs_ns)
        # largest time diffs HB0707 [ 17. 17.93749786  21.0781271  54.82812723  85.09374797, 113.56249805 204.87500006 216. 440.68749798 544.81249818]
        # largest diffs HB1906 [3.01015808e+00 3.01016013e+00 3.01017805e+00 3.01018701e+00, 3.01018701e+00 3.01018906e+00 3.01019802e+00 3.01021005e+00, 3.01021005e+00 3.01021414e+00 3.01022208e+00 3.01022899e+00, 3.01024998e+00 3.01025920e+00 3.01026202e+00 3.01028096e+00, 3.01119411e+00 3.01120896e+00 3.01120998e+00 3.01120998e+00, 3.01122099e+00 3.01122790e+00 3.01122790e+00 3.01124506e+00, 3.01125197e+00 3.01128090e+00 3.01142707e+00 3.01219814e+00, 3.01221120e+00 3.01223014e+00 3.01225498e+00 3.01225882e+00, 3.01226010e+00 3.01312998e+00 3.01316096e+00 3.01321190e+00, 3.01321293e+00 3.01322880e+00 3.01322906e+00 3.01323110e+00, 3.01323213e+00 3.01323290e+00 3.01326208e+00 3.01328512e+00, 3.01418112e+00 3.01420109e+00 3.01421107e+00 3.01421184e+00, 3.01421414e+00 3.01424819e+00 3.01512883e+00 3.01516006e+00, 3.01524198e+00 3.01619917e+00 3.01623194e+00 3.01623296e+00, 3.01917594e+00 3.01921408e+00 3.01921587e+00 3.02022195e+00, 3.02025216e+00 3.02121702e+00 3.02325811e+00 3.02410291e+00, 3.02421914e+00 3.02426701e+00 3.02523776e+00 3.02718694e+00, 3.02927590e+00 3.03621606e+00 3.03826304e+00 3.34047514e+00, 3.36345114e+00 3.39148595e+00 4.36819302e+00 4.50157901e+00, 4.50315699e+00 4.50330598e+00 4.50333491e+00 4.50428416e+00, 4.50430490e+00 4.50430694e+00 4.50526387e+00 4.50530790e+00, 4.50530995e+00 4.50532301e+00 4.50533478e+00 4.50629402e+00, 4.50730701e+00 4.50825882e+00 4.50939008e+00 6.50179098e+00, 2.25025029e+01 1.39939425e+02 1.54452331e+02 1.60632653e+03, 1.74574667e+05 4.33569587e+05 4.35150475e+05 8.00044883e+05]
        nanoseconds_per_second = 1e9
        speed_meters_per_second = (
            distance_diffs / time_diffs_ns * nanoseconds_per_second
        )
        # returns the speed in meters per second #TODO: get speed in knots
        return speed_meters_per_second.to_numpy(dtype="float32")  # includes nan

    def remove_null_island_values(
        self,
        epsilon=1e-5,
    ) -> None:
        # TODO: low priority
        print(epsilon)
        pass

    def break_linestring_into_multi_linestring(
        self,
    ) -> None:
        # TODO: medium priority
        # For any line-strings across the antimeridian, break into multilinestring
        # average cadence is measurements every 1 second
        # break when over 1 minute
        pass

    def simplify(
        self,
    ) -> None:
        # TODO: medium-high priority
        pass

    #######################################################


###########################################################
