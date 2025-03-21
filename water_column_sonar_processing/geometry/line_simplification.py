# import json
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# import matplotlib.pyplot as plt


# lambda for timestamp in form "yyyy-MM-ddTHH:mm:ssZ"
# dt = lambda: datetime.now().isoformat(timespec="seconds") + "Z"

# TODO: get line for example HB1906 ...save linestring to array for testing

MAX_SPEED_KNOTS = 50


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

    def mph_to_knots(self, mph_value):
        # 1mph === 0.868976 Knots
        return mph_value * 0.868976

    # TODO: in the future move to standalone library
    #######################################################
    def __init__(
        self,
    ):
        pass

    #######################################################
    #######################################################
    def speed_check(
        self,
        times,  # don't really need time, do need to segment the data first
        latitudes,
        longitudes,
    ) -> None:  # 90,000 points
        # TODO: high priority
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
        #
        # get time differences, TODO: get speed in knots
        # np.array(foo[1:]) - np.array(foo[:-1])# 1 second difference
        # numpy.timedelta64[ns] => (times[1:] - times[:-1]).astype(int)
        time_diffs_ns = np.append(0, (times[1:] - times[:-1]).astype(int))
        # time_diffs = [{x, y} for x, y in zip(longitudes, latitudes)]
        nanoseconds_per_second = 1e9
        speed_meters_per_second = (
            distance_diffs / time_diffs_ns * nanoseconds_per_second
        )
        return speed_meters_per_second

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
        pass

    def simplify(
        self,
    ) -> None:
        # TODO: medium-high priority
        pass

    def kalman_filter(
        self,
        times,  # don't really need time, do need to segment the data first
        latitudes,
        longitudes,
    ):
        # TODO: highest priority
        # for cruises with bad signal, filter so that
        # https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
        # plt.rcParams['figure.figsize'] = (10, 8)

        # intial parameters
        n_iter = 50
        sz = (n_iter,)  # size of array
        x = -0.37727  # truth value (typo in example at top of p. 13 calls this z)
        z = np.random.normal(
            x, 0.1, size=sz
        )  # observations (normal about x, sigma=0.1)

        Q = 1e-5  # process variance

        # allocate space for arrays
        xhat = np.zeros(sz)  # a posteri estimate of x
        P = np.zeros(sz)  # a posteri error estimate
        xhatminus = np.zeros(sz)  # a priori estimate of x
        Pminus = np.zeros(sz)  # a priori error estimate
        K = np.zeros(sz)  # gain or blending factor

        R = 0.1**2  # estimate of measurement variance, change to see effect

        # intial guesses
        xhat[0] = 0.0
        P[0] = 1.0

        for k in range(1, n_iter):
            # time update
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q
            # measurement update
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        # plt.figure()
        # plt.plot(z, 'k+', label='noisy measurements')
        # plt.plot(xhat, 'b-', label='a posteri estimate')
        # plt.axhline(x, color='g', label='truth value')
        # plt.legend()
        # plt.title('Estimate vs. iteration step', fontweight='bold')
        # plt.xlabel('Iteration')
        # plt.ylabel('Voltage')

        # plt.figure()
        valid_iter = range(1, n_iter)  # Pminus not valid at step 0
        print(valid_iter)
        # plt.plot(valid_iter, Pminus[valid_iter], label='a priori error estimate')
        # plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
        # plt.xlabel('Iteration')
        # plt.ylabel('$(Voltage)^2$')
        # plt.setp(plt.gca(), 'ylim', [0, .01])
        # plt.show()

    #######################################################


###########################################################
