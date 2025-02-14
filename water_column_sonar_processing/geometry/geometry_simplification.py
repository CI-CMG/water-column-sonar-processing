# import json


# lambda for timestamp in form "yyyy-MM-ddTHH:mm:ssZ"
# dt = lambda: datetime.now().isoformat(timespec="seconds") + "Z"

# https://shapely.readthedocs.io/en/stable/reference/shapely.MultiLineString.html#shapely.MultiLineString
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

"""
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


class GeometrySimplification:
    # TODO: in the future move to standalone library
    #######################################################
    def __init__(
        self,
    ):
        pass

    #######################################################
    def speed_check(
        self,
        speed_knots=50,
    ) -> None:
        print(speed_knots)
        pass

    def remove_null_island_values(
        self,
        epsilon=1e-5,
    ) -> None:
        print(epsilon)
        pass

    def stream_geometry(
        self,
    ) -> None:
        pass

    def break_linestring_into_multi_linestring(
        self,
    ) -> None:
        # For any line-strings across the antimeridian, break into multilinestring
        pass

    def simplify(
        self,
    ) -> None:
        pass

    def kalman_filter(self):
        # for cruises with bad signal, filter so that
        pass

    #######################################################


###########################################################
