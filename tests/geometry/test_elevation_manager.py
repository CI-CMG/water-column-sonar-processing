import pandas as pd

from water_column_sonar_processing.geometry import ElevationManager


#######################################################
def setup_module():
    print("setup")

def teardown_module():
    print("teardown")


#######################################################

# @mock_s3
def test_get_arcgis_elevation():
    # coordinates with known elevation
    lat = [48.633, 48.733, 45.1947, 45.1962, 13.03332]
    lon = [-93.9667, -94.6167, -93.3257, -93.2755, -31.70235]

    # create data frame
    df = pd.DataFrame({
        'lat': lat,
        'lon': lon
    })
    elevation_manager = ElevationManager()
    elevations = elevation_manager.get_arcgis_elevation(lngs=df['lon'].tolist(), lats=df['lat'].tolist())
    print(elevations)

    assert elevations[0] > 0.


#######################################################
