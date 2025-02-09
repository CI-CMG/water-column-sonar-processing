import pandas as pd

from water_column_sonar_processing.geometry import ElevationManager


#######################################################
def setup_module():
    print("setup")


def teardown_module():
    print("teardown")


#######################################################
# TODO: mock the api response... this fails if the api is down :(
def test_get_arcgis_elevation():
    # coordinates with known elevation
    lat = [48.633, 48.733, 45.1947, 45.1962, 13.03332]
    lon = [-93.9667, -94.6167, -93.3257, -93.2755, -31.70235]

    df = pd.DataFrame({"lat": lat, "lon": lon})

    elevation_manager = ElevationManager()
    elevations = elevation_manager.get_arcgis_elevation(
        lngs=df["lon"].tolist(), lats=df["lat"].tolist(), chunk_size=2
    )

    assert elevations[0] == 339.112
    assert elevations[-1] == -5733.0


#######################################################
