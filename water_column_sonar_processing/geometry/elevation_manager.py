"""
https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/identify?geometry=-31.70235%2C13.03332&geometryType=esriGeometryPoint&returnGeometry=false&returnCatalogItems=false&f=json

https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/
    identify?
        geometry=-31.70235%2C13.03332
        &geometryType=esriGeometryPoint
        &returnGeometry=false
        &returnCatalogItems=false
        &f=json
{"objectId":0,"name":"Pixel","value":"-5733","location":{"x":-31.702349999999999,"y":13.03332,"spatialReference":{"wkid":4326,"latestWkid":4326}},"properties":null,"catalogItems":null,"catalogItemVisibilities":[]}
-5733

(base) rudy:deleteME rudy$ curl https://api.opentopodata.org/v1/gebco2020?locations=13.03332,-31.70235
{
  "results": [
    {
      "dataset": "gebco2020",
      "elevation": -5729.0,
      "location": {
        "lat": 13.03332,
        "lng": -31.70235
      }
    }
  ],
  "status": "OK"
}
"""
import json
import requests


class ElevationManager:
    #######################################################
    def __init__(
        self,
    ):
        self.DECIMAL_PRECISION = 5  # precision for GPS coordinates
        self.TIMOUT_SECONDS = 10

    #######################################################
    def get_arcgis_elevation(
            self,
            lngs,
            lats,
    ) -> int:
        # Reference: https://developers.arcgis.com/rest/services-reference/enterprise/map-to-image/
        # Info: https://www.arcgis.com/home/item.html?id=c876e3c96a8642ab8557646a3b4fa0ff
        ### 'https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/identify?geometry={"points":[[-31.70235,13.03332],[-32.70235,14.03332]]}&geometryType=esriGeometryMultipoint&returnGeometry=false&returnCatalogItems=false&f=json'
        geometryType = "esriGeometryMultipoint" # TODO: allow single point?

        # TODO: break up into requests of 500

        # TODO: convert lists to zipped lists to strings
        list_of_points = [list(elem) for elem in list(zip(lngs, lats))]
        print(list_of_points)

        # lng = -31.70235
        # lat = 13.03332
        # lng, lat
        geometry = f'{{"points":{str(list_of_points)}}}'
        url=f'https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/identify?geometry={geometry}&geometryType={geometryType}&returnGeometry=false&returnCatalogItems=false&f=json'
        result = requests.get(url, timeout=self.TIMOUT_SECONDS)
        foo = json.loads(result.content.decode('utf8'))
        depths=[]
        for element in foo['results']:
            depths.append(float(element['value']))

        print(depths)
        # TODO: combine smaller lists
        # needs to be broken down into blocks of 500
        return depths

    # def get_gebco_bathymetry_elevation(self) -> int:
    #     # Documentation: https://www.opentopodata.org/datasets/gebco2020/
    #     latitude = 13.03332
    #     longitude = -31.70235
    #     dataset = "gebco2020"
    #     url = f"https://api.opentopodata.org/v1/{dataset}?locations={latitude},{longitude}"
    #     pass

    # def get_elevation(
    #         self,
    #         df,
    #         lat_column,
    #         lon_column,
    # ) -> int:
    #     """Query service using lat, lon. add the elevation values as a new column."""
    #     url = r'https://epqs.nationalmap.gov/v1/json?'
    #     elevations = []
    #     for lat, lon in zip(df[lat_column], df[lon_column]):
    #         # define rest query params
    #         params = {
    #             'output': 'json',
    #             'x': lon,
    #             'y': lat,
    #             'units': 'Meters'
    #         }
    #         result = requests.get((url + urllib.parse.urlencode(params)))
    #         elevations.append(result.json()['value'])
    #     return elevations
