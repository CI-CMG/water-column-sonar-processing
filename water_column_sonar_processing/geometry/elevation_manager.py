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
import time

import requests
from collections.abc import Generator

def chunked(
    ll: list,
    n: int
) -> Generator:
    # Yields successively n-sized chunks from ll.
    for i in range(0, len(ll), n):
        yield ll[i : i + n]


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
            lngs: list,
            lats: list,
            chunk_size: int=500, # I think this is the api limit
    ) -> int:
        # Reference: https://developers.arcgis.com/rest/services-reference/enterprise/map-to-image/
        # Info: https://www.arcgis.com/home/item.html?id=c876e3c96a8642ab8557646a3b4fa0ff
        ### 'https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/identify?geometry={"points":[[-31.70235,13.03332],[-32.70235,14.03332]]}&geometryType=esriGeometryMultipoint&returnGeometry=false&returnCatalogItems=false&f=json'
        if len(lngs) != len(lats):
            raise ValueError("lngs and lats must have same length")

        geometryType = "esriGeometryMultipoint" # TODO: allow single point?

        depths = []

        list_of_points = [list(elem) for elem in list(zip(lngs, lats))]
        for chunk in chunked(list_of_points, chunk_size):
            time.sleep(0.1)
            # order: (lng, lat)
            geometry = f'{{"points":{str(chunk)}}}'
            url=f'https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/identify?geometry={geometry}&geometryType={geometryType}&returnGeometry=false&returnCatalogItems=false&f=json'
            result = requests.get(url, timeout=self.TIMOUT_SECONDS)
            res = json.loads(result.content.decode('utf8'))
            if 'results' in res:
                for element in res['results']:
                    depths.append(float(element['value']))
            elif 'value' in res:
                depths.append(float(res['value']))

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
