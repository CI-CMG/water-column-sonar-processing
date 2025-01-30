from enum import Enum, Flag, unique


@unique
class Constants(Flag):
    TILE_SIZE = 1024

    # Average https://noaa-wcsd-zarr-pds.s3.us-east-1.amazonaws.com/level_2/Henry_B._Bigelow/HB0902/EK60/HB0902.zarr/time/927
    # chunk size is ~1.3 kB, HB0902 cruise takes ~30 seconds to load all time/lat/lon data
    # NOTE: larger value here will speed up the TurfJS download of data in the UI
    SPATIOTEMPORAL_CHUNK_SIZE = int(2**16) # 2**17


class Coordinates(Enum):
    """
    Should try to specify
        dtype
        units
        long_name — most readable description of variable
        standard_name — name in lowercase and snake_case
    """

    PROJECT_NAME = "echofish"

    DEPTH = "depth"
    DEPTH_DTYPE = "float32"
    DEPTH_UNITS = "m"  # TODO: Pint? <https://pint.readthedocs.io/en/stable/>
    DEPTH_LONG_NAME = "Depth below surface"
    DEPTH_STANDARD_NAME = "depth"

    TIME = "time"
    TIME_DTYPE = "float64"
    # Note: units and calendar are used downstream by Xarray
    TIME_UNITS = "seconds since 1970-01-01 00:00:00"
    TIME_LONG_NAME = "Timestamp of each ping"
    TIME_STANDARD_NAME = "time"
    TIME_CALENDAR = "proleptic_gregorian"
    # TODO: create test for reading out timestamps in Xarray

    FREQUENCY = "frequency"
    FREQUENCY_DTYPE = "uint"
    FREQUENCY_UNITS = "Hz"
    FREQUENCY_LONG_NAME = "Transducer frequency"
    FREQUENCY_STANDARD_NAME = "sound_frequency"

    LATITUDE = "latitude"
    LATITUDE_DTYPE = "float32"
    LATITUDE_UNITS = "degrees_north"
    LATITUDE_LONG_NAME = "Latitude"
    LATITUDE_STANDARD_NAME = "latitude"

    LONGITUDE = "longitude"
    LONGITUDE_DTYPE = "float32"
    LONGITUDE_UNITS = "degrees_east"
    LONGITUDE_LONG_NAME = "Longitude"
    LONGITUDE_STANDARD_NAME = "longitude"

    BOTTOM = "bottom"
    BOTTOM_DTYPE = "float32"
    BOTTOM_UNITS = "m"
    BOTTOM_LONG_NAME = "Detected sea floor depth"
    BOTTOM_STANDARD_NAME = "bottom"

    SV = "Sv"
    SV_DTYPE = "float32"  # TODO: experiment with dtype of int
    SV_UNITS = "dB"
    SV_LONG_NAME = "Volume backscattering strength (Sv re 1 m-1)"
    SV_STANDARD_NAME = "volume_backscattering_strength"
