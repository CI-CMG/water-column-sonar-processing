### https://xarray-datatree.readthedocs.io/en/latest/data-structures.html
import xarray as xr
from datatree import DataTree


class DatatreeManager:
    #######################################################
    def __init__(
        self,
    ):
        self.dtype = "float32"

    #################################################################
    def create_datatree(
        self,
        input_ds,
    ) -> None:
        ds1 = xr.Dataset({"foo": "orange"})
        dt = DataTree(name="root", data=ds1)  # create root node
        # ds2 = xr.Dataset({"bar": 0}, coords={"y": ("y", [0, 1, 2])})
        return dt
