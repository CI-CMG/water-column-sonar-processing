from datatree import DataTree

ds1 = xr.Dataset({"foo": "orange"})

dt = DataTree(name="root", data=ds1)  # create root node

dt
Out[4]: 
DataTree('root', parent=None)
    Dimensions:  ()
    Data variables:
        foo      <U6 24B 'orange'

