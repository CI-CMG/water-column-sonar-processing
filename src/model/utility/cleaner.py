import os
import glob
import shutil


###########################################################
class Cleaner:
    @staticmethod
    def delete_local_files(
            file_types=['*.raw*', '*.zarr']  # '*.json'
    ):
        print('Deleting all local raw and zarr files')
        for i in file_types:
            for j in glob.glob(i):
                if os.path.isdir(j):
                    shutil.rmtree(j, ignore_errors=True)
                elif os.path.isfile(j):
                    os.remove(j)
        print('done deleting')

###########################################################