'''
Reads data in .nc format and converts this to .npy files.
'''

# -- Built-in modules -- #
import os
from os import listdir

# -- Third-party modules -- #
import numpy as np
import xarray as xr

# --Proprietary modules -- #
from init import OPTIONS

read_path = OPTIONS['path_to_processed_data']
write_path = OPTIONS['path_to_npy_data']

SAR_VARIABLES = ['sar_primary', 'sar_secondary']
NERSC_VARIABLES = ['nersc_sar_primary', 'nersc_sar_secondary']


def get_all_files(mypath):
    """Returns all files in a directory"""
    from os.path import isfile, join
    return [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith('20')]


def get_x_sar(scene):
    """Returns x data from a .nc SAR scene"""
    x_patch = scene[SAR_VARIABLES].to_array().values
    x_patch = np.moveaxis(x_patch, 0, -1)
    return x_patch


def get_x_nersc(scene):
    """Returns x data from a .nc NERSC scene"""
    x_patch = scene[NERSC_VARIABLES].to_array().values
    x_patch = np.moveaxis(x_patch, 0, -1)
    return x_patch


def get_y(scene):
    """Returns y data from a .nc scene"""
    chart = 'SIC'
    data = scene[chart].values
    return data.astype(np.uint8)


def convert(file_name: str):
    """Converts a .nc file from the read path into a .npy file in the write path with name file_name"""
    scene = xr.open_dataset(os.path.join(read_path, file_name))  # Open the scene.

    patch_x = get_x_nersc(scene)
    patch_y = get_y(scene)

    scene_name = file_name.removesuffix('_pro.nc')

    np.save(os.path.join(write_path, scene_name + '-nersc-x.npy'), patch_x)
    np.save(os.path.join(write_path, scene_name + '-y.npy'), patch_y)

    patch_x = get_x_sar(scene)
    np.save(os.path.join(write_path, scene_name + '-sar-x.npy'), patch_x)

    print('converted file', file_name)


files = get_all_files(read_path)

count = 0
for file in files:
    count += 1
    print('Converting: ', count, '...')
    convert(file)

print('Done!')
