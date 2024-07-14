#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import rasterio
from rasterio.merge import merge
from rasterio.plot import show

import pandas as pd
import json
from collections import defaultdict
from shapely.geometry import shape
from shapely.ops import transform as stransform
from rasterio.mask import mask
from affine import Affine
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pyproj
import pickle as pkl
import geojson
import cv2


# In[2]:


def merge_images(filenames):
    src_files_mosaic_final = []
    for item in filenames:
        src_mosaic = rasterio.open(item)
        src_files_mosaic_final.append(src_mosaic)
    mosaic_final, out_trans_final = merge(src_files_mosaic_final)
    outpath_root = (
        "/datacommons/carlsonlab/ld243/mosaic_image/"
        + filenames[0][:8]
        + "_mosaic_final.tif"
    )
    with rasterio.open(
        outpath_root,
        "w",
        driver="Gtiff",
        count=src_mosaic.count,
        height=mosaic_final.shape[1],
        width=mosaic_final.shape[2],
        transform=out_trans_final,
        crs=src_mosaic.crs,
        dtype=src_mosaic.dtypes[0],
    ) as dest:
        dest.write(mosaic_final)
    return outpath_root


# In[3]:


def project_wsg_shape_to_csr(shape, csr):
    project = lambda x, y: pyproj.transform(
        pyproj.Proj(init="epsg:4326"),  # source coordinate system
        pyproj.Proj(init=csr),
        x,
        y,
    )
    return stransform(project, shape)


# In[4]:


def crop_image(exact_shape_dir, outpath_root):
    with open(exact_shape_dir) as data_file:
        geoms = json.loads(data_file.read())
        geoms = geoms["features"][0]["geometry"]
    with rasterio.open(outpath_root) as src:
        projected_shape = project_wsg_shape_to_csr(shape(geoms), src.crs)
        out_image, out_transform = mask(src, [projected_shape], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
    cropped_image_path = outpath_root.split(".")[-2] + "_cropped.tif"
    with rasterio.open(cropped_image_path, "w", **out_meta) as dest:
        dest.write(out_image)
    return cropped_image_path


#%%
def image_filter(sensor_loc: dict, lat: float, lon: float, radius: float) -> bool:
    """

    Weeding out the satellite images satisfied with following conditions:
    center located outside of the circle with ground sensor as the center and radius


    Args:
    sensor_loc = Geographical loaction of ground sensors
    lat: Latitude of the center of satellite image
    lon: Longitude of the center of satellite image
    radius: Radius of the circle with ground sensor as the center. determined by heruistic

    Returns: bool

    """

    for i, j in sensor_loc.items():
        if ((lat - j["lat"]) ** 2 + (lon - j["lon"]) ** 2) ** (1 / 2) <= radius:
            return True
    return False


# In[5]:


def grid_transform(cropped_image_path, grid_size, timestamp, save_path, sensor_loc):
    src_img = rasterio.open(cropped_image_path)
    X_image_mosaic = np.moveaxis(src_img.read(), 0, 2)
    X_image_mosaic_shape = np.shape(X_image_mosaic)
    X_image_mosaic_new = X_image_mosaic[
        : int(np.floor(X_image_mosaic_shape[0] / 100) * 100),
        : int(np.floor(X_image_mosaic_shape[1] / 100) * 100),
        :-1,
    ]
    X_image_mosaic_new_alpha = X_image_mosaic[
        : int(np.floor(X_image_mosaic_shape[0] / 100) * 100),
        : int(np.floor(X_image_mosaic_shape[1] / 100) * 100),
        -1,
    ]
    dim1, dim2 = int(np.shape(X_image_mosaic_new)[0] / grid_size), int(
        np.shape(X_image_mosaic_new)[1] / grid_size
    )

    T0 = src_img.transform
    p1 = pyproj.Proj(src_img.crs)

    cols, rows = np.meshgrid(
        np.arange(grid_size / 2, X_image_mosaic_new.shape[1], grid_size),
        np.arange(grid_size / 2, X_image_mosaic_new.shape[0], grid_size),
    )
    if cols[0][-1] + grid_size / 2 > X_image_mosaic_new.shape[1]:
        cols = cols[:][:-1]
        rows = rows[:][:-1]
    if rows[-1][0] + grid_size / 2 > X_image_mosaic_new.shape[0]:
        cols = cols[:-1][:]
        rows = rows[:-1][:]

    # Transform np.meshgrid to grid with latitudes and longitudes
    rc2en = lambda r, c: (c, r) * T0
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)
    p2 = pyproj.Proj(proj="latlong", datum="WGS84")
    lons, lats = pyproj.transform(p1, p2, eastings, northings)

    #     if os.path.isfile(save_path):
    #         with open(save_path, 'rb') as fp:
    #             grids = pkl.load(fp)
    #     else:
    #         grids = []

    grids_one_day = []
    for i in range(dim1):
        for j in range(dim2):
            grid_temp = X_image_mosaic_new[
                i * grid_size : (i + 1) * grid_size,
                j * grid_size : (j + 1) * grid_size,
                :,
            ]
            alpha_temp = X_image_mosaic_new_alpha[
                i * grid_size : (i + 1) * grid_size, j * grid_size : (j + 1) * grid_size
            ]
            score = np.mean(alpha_temp == 255)
            if score >= 0.5 and image_filter(
                sensor_loc, lats[i][j], lons[i][j], radius=0.08
            ):
                grids_one_day.append(
                    {
                        "Image": grid_temp,
                        "lat": lats[i][j],
                        "lon": lons[i][j],
                        "time": timestamp,
                    }
                )

    #     grids.append(grids_one_day)
    #     del grids_one_day
    if len(grids_one_day) > 1:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  #####
        with open(save_path, "ab") as fp:
            pkl.dump(grids_one_day, fp)

    del grids_one_day


# In[6]:


def main():
    sensor_loc = pd.read_pickle(
        r"/datacommons/carlsonlab/ld243/images/geometry_info/sensor_locations.pkl"
    )
    mydates = defaultdict(list)
    for file in os.listdir("/datacommons/carlsonlab/ld243/images/"):
        mydates[file[:8]].append(file)

    for date, filenames in tqdm(mydates, position=0, leave=True):
        print("Processing images on date: " + date)
        if (
            len(filenames) <= 38
        ):  # Filter the days with num_imgs <= 38 to save time and memory
            continue
        merged_image_path = merge_images(filenames)
        exact_shape_dir = (
            "/datacommons/carlsonlab/ld243/images/geometry_info/Delhi.geojson"
        )
        cropped_image_path = crop_image(exact_shape_dir, merged_image_path)
        grid_size = 100
        save_path = "/datacommons/carlsonlab/ld243/Delhi_unlabeled.pkl"
        grid_transform(
            cropped_image_path, grid_size, date, save_path, sensor_loc=sensor_loc
        )


# In[ ]:


main()
