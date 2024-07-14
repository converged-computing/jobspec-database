"""
Baseline Experiment

Experimental Parameters
-----------------------
* Exact GP
* RBF Kernel + ARD
* Exact Inputs (no transformations)
* Resolutions (0.2, 0.05)

Experimental Steps
------------------
1. Create the OI Grid of Interest
2. Load the Observations
3. Transform Observations to confine with Grid
4. 
"""
import sys, os
from pyprojroot import here

# spyder up to find the local root
local = here(project_files=[".local"])
sys.path.append(str(local))

import xarray as xr
import numpy as np
import requests as rq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tqdm import tqdm, trange
from kernellib.types import GeoData, Dimensions
from kernellib.preprocessing import create_oi_grid, correct_lon, add_vtime
from kernellib.data import load_data

print("Starting Script")
import torch

print("Cuda:", torch.cuda.is_available())

# TESTING PURPOSES
smoke_test = False
SAVE_PATH = "/bettik/johnsonj/data/data_challenges/ssh_mapping_2021/interim"
RAW_DATA_PATH = "/bettik/johnsonj/data/data_challenges/ssh_mapping_2021/raw/netcdf"


# CREATE OI GRID
# OI Grid
lon_min = 295.0  # domain min longitude
lon_max = 305.0  # domain max longitude
lat_min = 33.0  # domain min latitude
lat_max = 43.0  # domain max latitude
time_min = np.datetime64("2017-01-01")  # domain min time

if smoke_test:
    time_max = np.datetime64("2017-01-31")  #
else:
    time_max = np.datetime64("2017-12-31")  # domain max time
dx = 0.05  # zonal grid spatial step (in degree)
dy = 0.05  # meridional grid spatial step (in degree)
dt = np.timedelta64(1, "D")  # temporal grid step

if smoke_test:
    n_samples = 10
else:
    n_samples = 100
glon = np.arange(lon_min, lon_max + dx, dx)  # output OI longitude grid
glat = np.arange(lat_min, lat_max + dy, dy)  # output OI latitude grid
gtime = np.arange(time_min, time_max + dt, dt)  # output OI time grid

# create OI grid
ds_oi_grid = create_oi_grid(glon, glat, gtime, n_samples)


# KERNEL PARAMETERS
Lx = 1.0  # Zonal decorrelation scale (in degree)
Ly = 1.0  # Meridional decorrelation scale (in degree)
Lt = 7.0  # Temporal decorrelation scale (in days)
noise = 0.05


# LOAD OBSERVATIONS


inputs = load_data(RAW_DATA_PATH)

if smoke_test:
    inputs = [inputs[0]]

# Coarsening resolutions
coarsening = {"time": 5}


def preprocess(ds):
    return ds.coarsen(coarsening, boundary="trim").mean()


ds_obs = xr.open_mfdataset(
    inputs, combine="nested", concat_dim="time", parallel=True, preprocess=preprocess
).sortby("time")
# ds_obs = ds_obs.coarsen(coarsening, boundary="trim").mean().sortby('time')

# correct longitude values


# Subset time
ds_obs = ds_obs.sel(
    time=slice(
        time_min - np.timedelta64(int(2 * Lt), "D"),
        time_max + np.timedelta64(int(2 * Lt), "D"),
    ),
    drop=True,
)

# subset lat/lon values

ds_obs = correct_lon(ds_obs, lon_min)

ds_obs = ds_obs.where(
    (ds_obs["longitude"] >= lon_min - Lx)
    & (ds_obs["longitude"] <= lon_max + Lx)
    & (ds_obs["latitude"] >= lat_min - Ly)
    & (ds_obs["latitude"] <= lat_max + Ly),
    drop=True,
)

# add a vectorized time
ds_obs = add_vtime(ds_obs, time_min)

# drop all nans
ds_obs = ds_obs.dropna(dim="time")

import torch
import gpytorch
from kernellib.models import get_exact_gp, get_rff_gp, get_sparse_gp
from kernellib.model_utils import gp_batch_predict, gp_samples, fit_gp_model

n_batches_pred = 100
print("Starting Loop")


obs_data = GeoData(
    lat=ds_obs.latitude.values,
    lon=ds_obs.longitude.values,
    time=ds_obs.time.values,
    data=ds_obs.sla_unfiltered.values,
)

obs_coords = obs_data.coord_vector()

# ML MODEL

train_x = torch.Tensor(obs_coords)
train_y = torch.Tensor(obs_data.data)

# subset data
n_subset = 25_000
if smoke_test and n_subset > train_x.shape[0]:
    print(f"Subsetting the data (smoke test): {train_x.shape[0]} -> {n_subset}")
    seed = 123
    rng = np.random.RandomState(seed)
    ind = rng.choice(np.arange(train_x.shape[0]), size=(n_subset,))
else:
    print(f"Subsetting the data (smoke test): {train_x.shape[0]} -> {n_subset}")
    seed = 123
    rng = np.random.RandomState(seed)
    ind = rng.choice(np.arange(train_x.shape[0]), size=(n_subset,))

train_x = train_x[ind]
train_y = train_y[ind]
assert train_x.shape[0] == n_subset
assert train_y.shape[0] == n_subset

# fit gp model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
print("Fitting GP Model...")
model = get_exact_gp()(train_x, train_y, likelihood)

# set the kernel parameters
model.covar_module.base_kernel.lengthscale = [Lt, Lx, Ly]
model.likelihood.noise = 0.05

print("Training GP Model...")

if smoke_test:
    n_iterations = 10
else:
    n_iterations = 5_000

losses, model, likelihood = fit_gp_model(
    train_x, train_y, model, likelihood, n_iterations=n_iterations
)


for i_time in (pbar := trange(len(gtime))):
    pbar.set_description_str(f"time: {i_time}")

    # get indices where there are observations
    pbar.set_description("Subsetting Data...")

    ind1 = np.where(
        (np.abs(ds_obs.time.values - ds_oi_grid.gtime.values[i_time]) < 2.0 * Lt)
    )[0]

    ind_t = np.where(
        (np.abs(ds_obs.time.values - ds_oi_grid.gtime.values[i_time]) < 1.0)
    )[0]
    n_obs = len(ind1)

    # get state data
    state_data = Dimensions(
        lat=ds_oi_grid.fglat.values,
        lon=ds_oi_grid.fglon.values,
        time=ds_oi_grid.gtime.values[i_time],
    )

    state_coords = state_data.coord_vector()

    test_x = torch.Tensor(state_coords)

    pbar.set_description("Predictions...")
    y_mu, y_var = gp_batch_predict(model, likelihood, test_x, n_batches_pred)

    # samples
    pbar.set_description("Drawing Samples...")
    y_samples = gp_samples(model, likelihood, test_x, n_samples)

    # save into data arrays
    pbar.set_description("Putting in Data...")
    ds_oi_grid.gssh_mu[i_time, :, :] = y_mu.reshape(
        ds_oi_grid.lat.size, ds_oi_grid.lon.size
    )
    ds_oi_grid.gssh_var[i_time, :, :] = y_var.reshape(
        ds_oi_grid.lat.size, ds_oi_grid.lon.size
    )
    ds_oi_grid.gssh_samples[:, i_time, :, :] = y_samples.reshape(
        n_samples, ds_oi_grid.lat.size, ds_oi_grid.lon.size
    )
    ds_oi_grid.nobs[i_time] = n_obs

    # if smoke_test:
    #     break

print("Done!")

from pathlib import Path


if smoke_test:
    SAVE_PATH = Path(SAVE_PATH).joinpath("OSE_ssh_mapping_BASELINE_TRAIN_test.nc")
else:
    SAVE_PATH = Path(SAVE_PATH).joinpath("OSE_ssh_mapping_BASELINE_TRAIN.nc")

# ML MODEL
ds_oi_grid.to_netcdf(SAVE_PATH)

# mean and variance predictions
