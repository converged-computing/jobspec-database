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
import pandas as pd
import numpy as np
import requests as rq
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tqdm import tqdm, trange
# from kernellib.types import GeoData, Dimensions
# from kernellib.preprocessing import create_oi_grid, correct_lon, add_vtime
# from kernellib.data import load_data

from kernellib.data.files import files_factory
from kernellib.data.io import load_multiple_data
from kernellib.data.l3 import read_l3_data
from kernellib.features.aoi import aoi_factory, subset_data
from kernellib.features.oi import oi_params_factory, create_oi_grid, reformat_oi_output
from kernellib.features.space import correct_lon
from kernellib.features.time import add_vtime
from kernellib.features.interp import interp_on_alongtrack

# TESTING PURPOSES
smoke_test = False
SAVE_PATH = "/bettik/johnsonj/data/data_challenges/ssh_mapping_2021/interim"
RAW_DATA_PATH = "/bettik/johnsonj/data/data_challenges/ssh_mapping_2021/raw/netcdf"

print("Starting Script...")

# PARAMS
aoi = "dc2021sm"
oi = "default"
system = "cal1"

aoi_params = aoi_factory(aoi)
oi_params = oi_params_factory(oi)
file_params = files_factory(system)

# CREATE OI GRID
n_samples = 100

ds_oi_grid = create_oi_grid(aoi_params, n_samples)


# LOAD OBSERVATIONS
print("Loading Observations...")
inputs = load_multiple_data(file_params.train_data_dir)

# CORRECT OBSERVATIONS
print("Correcting Observations...")
ds_obs = subset_data(ds_obs, aoi_params, oi_params)

# TRANSFORM COORDINATES
ds_obs = add_vtime(ds_obs, aoi_params.time_min)






n_batches_pred = 100
print("Starting Loop")


n_time_steps = len(ds_oi_grid.gtime)

for i_time in (pbar := trange(len(ds_oi_grid.gtime))):
    pbar.set_description_str(f"time: {i_time}")

    # get indices where there are observations
    pbar.set_description("Subsetting Data...")

    ind1 = np.where(
        (np.abs(ds_obs.time.values - ds_oi_grid.gtime.values[i_time]) < 2.0 * oi_params.Lt)
    )[0]

    ind_t = np.where(
        (np.abs(ds_obs.time.values - ds_oi_grid.gtime.values[i_time]) < 1.0)
    )[0]
    n_obs = len(ind1)

    # get observation data
    obs_data = GeoData(
        lat=ds_obs.latitude.values[ind1],
        lon=ds_obs.longitude.values[ind1],
        time=ds_obs.time.values[ind1],
        data=ds_obs.sla_unfiltered.values[ind1],
    )

    # get state data
    state_data = Dimensions(
        lat=ds_oi_grid.fglat.values,
        lon=ds_oi_grid.fglon.values,
        time=ds_oi_grid.gtime.values[i_time],
    )

    # DATA KERNEL

    state_coords = state_data.coord_vector()
    obs_coords = obs_data.coord_vector()

    # ML MODEL
    if smoke_test:
        train_x = torch.Tensor(obs_coords)[:1000]
        train_y = torch.Tensor(obs_data.data)[:1000]
    else:
        train_x = torch.Tensor(obs_coords)
        train_y = torch.Tensor(obs_data.data)
    test_x = torch.Tensor(state_coords)#[:1000]
    


    pbar.set_description("Fitting GP Model...")
    gp_params = gp_model_factory(model="exact")
    gp_params.length_scale = [oi_params.Lt, oi_params.Lx, oi_params.Ly]
    model, likelihood = gp_params.init_gp_model(train_x, train_y)

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
    if smoke_test:
        break


print("Done!")

from pathlib import Path


if smoke_test:
    SAVE_PATH = Path(SAVE_PATH).joinpath("OSE_ssh_mapping_BASELINE_test.nc")
else:
    SAVE_PATH = Path(SAVE_PATH).joinpath("OSE_ssh_mapping_BASELINE.nc")

# ML MODEL
ds_oi_grid.to_netcdf(SAVE_PATH)

# mean and variance predictions

# Load Results
print("opening oi results...")
ds_oi = xr.open_dataset(
    Path(config.results_dir).joinpath(config.results_filename)
)

# reformat oi results
print("reformatting oi results...")
ref_ds = xr.open_dataset(
    Path(config.ref_data_dir).joinpath(config.ref_data_filename)
)
ds_oi = reformat_oi_output(ds_oi, ref_ds)

# load test data
print("opening test dataset...")
ds_along_track = read_l3_data(
    Path(config.test_data_dir).joinpath(config.test_data_filename), 
    config.aoi
)

# interpolation
print("interpolating along track...")
time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_map_interp = interp_on_alongtrack(
    ds_oi, 
    ds_along_track,
    aoi=config.aoi,
    is_circle=True
)

# Compute Error Statistics
leaderboard_nrmse, leaderboard_nrmse_std = compute_stats(
    time_alongtrack, 
    lat_alongtrack, 
    lon_alongtrack, 
    ssh_alongtrack, 
    ssh_map_interp, 
    rmse_binning.bin_lon_step,
    rmse_binning.bin_lat_step, 
    rmse_binning.bin_time_step,
    Path(config.results_dir).joinpath(f"stats_{config.results_filename}"),
    Path(config.results_dir).joinpath(f"stats_ts_{config.results_filename}"),
)

plot_spatial_statistics(Path(config.results_dir).joinpath(f"stats_{config.results_filename}"))


plot_temporal_statistics(
    Path(config.results_dir).joinpath(f"stats_ts_{config.results_filename}")
)

leaderboard_psds_score = plot_psd_score(Path(config.results_dir).joinpath(f"psd_{config.results_filename}"))



data = [['BASELINE', 
         leaderboard_nrmse, 
         leaderboard_nrmse_std, 
         int(leaderboard_psds_score),
        'Covariances TRAIN OPT OI',
        'example_eval_baseline.ipynb']]
Leaderboard = pd.DataFrame(data, 
                           columns=['Method', 
                                    "µ(RMSE) ", 
                                    "σ(RMSE)", 
                                    'λx (km)',  
                                    'Notes',
                                    'Reference'])
print("Summary of the leaderboard metrics:")
print(Leaderboard.to_markdown())