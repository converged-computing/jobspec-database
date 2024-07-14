import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import six.moves.cPickle as pickle
from tqdm import trange

from latline.common.plot3d import plot3d
from latline.experiment_config import DataConfig
from latline.latline import Latline
import argparse


def generate_data(mode='train'):
    """
    This function generates the simulated sensor data given the following parameters:
    """
    N_examples = DataConfig.N_train if mode == 'train' else DataConfig.N_test

    # pre-alloc data and labels
    data    = np.zeros((N_examples, 2, DataConfig.n_sensors, DataConfig.tau))
    targets  = np.zeros((N_examples, 2, DataConfig.n_sensors, DataConfig.resolution))

    # s should be interpreted as in Boulogne et al (2015)
    s, target_mesh, x_mesh2d, x_mesh3d, y_mesh3d, z_mesh3d = get_meshes()

    # Init lateral line for train data
    latline = Latline(DataConfig)

    if DataConfig.display:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # We initialize buffers for the two sensor arrays that contain the excitations over multiple timesteps
    buffer0 = deque()
    buffer1 = deque()

    # Loop through all data points
    t = trange(N_examples + DataConfig.tau)
    t.set_description("Generating {} data".format(mode))
    for i in t:
        # Obtain x and y locations of all spheres and the corresponding sensor measurements
        xs, ys, zs, fluid_v0, fluid_v1 = latline.step()  # xs' rows correspond to the spheres

        # Compute each Gaussian separately
        target0 = np.asarray([np.exp(-((x_mesh2d - xi) ** 2 + (target_mesh - np.sqrt((yi+0.5)**2 + zi**2)) ** 2)
                                     / (2 * DataConfig.sigma**2))
                              for (xi, yi, zi) in zip(xs, ys, zs)])
        target1 = np.asarray([np.exp(-((x_mesh2d - xi) ** 2 + (target_mesh - np.sqrt((yi-0.5)**2 + zi**2)) ** 2)
                                     / (2 * DataConfig.sigma**2))
                              for (xi, yi, zi) in zip(xs, ys, zs)])
        # And then take the maximum over all spheres.
        target0 = np.max(target0, axis=0)
        target1 = np.max(target1, axis=0)

        # Save measurements and heatmap
        buffer0.append(fluid_v0 * DataConfig.sensitivity)
        buffer1.append(fluid_v1 * DataConfig.sensitivity)

        if i >= DataConfig.tau:
            buffer0.popleft()
            buffer1.popleft()

            data[i - DataConfig.tau][0] = np.transpose(buffer0)
            data[i - DataConfig.tau][1] = np.transpose(buffer1)
            targets[i - DataConfig.tau][0] = np.transpose(target0.reshape((DataConfig.resolution, DataConfig.n_sensors)))
            targets[i - DataConfig.tau][1] = np.transpose(target1.reshape((DataConfig.resolution, DataConfig.n_sensors)))

        if DataConfig.display and i < 100:
            # Plot the situation in 3D
            plot3d(DataConfig, ax, fluid_v0, fluid_v1, xs, ys, zs, x_mesh3d, y_mesh3d, z_mesh3d, latline, s, target0, target1)

    return data, targets


def get_meshes():
    s = np.linspace(DataConfig.sensor_range[0], DataConfig.sensor_range[1], DataConfig.n_sensors)
    x_grid = s
    y_grid = s
    z_grid = np.linspace(DataConfig.z_range[0], DataConfig.z_range[1], DataConfig.resolution)
    # xx and yy are matrices for evaluating a Gaussian that is centered at a sphere
    # xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
    x_mesh2d, target_mesh = np.meshgrid(x_grid, z_grid)
    x_mesh3d, y_mesh3d, z_mesh3d = np.meshgrid(x_grid, y_grid, z_grid)
    return s, target_mesh, x_mesh2d, x_mesh3d, y_mesh3d, z_mesh3d


if __name__ == "__main__":
    """
    This script generates the data to be used for the training of the CNNs in train.py
    """
    DataConfig.load()

    project_folder = os.path.dirname(os.path.realpath(__file__))
    fnm = os.path.join(project_folder, 'data', '{}.pickle'.format(DataConfig.fnm))
    if not DataConfig.force and os.path.exists(fnm):
        print("Already exists")
        exit(0)

    # Obtain data for train, test and validation sets
    train_data, train_labels = generate_data(mode='train')
    test_data, test_labels = generate_data(mode='test')

    # Serialize the data
    os.makedirs(os.path.join(project_folder, 'data'), exist_ok=True)

    with open(os.path.join(project_folder, 'data', '{}.pickle'.format(DataConfig.fnm)), 'wb') as f:
        pickle.dump([train_data, train_labels, test_data, test_labels], f)
