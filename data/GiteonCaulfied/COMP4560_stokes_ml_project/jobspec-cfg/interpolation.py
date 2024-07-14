import h5py
import os

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial import cKDTree

# Folder Path
path = "/scratch/kr97/xh7958/comp4560/solutions"
  
class HDF5Interpolator:
    def __init__(self, filename):
        # Load the HDF5 file
        with h5py.File(filename, 'r') as file:
            self.timestamps = file['timestamps'][:]  # Assuming the timestamps are stored as a dataset
            self.arrays = file['temperature'][:]  # Replace with actual path to your arrays

        # Build the KDTree
        self.kdtree = cKDTree(self.timestamps.reshape(-1, 1))  # Reshape to meet KDTree input requirement

    def interpolate(self, time):
        # Find the two nearest timestamps
        distances, indices = self.kdtree.query(np.array([[time]]), k=2)
        t1, t2 = self.timestamps[indices[0]]
        
        '''
        # Ensure the time is within the bounds of the timestamps
        if t1 > time or t2 < time:
            raise ValueError(f"Time {time} is out of bounds of the data timestamps.")
        '''

        # Get the corresponding arrays
        array1, array2 = self.arrays[indices[0][0]], self.arrays[indices[0][1]]

        # Interpolate between the two arrays
        alpha = (time - t1) / (t2 - t1)
        interpolated_array = array1 * (1 - alpha) + array2 * alpha

        return interpolated_array
    
standard_timestamps = np.linspace(2e-6, 1e-4, 100)
new_temperature_fields = []

for file in os.listdir(path):
    new_temperature_fields_file = []
    
    file_path = f"{path}/{file}"
    interpolator = HDF5Interpolator(file_path)

    for i in range(len(standard_timestamps)):
        new_temperature_fields_file.append(interpolator.interpolate(standard_timestamps[i]))
        print(str(i),"of temperature fields has been generated")

    new_temperature_fields.append(np.asarray(new_temperature_fields_file))
    
new_temperature_fields = np.asarray(new_temperature_fields)
with open('/scratch/kr97/xh7958/comp4560/solutions_standard.npy', 'wb') as f:
    np.save(f, new_temperature_fields)


