#!/bin/bash
#
#SBATCH --job-name=Train_Py
#SBATCH --output=log.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=skylake
#SBATCH --gres=gpu:1
#SBATCH --reservation=ml

# Load the modules
module load python
module load numpy/1.14.1-python-2.7.14
module load tensorflowgpu/1.6.0-python-2.7.14
module load scikit-learn/0.19.1-python-2.7.14
module load keras/2.1.4-python-2.7.14
module load h5py/2.7.1-python-2.7.14-serial

# Run the script
python train.py