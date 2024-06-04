#!/bin/bash

# DESCRIPTION: This bash script is to execute driver_decomp.m which constructs
#              feature vectors from tensors using either the tensor decomposition
#              algorithm, which has parameters that can be set, or by vectorizing
#              (unfolding) the tensor. The parameters for the tensor decomposition
#              should be set in the load_<data_set>_tensors.m file in the utils/
#              directory. This script is designed to run as an array job, where
#              each index corresponds to the number of windows in the tensor
#              being used.
#
#              When driver_decomp.m runs to completion, compile_feature-vectors.m
#              is used to automatically compile the results from all jobs in
#              the job array.
#
# TODO:        Before running the script, there are 3 fields that should
#              change with every job submission:
#              1. job name
#              2. output file
#              3. parameters passed to driver_decomp
#              4. compile_feature_vectors

#SBATCH --job-name=Tensor-Decomp
#SBATCH --mail-user=jpic@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=50G
#SBATCH --time=4:00:00
#SBATCH --account=kayvan1
#SBATCH --partition=standard
#SBATCH --output=/nfs/turbo/med-kayvan-lab/Projects/DoD_Multimodal/Code/Joshua/datasets/EMG/epsilon_bounds/upper_bounds/1/logspaced/%x-%j.log
#SBATCH --array=2-10

matlab -nodisplay -r "driver_decomp(string('EMG'), string('epsilon_bounds/upper_bounds/1/logspaced'), $SLURM_ARRAY_TASK_ID); compile_feature_vectors(string('EMG'), string('epsilon_bounds/upper_bounds/1/logspaced')); exit"

