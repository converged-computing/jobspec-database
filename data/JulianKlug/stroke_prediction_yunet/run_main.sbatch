#!/bin/bash
#SBATCH --job-name=run_main_yunet    # Job name
#SBATCH -G volta:2                    # Number of GPUs
#SBATCH --ntasks=3                    # Run on a single CPU
#SBATCH --mem=30G                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=17                     # Job memory request
#SBATCH --output=julian/logs/yunet_log_%j.log   # Standard output and error log

module load anaconda

cd /home/gridsan/gleclerc/julian/yunet
export PYTHONPATH=$(pwd)

source /home/gridsan/gleclerc/.bashrc

conda activate yunet

ulimit -S -n 131072
ulimit -S -u 1546461

srun /home/gridsan/gleclerc/.conda/envs/yunet/bin/python /home/gridsan/gleclerc/julian/yunet/data_prepro/convert_gsd_to_hdf5.py /home/gridsan/gleclerc/julian/data/padded_flipped_rescaled_with_ncct_dataset_with_core_with_penumbra.npz -o /home/gridsan/gleclerc/julian/data/yunet_datasets/padded_flipped_rescaled_with_ncct_with_core_with_penumbra_hdf5_dataset/
srun /home/gridsan/gleclerc/.conda/envs/yunet/bin/python /home/gridsan/gleclerc/julian/yunet/data_prepro/convert_gsd_to_nifti.py /home/gridsan/gleclerc/julian/data/padded_flipped_rescaled_with_ncct_dataset_with_core_with_penumbra.npz -o /home/gridsan/gleclerc/julian/data/yunet_datasets/padded_flipped_rescaled_with_ncct_with_core_with_penumbra_nifti_dataset/

srun /home/gridsan/gleclerc/.conda/envs/yunet/bin/python /home/gridsan/gleclerc/julian/yunet/main.py