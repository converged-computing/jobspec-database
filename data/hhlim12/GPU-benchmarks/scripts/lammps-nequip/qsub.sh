#!/bin/sh
#PBS -q i1accs
#PBS -l select=1:ncpus=64:mpiprocs=1:ompthreads=1
#PBS -l walltime=00:15:00
#PBS -N GCNLMP
##PBS -I

module purge
module load openmpi_nvhpc/4.1.2
module load nvhpc-nompi/22.2_cuda11.6

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nequip_GPU2

export CUDA_VISIBLE_DEVICES="0"

./lmp -in input.data
