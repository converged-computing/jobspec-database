#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 2
#SBATCH -t 16:00:00
#SBATCH -J my_job

module load python
module load mpi4py
module load boost/1.55
module load szip/2.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/syscom/nsg/lib:/global/homes/c/cslage/Software/hdf5-1.8.14/lib

srun -n 64 python-mpi Run_BF_Multi.py 
