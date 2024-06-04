#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J LMDZ_RUN
#SBATCH --mail-type=ALL,TIME_LIMIT_50
#SBATCH --mail-user=jessed@berkeley.edu
#SBATCH -o lmdzrun-%j.out
#SBATCH -e lmdzrun-%j.err

module swap PrgEnv-intel PrgEnv-gnu
cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
srun -n 24 ./gcm.e
