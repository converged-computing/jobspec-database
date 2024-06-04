#!/bin/bash -l

#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --account=desi
#SBATCH --image=driftingpig/obiwan_composit:v3
#SBATCH -J obiwan
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --mail-user=kong.291@osu.edu
#SBATCH --mail-type=ALL


source /global/cscratch1/sd/huikong/obiwan_Aug/repos_for_docker/bashrc_obiwan

# USE PY3, py2 doesn't work with draw_points_eboss
export CSCRATCH_OBIWAN=/global/cscratch1/sd/huikong/obiwan_Aug/repos_for_docker


export PYTHONPATH=$CSCRATCH_OBIWAN/obiwan_code/py:$CSCRATCH_OBIWAN/legacypipe/py:$PYTHONPATH
# We need 3 directories for Obiwan
export obiwan_data=$CSCRATCH_OBIWAN/obiwan_data
export obiwan_code=$CSCRATCH_OBIWAN/obiwan_code
export obiwan_out=$CSCRATCH_OBIWAN/obiwan_out

#source $CSCRATCH_OBIWAN/obiwan_code/bin/run_atnersc/bashrc_desiconda
export outdir=$CSCRATCH_OBIWAN/obiwan_out/eboss_elg/randoms_test_2
export survey=eboss

export startid=77060581 #1618001
export max_prev_seed=65
# SGC
# SGC A
export nrandoms=66248860 #418000
export ra1=316.5
export ra2=360.
export dec1=-2.
export dec2=2.
# SGC B
#export nrandoms=66248860
#export ra1=0.
#export ra2=45.
#export dec1=-5.
#export dec2=5.
# NGC
#export nrandoms=1200000
#export ra1=126.
#export ra2=165.
#export dec1=14.
#export dec2=29.


# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Protect against astropy configs
export XDG_CONFIG_HOME=/dev/shm
srun -n $SLURM_JOB_NUM_NODES mkdir -p $XDG_CONFIG_HOME/astropy

#let tasks=32*$SLURM_JOB_NUM_NODES
export tasks=16
srun -n ${tasks} -c 2 shifter ./docker_job_randoms_init.sh

