#!/bin/bash -l

#SBATCH -p regular
#SBATCH -N 7
#SBATCH -t 18:00:00
#SBATCH --account=desi
#SBATCH --image=driftingpig/obiwan_composit:v3
#SBATCH -J obiwan
#SBATCH -o ./slurm_output/elg_like_%j.out
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --mail-user=kong.291@osu.edu  
#SBATCH --mail-type=ALL

export name_for_run=elg_like_run
export randoms_db=None #run from a fits file
export dataset=dr3
export rowstart=0
export do_skipids=no
export do_more=yes
export minid=1
export object=elg
export nobj=100

export usecores=32
export threads=$usecores
#threads=1
export CSCRATCH_OBIWAN=$CSCRATCH/obiwan_Aug/repos_for_docker
#obiwan paths
export obiwan_data=$CSCRATCH_OBIWAN/obiwan_data 
export obiwan_code=$CSCRATCH_OBIWAN/obiwan_code 
export obiwan_out=$CSCRATCH_OBIWAN/obiwan_out   

# Load production env
#source $CSCRATCH/obiwan_code/obiwan/bin/run_atnersc/bashrc_obiwan
export LEGACY_SURVEY_DIR=$obiwan_data/legacysurveydir_dr3_origin

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Protect against astropy configs
export XDG_CONFIG_HOME=/dev/shm
srun -n $SLURM_JOB_NUM_NODES mkdir -p $XDG_CONFIG_HOME/astropy

srun -N 7 -n 14 -c $usecores shifter ./example1.sh
wait
