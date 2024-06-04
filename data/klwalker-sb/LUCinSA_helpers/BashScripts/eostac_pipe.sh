#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 2 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o stacpipe_crg.%N.%a.%j.out # STDOUT
#SBATCH -e stacpipe_crg.%N.%a.%j.err # STDERR
#SBATCH --job-name="stpipe_crg"
#SBATCH --array=283
#32,33,36,37,38,39

#GRIDS="${SLURM_ARRAY_TASK_ID}"
GRIDS="$(($SLURM_ARRAY_TASK_ID + 3000))"

#############################################
# Turn off NumPy parallelism and rely on dask
#############################################
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# This should be sufficient for OpenBlas and MKL
export OMP_NUM_THREADS=1
################################################

#STEP="preprocess"
STEP="preprocess"
SAT_SENSORS=S2,S2cp,LT05,LE07,LC08,LC09
#SAT_SENSORS=S2,S2cp,LC08
NCHUNKS=512
RERUN="True"

###############################
# DO NOT MODIFY BELOW THIS LINE
###############################

# activate the virtual environment
# conda activate venv.lucinsa38_pipe
#conda activate venv.lucinsa38_pipe
conda activate .lucinla38_pipe

CONFIG_UPDATES="grids:[${GRIDS}] res:${REF_RES} crs:${REF_CRS} 
cloud_mask:sat_sensors:${SAT_SENSORS}
cloud_mask:reset_db:${RERUN}
main_path:/home/sandbox-cel/paraguay_lc/stac/grid
backup_path:/home/downspout-cel/paraguay_lc/stac/grids
num_workers:${SLURM_CPUS_ON_NODE} 
io:n_chunks:${NCHUNKS} cloud_mask:reset_db:${RESET_CLOUD_DB} 
cloud_mask:ref_res:${REF_RES}"

tuyau $STEP --config-updates $CONFIG_UPDATES

conda source deactivate
