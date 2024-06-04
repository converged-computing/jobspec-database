#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1

# Load modules
module load OpenMPI/4.1.0-GCC-9.3.0

# Set environment variables for this machine

#Location of the case1 folder
export BASE_DIR="/mnt/beegfs/work/stfc/pearl008/intel-e2e-benchmark/case1"
# Location of the relion singularity image
export RELION_IMG="$BASE_DIR/relion.sif"
# Relion project directory data
export RELION_PROJ_DIR="$BASE_DIR/data/10338"
# Location to store output files
export RELION_OUTPUT_DIR="$BASE_DIR/runs/pearl/job_$SLURM_JOB_ID"
# Relion command
export RELION_CMD="singularity run --nv -B $BASE_DIR -H $RELION_PROJ_DIR $RELION_IMG -gpu_check_disable"
# Number of cpus to use with -j option
export RELION_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
# Additional optimization flags
export RELION_OPT_FLAGS='--gpu --dont_combine_weights_via_disc --pool 30'

# Run pipeline
./benchmark_scripts/benchmark_relion.py ./benchmark_scripts/10338/pipeline_class3d_1.sh
