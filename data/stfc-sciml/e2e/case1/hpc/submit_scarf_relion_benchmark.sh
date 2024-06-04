#!/bin/bash
#SBATCH -p scarf 
#SBATCH -C scarf17
#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --time 4-0

# Load modules
module load OpenMPI/4.1.0-iccifort-2018.3.222-GCC-7.3.0-2.30

# Set environment variables for this machine

#Location of the case1 folder
export BASE_DIR="/home/vol08/scarf688/git/intel-e2e-benchmark/case1"
# Location of the relion singularity image
export RELION_IMG="$BASE_DIR/relion.sif"
# Relion project directory data
export RELION_PROJ_DIR="/work3/projects/sciml/scarf688/relion/relion_benchmark"
# Location to store output files
export RELION_OUTPUT_DIR="/work3/projects/sciml/scarf688/relion/runs/scarf/job_$SLURM_JOB_ID"
# Relion command
export RELION_CMD="singularity run -B $RELION_OUTPUT_DIR -B $BASE_DIR -H $RELION_PROJ_DIR $RELION_IMG -gpu_disable_check"
# Number of cpus to use with -j option
export RELION_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
# Additional optimization flags
export RELION_OPT_FLAGS='--dont_combine_weights_via_disc --pool 30'
# Additional MPI flags
export RELION_MPI_FLAGS='--mca opal_warn_on_missing_libcuda 0'

# Run pipeline
./benchmark_scripts/benchmark_relion.py ./benchmark_scripts/plasmodium_ribosome_benchmarks.sh 
