#!/bin/bash
#SBATCH -p scarf 
#SBATCH -C scarf18
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time 4-0

# Set environment variables for this machine
module load intel/2019b

#Location of the case1 folder
export BASE_DIR="/home/vol08/scarf688/git/intel-e2e-benchmark/case1"
# Location of the relion singularity image
export RELION_IMG="$BASE_DIR/relion.sif"
# Relion project directory data
export RELION_PROJ_DIR="/work3/projects/sciml/scarf688/relion/10338"
# Location to store output files
export RELION_OUTPUT_DIR="/work3/projects/sciml/scarf688/relion/runs/scarf/job_$SLURM_JOB_ID"
# Relion command
export PATH="$BASE_DIR/relion/build/bin:$PATH"
export RELION_CMD=""
# Number of cpus to use
export RELION_NUM_CPUS=${SLURM_NTASKS:-23}
# Number of cpu threads to use with -j option
export RELION_CPU_THREADS_PER_TASK=4
# Additional optimization flags
export RELION_OPT_FLAGS="--dont_combine_weights_via_disc --cpu --pool $RELION_NUM_CPUS --j $RELION_CPU_THREADS_PER_TASK"
# Additional MPI flags
export RELION_MPI_FLAGS=''


# Run pipeline
./benchmark_scripts/benchmark_relion.py $RELION_SCRIPT_NAME
