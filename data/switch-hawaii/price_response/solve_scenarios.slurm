#!/bin/sh

## reporting
#SBATCH --open-mode=append
#SBATCH --error=logs/%A_%a.err
#SBATCH --output=logs/%A_%a.out

# change the following lines to have one pound sign instead of two
# if you want to receive notices about jobs
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
##SBATCH --mail-user=8083489586@vtext.com

#SBATCH --mem-per-cpu=6400

#SBATCH --job-name=demand_scenarios
#SBATCH --partition=shared
#SBATCH --time=3-00:00:00
# specify the number of parallel tasks to run in the commmand below
#SBATCH --array=1-40

module load lang/Anaconda3
# conda activate requires `conda init bash` or similar, which may not work here
source activate demand_system
# previously created with
# conda create --name demand_system -c conda-forge python=3.7.3 switch_model=2.0.6 rpy2=3.1.0 scipy=1.3.1 pyomo=5.6.6 pyutilib=5.7.1
# conda activate demand_system
# pip install --editable switch/

# Note: we use --tempdir tmp to ensure that temporary files are created in the
# file system, not in the system temporary dir, because on the UH HPC system,
# the system temporary dir is a ramdisk and storing intermediate files there can
# cause an out-of-memory error while solving.
echo ============================================================
echo running job: switch solve-scenarios --scenario-queue sq/$SLURM_ARRAY_JOB_ID --job-id "$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID" "$@" --tempdir ./tmp
echo ============================================================
echo
srun --unbuffered switch solve-scenarios --scenario-queue sq/$SLURM_ARRAY_JOB_ID --job-id "$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID" "$@" --tempdir ./tmp

