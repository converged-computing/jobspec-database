#!/bin/sh

## reporting
#SBATCH --error=logs/%A_%a.err
#SBATCH --output=logs/%A_%a.out
# change the following lines to have one pound sign instead of two
# if you want to receive notices about jobs
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
##SBATCH --mail-user=8083489586@vtext.com

#SBATCH --mem-per-cpu=6400

#SBATCH --job-name=demand_scenarios
#SBATCH --partition=community.q
#SBATCH --time=3-00:00:00
# specify the number of parallel tasks to run in the commmand below
#SBATCH --array=1-40

# find the name of the scenario to run for this job
# see https://www.unix.com/shell-programming-and-scripting/214381-get-nth-word-mth-line-file.html
scen_name=$(awk -v line="${SLURM_ARRAY_TASK_ID}" -v field="2" 'NR==line{print $field}' scenarios_2018_04_29_theta01.txt)
echo "Solving scenario $scen_name."

# Note: we use --tempdir tmp to ensure that temporary files are created in the
# file system, not in the system temporary dir, because on the UH HPC system,
# the system temporary dir is a ramdisk and storing intermediate files there can
# cause an out-of-memory error while solving.

module load lang/Python/2.7.10/python
srun --unbuffered --time=3-00:00:00 --partition=community.q switch solve-scenarios --scenario $scen_name --scenario-queue sq/sq_$SLURM_ARRAY_JOB_ID "$@" --tempdir ./tmp

