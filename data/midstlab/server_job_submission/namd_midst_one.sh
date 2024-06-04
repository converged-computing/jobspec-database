#!/bin/bash
#
# CompecTA (c) 2018
#
# NAMD job submission script
#
# TODO:
#   - Set name of the job below changing "NAMD" value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#     - cuda  : For CUDA jobs. Solver that can utilize CUDA acceleration can use this queue. 15 days limit.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch namd_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=1pga_WT
#SBATCH --account=midst
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --qos=midst
#SBATCH --partition=midst
#SBATCH --time=7-00:00:00
#SBATCH --output=%j-log.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tguclu@sabanciuniv.edu

INPUT_FILE="1pga_autopsf_wb_ionized_config.conf"
#INPUT_FILE="PATZ1-SMRT.conf"

################################################################################
source /etc/profile.d/modules.sh
echo "source /etc/profile.d/modules.sh"
################################################################################

# Module File
echo "Loading NAMD..."
module load namd/2.13/multicore
echo

echo
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo

echo
echo "=================================== STACK SIZE ====================================="
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo "===================================================================================="
echo

COMMAND="namd2 $INPUT_FILE +p$SLURM_NTASKS"

# Running Solver.
echo "Running NAMD command..."
echo $COMMAND
echo "===================================================================================="
NAMD_OUTPUT="namd.out-$SLURM_JOB_ID"
echo "Redirecting output to file: $NAMD_OUTPUT"
echo "-------------------------------------------"
$COMMAND > $NAMD_OUTPUT 2>&1
RET=$?

echo
echo "Solver exited with return code: $RET"
exit $RET
