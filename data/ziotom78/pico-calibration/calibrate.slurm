#!/bin/bash -l
#SBATCH --account=mp107
#SBATCH -q debug
#SBATCH -N 80
#SBATCH -L cscratch1
#SBATCH --job-name=pico-calibration
#SBATCH -t 19:59
#SBATCH --get-user-env

set -o errexit

nodes=80
nprocs=$(($nodes * 16))

################################################################################
# To use this script, you must pass the name of the INI file via the
# INI_FILE environment variable, e.g.:
#
# $ INI_FILE=$HOME/myfile.ini sbatch calibrate.slurm
#
################################################################################
# WARNING: in order to run this script, you must ensure you have "calibrate.py"
# in your PATH!
################################################################################

if [ "$INI_FILE" == "" ]; then
    echo "You forgot to set the INI_FILE variable!"
    exit 1
fi

calibrate=$(which calibrate.py)

echo "Starting the script on $(date), directory is $(pwd)"

module swap PrgEnv-intel PrgEnv-gnu
module unload altd
module unload darshan

export PYTHONPATH=
module load python/3.6-anaconda-4.4

echo "Python executable: $(which python)"
echo "Python version: $(python --version &> /dev/stdout)"
echo "Calibrate script: ${calibrate}"
echo "----------------------------------------"
env | sort -d
echo "----------------------------------------"

logfile=./log/$(basename $INI_FILE .ini).log
echo "Going to run srun on $(date) with $nodes nodes"
srun -n $nprocs python ${calibrate} $INI_FILE 2>&1 | tee $logfile

echo "Script ended on $(date)"
