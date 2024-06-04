#!/bin/bash
#SBATCH --account=mpccc
#SBATCH --image=docker:registry.services.nersc.gov/pynamic:2.6a1
#SBATCH --job-name=cori-pynamic-150-shifter
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rcthomas@lbl.gov
#SBATCH --nodes=150
#SBATCH --ntasks-per-node=32
#SBATCH --output=logs/slurm-cori-pynamic-150-shifter-%j.out
#SBATCH --partition=regular
#SBATCH --qos=normal
#SBATCH --time=25

# Configuration.

commit=true
debug=false

# Optional debug output.

if [ $debug = true ]; then
    module list
    set -x
fi

# Unset variables.

unset PYTHONSTARTUP
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset CRAY_LD_LIBRARY_PATH
unset LIBRARY_PATH

# Allows up to 5 minutes for pynamic-pyMPI to MPI_Init().

export PMI_MMAP_SYNC_WAIT_TIME=300

# Initialize benchmark result.

if [ $commit = true ]; then
    module unload python
    module unload altd
    module swap PrgEnv-intel PrgEnv-gnu
    module load python_base
    module load mysql
    module load mysqlpython
    python report-benchmark.py initialize
    module unload mysqlpython
    module unload python_base
fi

# Run benchmark.

output=latest-$SLURM_JOB_NAME.txt
module load shifter
srun shifter /bench/pynamic-pyMPI /bench/pynamic_driver.py $(date +%s) | tee $output

# Extract result.

startup_time=$( grep '^Pynamic: startup time' $output | awk '{ print $(NF-1) }' )
import_time=$( grep '^Pynamic: module import time' $output | awk '{ print $(NF-1) }' )
visit_time=$( grep '^Pynamic: module visit time' $output | awk '{ print $(NF-1) }' )
total_time=$( echo $startup_time + $import_time + $visit_time | bc )

# Finalize benchmark result.

if [ $commit = true ]; then
    module load python_base
    module load mysqlpython
    python report-benchmark.py finalize $total_time
fi

