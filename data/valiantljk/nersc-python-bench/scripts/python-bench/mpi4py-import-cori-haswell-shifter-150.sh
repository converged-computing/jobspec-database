#!/bin/bash
#SBATCH --account=nstaff
#SBATCH --constraint=haswell
#SBATCH --image=docker:rcthomas/nersc-python-bench:0.1.6
#SBATCH --job-name=mpi4py-import-cori-haswell-shifter-150
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rcthomas@lbl.gov
#SBATCH --nodes=150
#SBATCH --ntasks-per-node=32
#SBATCH --output=logs/mpi4py-import-cori-haswell-shifter-150-%j.out
#SBATCH --partition=regular
#SBATCH --qos=normal
#SBATCH --time=30

# Configuration.

commit=true

# Initialize benchmark result.

if [ $commit = true ]; then
    shifter python /usr/local/bin/report-benchmark.py initialize
fi

# Run benchmark.

export OMP_NUM_THREADS=1

output=tmp/latest-$SLURM_JOB_NAME.txt
srun -c 2 shifter python /usr/local/bin/mpi4py-import.py $(date +%s) | tee $output

# Finalize benchmark result.

if [ $commit = true ]; then
    shifter python /usr/local/bin/report-benchmark.py finalize $( grep elapsed $output | awk '{ print $NF }' )
fi

