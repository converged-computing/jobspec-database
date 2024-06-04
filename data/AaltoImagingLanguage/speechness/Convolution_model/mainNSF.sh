#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=6G
# srun ./run_mainNSF_1.sh /appl/math/matlab/2014a #matlab version should match to the one used in submit_run.sh(run.sh) where we compile code.
srun ./run_mainNSF_1.sh /appl/math/matlab/R2014a
