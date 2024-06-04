#!/bin/bash -l
# The -l above is required to get the full environment with modules

#SBATCH --job-name=dropCount
#SBATCH --partition=short
#SBATCH -C xeon
###SBATCH -C epyc
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=250
#SBATCH --mem-per-cpu=3G
#SBATCH --mail-user=big.jimmy@email.com
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=job_%j.out

ulimit -s unlimited
srun --mpi=pmix ./drop_count
module load python
python combineOutputs.py






