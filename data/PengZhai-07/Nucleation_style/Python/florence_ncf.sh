#!/bin/sh
#SBATCH --job-name florence_ncf
#SBATCH --mail-user=yaolinm@umich.edu
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6500
#SBATCH --time=4:00:00
#SBATCH --account=zspica0
#SBATCH --partition=standard
#SBATCH --array=105-112

module load python3.9-anaconda
source activate eqcorrscan
srun python3 florence_ncf.py $SLURM_ARRAY_TASK_ID