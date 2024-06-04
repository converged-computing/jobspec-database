#!/bin/bash
#SBATCH --job-name=cbpm          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:50:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/mmfs1/scratch/kurkela/output/cbpm.out
#SBATCH --mail-type=end
#SBATCH --mail-user=kurkela@bc.edu

cd /mmfs1/data/kurkela/Desktop/CamCan/code
module load matlab
matlab -nodisplay -nosplash -r "D_cbpm('memoryability', 'all', 'default', 0.01, 'none');"
