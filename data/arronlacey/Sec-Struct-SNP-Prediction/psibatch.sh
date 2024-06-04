#!/bin/bash

#SBATCH --job-name psipred-array
#SBATCH --time 00-06:00 
#SBATCH --output psipred-array-%a.out
#SBATCH -o psibatchout.%J
#SBATCH -e psibatcherr.$J
#SBATCH -n 128
#SBATCH --array=1001
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=8
#SBATCH --mail-user=arronslacey@gmail.com
module load compiler/gnu/4.8.0
module load R/3.2.3

code=${HOME}/Phd/script_dev/rfpipeline.sh

data_file="epsnps_${SLURM_ARRAY_TASK_ID}.fasta"
echo ${data_file}
${code} ${data_file}

#cp -r humvarsnpad* humvarsnpproc/
#rm -r polyid4ssnp*
