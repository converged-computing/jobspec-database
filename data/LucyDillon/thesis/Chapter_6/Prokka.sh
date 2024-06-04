#!/bin/sh
#SBATCH --time=7-23:59:59
#SBATCH --partition=bio-compute,lowpri
#SBATCH --job-name=prokka_EC
#SBATCH --array=1-96%100
#SBATCH --mail-user=ldillon05@qub.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --error=Prokka_EC-%A-%a.err
cd /mnt/scratch2/users/40309916/E_coli_genomes/genomes/prokka_analysis
module load apps/anaconda3
source activate /mnt/scratch2/igfs-anaconda/conda-envs/prokka

file_list=$(ls *.fna | sed -n ${SLURM_ARRAY_TASK_ID}p)
for file_list in `ls *.fna | sed -n $(expr $(expr ${SLURM_ARRAY_TASK_ID} \* 100) - 99),$(expr ${SLURM_ARRAY_TASK_ID} \* 100)p`; do
  prokka --outdir prokka_output/${file_list%.*} --prefix ${file_list%.*} $file_list;
done
