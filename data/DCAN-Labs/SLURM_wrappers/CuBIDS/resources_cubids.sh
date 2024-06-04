#!/bin/bash -l
#SBATCH -J cubids-validate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --tmp=10gb
#SBATCH -t 00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<YOUR-EMAIL>@email.com
#SBATCH -o output_logs/cubids_%A_%a.out
#SBATCH -e output_logs/cubids_%A_%a.err
#SBATCH -A miran045

cd run_files.cubids

module load singularity

file=run${SLURM_ARRAY_TASK_ID}

bash ${file}