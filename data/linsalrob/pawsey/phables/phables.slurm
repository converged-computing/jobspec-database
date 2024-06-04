#!/bin/bash
#SBATCH --job-name=phables
#SBATCH --account=pawsey1018
#SBATCH --time=1-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o phables-%A_%a.out
#SBATCH -e phables-%A_%a.err

set -euo pipefail

ACCLIST="SRR_Acc_List.txt"
SRR=$(head -n $SLURM_ARRAY_TASK_ID $ACCLIST | tail -n 1)

module load singularity/3.11.4-slurm
singularity exec --bind /scratch/pawsey1018/edwa0468/tmp/conda:/conda,$PWD/phables/$SRR:/phables,$HOME/gurobi.lic:/opt/gurobi/gurobi.lic sif/phables_v0.6_gogo.sif \
	phables run --input /phables/$SRR.graph.gfa --reads /phables/fastq/ --output /phables/phables --threads 32 --prefix $SRR --mincov 1



