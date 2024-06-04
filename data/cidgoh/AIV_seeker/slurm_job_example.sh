#!/bin/bash
#SBATCH --account=rrg-whsiao-ab
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=8g
#SBATCH --cpus-per-task=4
#SBATCH --job-name=aiv_seeker
#SBATCH --output=/scratch/djhyq557/aiv_sim/logs/%j.out
#SBATCH --error=/scratch/djhyq557/aiv_sim/logs/%j.err




 module load nextflow/22.04.3
 nextflow run main.nf --input /scratch/djhyq557/test_aiv/AIV_seeker/demo_data/samplesheet.csv -profile singularity,slurm
