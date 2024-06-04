#!/bin/bash
#SBATCH -J MUGA_REF_INFER_GENOS
#SBATCH --mem 100GB
#SBATCH -t 1:00:00

# submit from MUGA_reference_data directory using:
# config=/projects/compsci/vmp/USERS/widmas/MUGA_reference_data/data/GigaMUGA/chrs.txt
# chrs=$(wc -l < ${config})
# sbatch --array=1-${chrs} code/GigaMUGA_Infer_Genotypes_submit.sh

# chromosome config file
config=/projects/compsci/vmp/USERS/widmas/MUGA_reference_data/data/GigaMUGA/chrs.txt

# chromosome
# find the line in the config file
chr=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${config})
echo ${chr}
echo "Running Chromosome ${chr}"

# Performing Background QC on GigaMUGA samples
singularity exec docker://sjwidmay/muga_qc:latest code/GigaMUGA_Infer_Genotypes.R ${chr}
