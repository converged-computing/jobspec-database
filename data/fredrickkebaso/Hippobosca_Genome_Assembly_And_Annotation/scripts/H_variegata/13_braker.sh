#!/bin/bash
#PBS -l select=1:ncpus=48:mpiprocs=48:mem=800gb
#PBS -q bigmem
#PBS -W group_list=bigmemq
#PBS -l walltime=48:00:00
#PBS -o /mnt/lustre/users/fkebaso/hippo/hvariegata_female/results/braker/velvet/braker.out
#PBS -e /mnt/lustre/users/fkebaso/hippo/hvariegata_female/results/braker/velvet/braker.err
#PBS -m abe
#PBS -M fredrickkebaso@gmail.com
#PBS -N braker_velvet_hv

set -eu 

# ---------------- braker Gene Prediction ----------------

# Ab inition gene prediction

# ----------------Modules------------------

# Load Singularity module

module load chpc/singularity/3.5.3

# Create output variables
echo "Creating output variables..."
basedir="/mnt/lustre/users/fkebaso/hippo/hvariegata_female"
results="${basedir}/results/braker/velvet" 
genome="${basedir}/results/repeatmasker/velvet/hvariegata_f_velvet_genome.fa"
protein_file="${basedir}/results/braker/protein_db/Glossina_proteins.fa"
threads=48
mincontig=100   
email="fredrickkebaso@gmail.com"

# Set path to BRAKER Singularity container
export BRAKER_SIF="/apps/chpc/bio/BRAKER-3.0.3/braker3.sif"

# Check whether Singularity is installed
echo "Checking for Singularity..."
if ! command -v singularity &> /dev/null
then
    echo "Singularity could not be found."
    exit 1
fi

# Create new output directory
echo "Creating new output directory..."

mkdir -p "${results}"

# Create empty output files for braker.out and braker.err
echo "Creating empty output files..."
touch "${results}/braker.err" "${results}/braker.out"

# ---------------- Run -----------------------

# Run BRAKER using Singularity
echo "Running BRAKER..."
singularity exec -B ${PWD}:${PWD} ${BRAKER_SIF} braker.pl \
--genome="${genome}" \
--prot_seq="${protein_file}" \
--threads "${threads}" \
--workingdir="${results}" \
--min_contig="${mincontig}" \
--augustus_args="--species=fly" \
--softmasking \
--gff3 \
--makehub \
--email "${email}" 

echo "BRAKER completed successfully !!!"
