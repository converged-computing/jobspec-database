#!/usr/bin/env bash
#PBS -l select=1:ncpus=48:mpiprocs=48:mem=800gb
#PBS -q bigmem
#PBS -W group_list=bigmemq
#PBS -l walltime=48:00:00
#PBS -o /mnt/lustre/users/fkebaso/hippo/hcamelina_male/results/braker/spades_restart/braker.out
#PBS -e /mnt/lustre/users/fkebaso/hippo/hcamelina_male/results/braker/spades_restart/braker.err
#PBS -m abe
#PBS -M fredrickkebaso@gmail.com
#PBS -N braker_spades_hc_restart

set -eu 

# ---------------- braker restart after augustus has finished training  ----------------

# Option 3: starting BRAKER after AUGUSTUS training

# ----------------Modules------------------

# Load Singularity module

module load chpc/singularity/3.5.3

# Create output variables

# Set path to BRAKER Singularity container
export BRAKER_SIF="/apps/chpc/bio/BRAKER-3.0.3/braker3.sif"

# Set the base directory
basedir="/mnt/lustre/users/fkebaso/hippo/hcamelina_male"
threads=48
# Set the working directory for the new run
new_work_dir="${basedir}/results_old/braker/spades_restart"

# Set the directory for the previous run
oldDir="${basedir}/results_old/braker/spades"

# Check if the working directory exists
if [ ! -d $new_work_dir ]; then
    echo "Working directory $new_work_dir does not exist. Creating directory..."
    mkdir -p $new_work_dir
fi

# Check if the directory of the previous run exists
if [ ! -d $oldDir ] ; then
  echo "ERROR: Directory (with contents) of old BRAKER run $oldDir does not exist, yet." 
else
    species=$(cat $oldDir/braker.log | perl -ne 'if(m/AUGUSTUS parameter set with name ([^.]+)\./){print $1;}')
    echo "Running BRAKER with restart..."
    # Run BRAKER with restart and measure the execution time
    singularity exec -B ${PWD}:${PWD} ${BRAKER_SIF} braker.pl \
    --genome="${oldDir}"/genome.fa \
    --hints="${oldDir}"/hintsfile.gff \
    --skipAllTraining  \
    --species="${species}" \
    --workingdir="${new_work_dir}" \
    --threads "${threads}" > "${new_work_dir}"/braker_restart.log
fi

echo "Braker run completed successfully!"


