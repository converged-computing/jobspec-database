#!/bin/bash

# Download databases for pb_16s_nf
# Akiris Moctezuma  Jun 2023
# See https://github.com/PacificBiosciences/pb-16S-nf
# using un-updated nextflow.config

# Note: this script should be run on a compute node
# qsub s02_download_databases.sh

# PBS directives
#---------------

#PBS -N s02_download_databases
#PBS -l nodes=1:ncpus=4
#PBS -l walltime=00:30:00
#PBS -q half_hour  
#PBS -m abe 
#PBS -M akiris.moc.901@cranfield.ac.uk 

#===============
#PBS -j oe
#PBS -W sandbox=PRIVATE
#PBS -k n
ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID
## Change to working directory
cd $PBS_O_WORKDIR
## Calculate number of CPUs
export cpus=`cat $PBS_NODEFILE | wc -l`
## =============

# Start message
echo "Started"
date
echo ""

# Stop at runtime errors
set -e

# Initialise conda
# module load Miniconda3/4.11.0
# conda init bash
source ~/.bashrc

# Activate conda dedicated environment
conda activate thesis 

# Load Singularity module
module load Singularity/2.6.1-GCC-5.4.0-2.26
echo "Singularity version: "
singularity --version
echo ""

# Download databases
project_folder="/scratch/s394901/thesis"
cd "${project_folder}"/tools/pb-16S-nf

nextflow run main.nf \
--download_db \
-profile singularity

# Completion message
echo "Done"
date

# --- Your code ends here --- #

## Tidy up the log directory
## =========================
rm $PBS_O_WORKDIR/$PBS_JOBID
