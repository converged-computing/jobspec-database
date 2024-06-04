#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name="Singularity install VEP annotator"
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G
module load any/singularity/3.7.3
module load squashfs/4.4

#singularity pull --name vep.sif docker://ensemblorg/ensembl-vep

singularity exec vep.sif INSTALL.pl -c $HOME/vep_data -a c -s homo_sapiens -y GRCh37 -n --CACHE_VERSION 108
