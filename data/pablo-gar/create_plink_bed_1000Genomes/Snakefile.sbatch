#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH -e /scratch/users/paedugar/somaticMutationsProject/clusterFiles/plink.err
#SBATCH -o /scratch/users/paedugar/somaticMutationsProject/clusterFiles/plink.out
#SBATCH --mem=4G
#SBATCH --partition=hbfraser,hns,normal
#SBATCH --time=4:00:00


####
# MODS

#PATH=$PATH:$HOME/bin:$HOME/vcftools_0.1.13/bin:$HOME/bcftools_1.6/bin
PATH=$HOME/bin:$PATH:$HOME/.local/bin:$HOME/gatk-4.0.3.0:$HOME/samtools_1.6/bin
export PATH

MODULEPATH=$MODULEPATH:/share/PI/hbfraser/modules/modules
export MODULEPATH

# END MODS
######

module load anaconda3
source activate fraserconda

cd ~/scripts/FraserLab/make_plink_1000genomes_phase_3_GRCh38/

snakemake --nolock --printshellcmds --keep-going --cluster-config ./cluster.json --cluster-status jobState --jobs 500 --cluster "./submit.py"

