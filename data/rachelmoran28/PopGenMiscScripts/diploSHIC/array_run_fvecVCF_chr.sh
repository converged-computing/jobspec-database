#!/bin/bash
#SBATCH -p  small,amdsmall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=62gb
#SBATCH -t 2:00:00
#SBATCH -J fvecVCF
#SBATCH --array=0-26


source /home/mcgaughs/rmoran/miniconda3/etc/profile.d/conda.sh
conda activate diplo

#generating feature vector stats
cd /home/mcgaughs/shared/Software/diploSHIC

#vcf="/home/mcgaughs/shared/Datasets/all_sites_LARGE_vcfs/filtered_surfacefish/combined_filtered/246Indvs_RepsMasked/phased_masked/Split_All_Scafolds_VCFs"

#vcf="/home/mcgaughs/shared/Datasets/all_sites_LARGE_vcfs/filtered_surfacefish/combined_filtered/246Indvs_RepsMasked/phased_masked/biallelicSNPs_wInvar_noIndels_phased_masked_ExcludeHet"

vcf="/home/mcgaughs/shared/Datasets/all_sites_LARGE_vcfs/filtered_surfacefish/filtered_snps/HardFilteredSNPs_250Samples"
Pop="Tinaja"

CMD_LIST="HardFiltered_All_Commands_fvecVCF.txt"

CMD="$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${CMD_LIST})"
eval ${CMD}

