#!/bin/bash
#SBATCH --partition=general-compute --qos=general-compute
#SBATCH --time=12:00:00
#SBATCH --job-name=RS-03684230_methyl_extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8000
##SBATCH --requeue

#SBATCH --output=logs_bismark/%x.%j.out
#SBATCH --error=logs_bismark/%x.%j.err

module load gcc
module load samtools
conda init bash
conda activate bismark

cd $PBS_O_WORKDIR 

bismark_methylation_extractor \
--bedGraph /projects/rpci/joyceohm/pnfioric/RRBS_samfile_links/RS-03684230_001_val_1.fq_trimmed_bismark_bt2_pe.sam \
-o /panasas/scratch/grp-joyceohm/rrbs_methyl_extract/methyl_extract_RS-03684230