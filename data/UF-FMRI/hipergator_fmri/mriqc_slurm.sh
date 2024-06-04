#!/bin/bash
#SBATCH --account=stevenweisberg
#SBATCH --qos=stevenweisberg-b
#SBATCH --job-name=mriqc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stevenweisberg@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --time=12:00:00
#SBATCH --output=mriqc_%j.out
#pwd; hostname; date

module load singularity

#set up the following paths
BIDS_dir=/blue/stevenweisberg/stevenweisberg/MVPA_ARROWS/
MRIQC_output_dir=/blue/stevenweisberg/stevenweisberg/MVPA_ARROWS/derivatives/mriqc
code_dir=/blue/stevenweisberg/stevenweisberg/MVPA_ARROWS/code/hipergator
MRIQC_singularity=/blue/stevenweisberg/stevenweisberg/hipergator_neuro/mriqc/mriqc_latest.sif

# loops through subjects
for SUB in {107..128}
do


  # Skip 111 and 119
  [ "$SUB" -eq 111 ] && continue
  [ "$SUB" -eq 119 ] && continue

  # loops through sessions. Get rid of this loop entirely if there is only one session. Also get rid of '-s 0${ses}' in step 2
  for ses in 1
  do

# running dataset for single participants
  singularity run --cleanenv $MRIQC_singularity $BIDS_dir $MRIQC_output_dir participant --participant-label $SUB --hmc-fsl --fd_thres 2 --work-dir $MRIQC_output_dir --float32 group

  done
done
