#!/bin/bash
##########################################################################
#
#  Author:    Amy Stringer
#  Created:   2022-27-07
#
##########################################################################

#PBS -N LST7T_prepro
#PBS -l mem=43gb,walltime=10:00:00,ncpus=15
#PBS -m abe
#PBS -M amy.stringer@qimrberghofer.edu.au
#PBS -o /mnt/lustre/working/lab_lucac/shared/projects/LST7T/scratch/batch
#PBS -e /mnt/lustre/working/lab_lucac/shared/projects/LST7T/scratch/batch
#PBS -k eod
#PBS -J 0-65

# set project directory 
proj_dir=/mnt/lustre/working/lab_lucac/shared/projects/LST7T/

# get subject from list
mapfile -t subj_array < ${proj_dir}code/subject_list_fmri.txt
subj=${subj_array[$PBS_ARRAY_INDEX]}
echo "Current subject: " ${subj}

fmrip_con=/working/genomeinfo/share/containers/singularity/fmriprep-21.0.1.simg

# paths
bids_dir=${proj_dir}data/bids/
out_dir=${proj_dir}data/derivatives/fmriprep/
work_dir=${proj_dir}scratch/fmriprep/${subj}

# cd into code
cd ${proj_dir}code

# load modules
module load singularity/3.7.1

# run proxy script for internet access
source ~/.proxy

# run fmriprep through singularity
singularity run --cleanenv -B /home ${fmrip_con} ${bids_dir} ${out_dir} participant \
--participant-label ${subj} \
--work-dir ${work_dir} \
--output-spaces MNI152NLin2009cAsym \
--output-layout bids \
--n_cpus 12 \
--mem-mb 30000 \
--low-mem \
--resource-monitor \
--fs-no-reconall \
--fs-license-file /software/freesurfer/freesurfer-6.0.1/license.txt
# --skull-strip-t1w skip
# --use-syn-sdc error 