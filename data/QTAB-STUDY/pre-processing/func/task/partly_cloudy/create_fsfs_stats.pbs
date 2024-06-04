#!/bin/bash
#PBS -A UQ-QBI
#PBS -N FEAT_array
#PBS -l select=1:ncpus=1:mem=6GB
#PBS -l walltime=03:00:00
#PBS -J 1-57

module load singularity/2.4.2
code_dir=/home/uqlstri1/bin/qtab/tfMRI/partly_cloudy

participants=`cat ${code_dir}/3_FEAT_list.txt`
participantsArray=($participants)
imgID=`echo ${participantsArray[${PBS_ARRAY_INDEX}]}`

cd ${code_dir}
pwd
echo ${imgID}
cat ${code_dir}/template_stats.fsf | sed s/QTABID/${imgID}/ > ${code_dir}/${imgID}_stats.fsf

singularity exec -B /home/uqlstri1/bin/qtab/tfMRI/partly_cloudy:/code -B /30days/uqlstri1/qtab_analysis/tfMRI/partly_cloudy:/qtab_output /90days/uqlstri1/containers/debian_9.4_mrtrix3_fsl5.img feat ${code_dir}/${imgID}_stats.fsf
