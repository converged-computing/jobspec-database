#!/bin/bash
#PBS -N DMT_data_PCB_group_preprocess
#PBS -o /headnode2/mtan8991/BrainVortexToolbox-main/data/DMT_data/PCB_group/Preprocessed_Data_log/$PBS_JOBNAME.o$PBS_ARRAY_INDEX
#PBS -e /headnode2/mtan8991/BrainVortexToolbox-main/data/DMT_data/PCB_group/Preprocessed_Data_log/$PBS_JOBNAME.e$PBS_ARRAY_INDEX
#PBS -j oe
##PBS -q yossarian
#PBS -l select=1:ncpus=1:mem=25GB
#PBS -l walltime=20:00:00
#PBS -m n
##PBS -M mtan8991@uni.sydney.edu.au
#PBS -V
#PBS -J 1-20

set -x
cd /headnode2/mtan8991/BrainVortexToolbox-main/
logfile_path="/headnode2/mtan8991/BrainVortexToolbox-main/data/DMT_data/PCB_group/Preprocessed_Data_log/${PBS_JOBNAME}_$PBS_ARRAY_INDEX.log"


mkdir -p /headnode2/mtan8991/BrainVortexToolbox-main/data/DMT_data/PCB_group/Preprocessed_Data_log/


file1=pre_process

# matlab2019b -singleCompThread -nodisplay -r "${file1}($PBS_ARRAY_INDEX); exit "
matlab2019b -singleCompThread -nodisplay -r "${file1}($PBS_ARRAY_INDEX); exit" -logfile "$logfile_path"


exit
