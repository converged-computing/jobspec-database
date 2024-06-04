#!/bin/bash

## Specify the needed settings from the server 
#SBATCH --nodes=1  # number of nodes
#SBATCH --tasks-per-node=1  # tasks per node
#SBATCH --mem-per-cpu=10G  # amount of memory the job requires, default is 2G

## Assign the name of job, output & error files
#SBATCH --job-name=pmf_noGUI_try
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=pmf_noGUI_try_%N_%j.out # output file
#SBATCH --error=pmf_noGUI_try_%N_%j.err # error file

## Email info for updates from Slurm
#SBATCH --mail-type=BEGIN,END,FAIL # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=tzhang23@gmu.edu

## Specify the maximum of time in need
#SBATCH --time=03-00:00  # Total time needed for job: Days-Hours:Minutes

## create an array of jobs
#SBATCH --array=1-150

#load modules
module load r  #will load default r version
# module load dosbox
module load singularity

# Define the DOS command to be used
DOS_COMMAND="ME-2 PMF_bs_6f8xx_sealed_GUI_MOD.ini"

# Calculate the Cluster and Factor numbers based on the array index
Cluster_number=$(( ($SLURM_ARRAY_TASK_ID - 1) / 6 + 1 ))
Factor_number=$(( ($SLURM_ARRAY_TASK_ID - 1) % 6 + 6 ))

# Set the directory path for the Cluster and Factor folders
cd "CSN_CMD_txt/Cluster_${Cluster_number}/Factor_${Factor_number}"

## Run the tasks

# 1. PMF base model analysis
cp iniparams_base.txt iniparams.txt

# Run the MS-DOS command using Singularity and DOSBox
# Updated  DOSBox command to mount  current working directory as the virtual C: drive with mount c .
# singularity exec dosbox_container.sif dosbox -c "mount c ${input_dir}" -c "c:" 
singularity exec ../dosbox_container.sif dosbox -c "mount c ." -c "c:" -c "$DOS_COMMAND" -c "exit"

rm iniparams.txt

# 2. Analyze the output .txt file, generate the new value for numoldsol, and replace it in other iniparams.txt series usi$
mv CSN_C_${Cluster_number}_F_${Cluster_number}_.txt CSN_C_${Cluster_number}_F_${Factor_number}_base.txt
Rscript /projects/HAQ_LAB/tzhang/pmf_no_gui/file_try/minQ_Task_numoldsol.R CSN_C_${Cluster_number}_F_${Factor_number}_base.txt

# 3. PMF BS, DISP, and BS-DISP analyses 
for param_file in iniparams_BS.txt iniparams_DISP.txt #iniparams_before_dualc.txt iniparams_BS_DISP.txt
do
cp ${param_file} iniparams.txt
$DOS_COMMAND
rm iniparams.txt
done