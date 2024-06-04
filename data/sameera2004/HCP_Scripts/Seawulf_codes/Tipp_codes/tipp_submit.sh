#!/bin/bash 

#set the number of nodes and processes per node and wallclock time
#PBS -l nodes=1:ppn=28,walltime=08:00:00

#PBS -q gpu-long

# set name of job
#PBS -N Tipp_nyspi

# mail alert at (b)eginning, (e)nd and (a)bortion of execution
#PBS -m bea

# send mail to the following address
#PBS -M sameera.abeykoon@stonybrookmedicine.edu

# use submission environment 
#PBS -V

# start job from the directory it was submitted (see following)
cd $PBS_O_WORKDIR
#cd /gpfs/projects/VanSnellenbergGroup/nyspi_data

killall matlab
killall MATLAB

module load shared
module load matlab/2017b
module load spm/12

# Set variable values that locate and specify data to process
StudyFolder="/gpfs/scratch/sabeykoon/HCP_data/nyspi" # Location of Subject folders

#sh /gpfs/software/Pipelines-3.24.0/Examples/Scripts/PreFreeSurferPipelineBatch_SB_SpinEcho_FieldMap.sh $StudyFolder $Subjlist>/gpfs/home/sabeykoon/PFsurfer_50027.out 2>&1
#sh /gpfs/software/Pipelines-3.24.0/Examples/Scripts/PreFreeSurferPipelineBatch_XN_B0_${Subjlist}.sh>/gpfs/scratch/sabeykoon/HCP_data/outputs/PFsurfer_${Subjlist}.out 2>/gpfs/scratch/sabeykoon/HCP_data/outputs/PFsurfer_${Subjlist}.error

matlab -nodisplay -r "tipp_nyspi('/gpfs/scratch/sabeykoon/HCP_data/nyspi'); quit" 2> "/gpfs/scratch/sabeykoon/HCP_data/outputs/Tipp_error.log" 1> "/gpfs/scratch/sabeykoon/HCP_data/outputs/Tipp_output.log"

