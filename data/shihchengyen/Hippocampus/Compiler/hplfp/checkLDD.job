#!/bin/bash

#PBS -P Project_Name_of_Job
#PBS -q matlab
#PBS -l select=1:ncpus=1:mem=10GB
#PBS -j oe
#PBS -N Job_Name
###  -N Job_Name: set filename for standard output/error message.

cd $PBS_O_WORKDIR;   ## This line is needed, do not modify.

##--- Put your exec/application commands below ---
## If your matlab program is < my_matlab_prog.m >.

# matlab -nojvm -nodisplay -nosplash -r eyehplfp
ldd /nfs/app1/common/matlab/R2018a/bin/glnxa64/libQt5WebEngineCore.so.5

