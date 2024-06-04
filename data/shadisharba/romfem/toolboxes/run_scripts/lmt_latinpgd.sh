#!/bin/bash -login
#PBS -N romfem
#PBS -j oe
#PBS -l nodes=1:ppn=24
#PBS -l walltime=04:59:20
#PBS -l mem=32gb
#PBS -V
#PBS -o output/output.txt
#PBS -P lmt

#PBS -J 1-100

echo "**************"
echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
echo "Job ran on:" $(hostname)
echo "Current working directory is now: " `pwd`
echo "Starting job at `date`"
echo "**************"
export MATLAB_LOG_DIR=/tmp
module load matlab/r2018a

command="main($PBS_ARRAY_INDEX,@random_amplitude)"
/usr/bin/time -v  matlab -nodisplay -nodesktop -nosplash -nosoftwareopengl -r $command

echo "**************"
echo "JOB run completed at `date`"
echo "**************"
module purge

# ssh alameddin@fusion.centralesupelec.fr
# 10GB    /home/username     $HOME
# 200GB   /workdir/username  $WORKDIR
