#!/bin/bash -login
#PBS -N romfem
#PBS -j oe
#PBS -l nodes=1:ppn=12
#PBS -l walltime=04:59:20
#PBS -l mem=32gb
#PBS -V
#PBS -o output/output.txt
#PBS -q all

# Array 1-n%k -> n jobs with maximum of k jobs simultaneously
#PBS -t 1-100%10

echo "**************"
echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
echo "Job ran on:" $(hostname)
echo "Current working directory is now: " `pwd`
echo "Starting job at `date`"
echo "**************"
export MATLAB_LOG_DIR=/tmp
module load MATLAB

command="main($PBS_ARRAYID,@random_amplitude)"
/usr/bin/time -v  matlab -nodisplay -nodesktop -nosplash -nosoftwareopengl -r $command

echo "**************"
echo "JOB run completed at `date`"
echo "**************"
module purge

# fquota
# lquota
# showq -u $USER
# qstat -u $USER
# qselect -u $USER | xargs qdel

# ls -lh
# qsub -I -X -l nodes=1:ppn=24 -l walltime=48:00:00 -l mem=64gb

# qstat -t -u $USER
# http://docs.adaptivecomputing.com/torque/4-2-7/help.htm#topics/commands/qsub.htm#-t
