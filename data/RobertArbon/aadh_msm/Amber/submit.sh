#!/bin/bash 
#PBS -N long_equil 
#PBS -j oe 
#PBS -q gpu
#PBS -l nodes=1:ppn=1:gpus=2,walltime=1:00:00
#! Mail to user if job aborts
#PBS -m a

fname=equi_traj
module load apps/amber-16-u6
###############################################################
### You should not have to change anything below this line ####
###############################################################

#! change the working directory (default is home directory)

cd $PBS_O_WORKDIR

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo PBS job ID is $PBS_JOBID
echo This jobs runs on the following GPUs:
echo `cat $PBS_GPUFILE | uniq`


#Run nvidia-smi on visible GPUs
nvidia-smi
./get_X_GPUs.sh 2

rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

echo $CUDA_VISIBLE_DEVICES 

#! Run the executable
#pmemd.cuda  -O -i $fname.in -p ../common/2agy_final_min.prmtop \
#              -c ../common/equil6.rst -x $fname.mdcrd -r $fname.rst -o $fname.out -inf $fname.info
#
