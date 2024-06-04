#!/bin/bash 
#PBS -j oe 
#PBS -q gpu
#PBS -l nodes=1:ppn=1:gpus=1,walltime=300:00:00
#! Mail to user if job aborts
#PBS -m a

fname=$INPUT
rstfile=$RESTART
module load apps/amber-16
module load cuda/toolkit/7.5.18
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
#nvidia-smi
#./get_X_GPUs.sh 2

#rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

#export CUDA_VISIBLE_DEVICES=0,1

# Run the executable
pmemd.cuda  -O -i $fname.in -p 2agy_final_min.prmtop \
              -c $rstfile -x $fname.mdcrd -r $fname.rst -o $fname.out -inf $fname.info

