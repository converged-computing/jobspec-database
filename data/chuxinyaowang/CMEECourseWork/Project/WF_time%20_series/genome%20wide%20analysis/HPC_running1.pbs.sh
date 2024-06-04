
#!/bin/bash

#PBS -l walltime=24:00:00

#PBS -l select=1:ncpus=128:ompthreads=128:mem=920g

#PBS -N ROC_T1000dt5

 

module load matlab/R2021a 2>&1>/dev/null ## to suppress the module load output


 

cd $PBS_O_WORKDIR

 

# MFILE="/rds/general/user/cw422/home/project/code"

MATLAB_SCRIPT="/rds/general/user/cw422/home/project/code/ROC_simulations.m"
GENOME_INPUT="/rds/general/user/cw422/home/project/code/HPC_input1.mat"


echo "DBG: pwd is : $PWD"

echo "DBG: mfile is : $(file $MATLAB_SCRIPT)"

echo "DBG: matlab is : $(which matlab)"

export MATLABPATH=../fminsearchbnd:$MATLABPATH


 

echo "DBG: running matlab.. "

matlab -nosplash -nodisplay -logfile /rds/general/user/cw422/home/project/results/fixmu/results1/matrun_out1.log -r "addpath(genpath('/rds/general/user/cw422/home/project/code')); load('$GENOME_INPUT'); [xx_2d,xx_2d_neg,xx_2d_pos,time_points,type,Nopt,sopt,muopt,d0,s_neg,s_pos]=ROC_simulations(N,mu,sn,sp,fr,L,T,dt,ns,I,Dt,N0,mu0,samplcorrections,initialparams,sinitopt,null,fine, prior, plotfig,suppressoutput,CI,boundary,optimmethod); save('/rds/general/user/cw422/home/project/results/fixmu/results1/output1.mat', 'xx_2d', 'xx_2d_neg', 'xx_2d_pos', 'time_points', 'type', 'Nopt', 'sopt', 'muopt', 'd0', 's_pos', 's_neg'); exit;" && echo "matlab run OK"
