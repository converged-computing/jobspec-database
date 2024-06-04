#!/bin/bash -login 
#PBS -l nodes=1:ppn=1,walltime=70:59:00,mem=8gb 
#PBS -N pmf_zzz
#PBS -l feature='intel16'
#PBS -j oe

qstat -f $PBS_JOBID
df -h /tmp
cp -r $PBS_O_WORKDIR /tmp/$PBS_JOBID
df -h /tmp/$PBS_JOBID

cd /tmp/$PBS_JOBID


module load MATLAB-compiler/R2016a
./GARF_MT_EXE_pro_dockedligand /mnt/scratch/bansalnu/MT_conf_sort/xxx_yyy/

df -h /tmp/$PBS_JOBID
rm -R /tmp/$PBS_JOBID
