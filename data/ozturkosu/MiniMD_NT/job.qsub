#!/bin/bash --login
#PBS -l nodes=1,walltime=0:10:00
#PBS -l mem=4gb
#PBS -q main
#PBS -m abe
#PBS -N miniMD.lj.131K 
#PBS -o miniMD.lj.131K 

#module swap GNU Intel
#module load LAMMPS
#cd ${PBS_O_WORKDIR}
#rm -f dump/*
#rm -f dump.tar.gz

cd ${PBS_O_WORKDIR}
module swap GNU Intel
export OMP_NUM_THREADS=20

time mpirun -np 1 ./miniMD_intel -t 1 -hma 0 

#tar -cf dump.tar dump
#gzip -9 dump.tar

#qstat -f ${PBS_JOBID}
